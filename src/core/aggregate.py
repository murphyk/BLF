#!/usr/bin/env python3
"""aggregate.py — Compute aggregation variants from multi-trial forecasts.

Materializes different aggregation methods into the forecast JSON files so that
eval.py can read them directly. Each variant is stored in
trial_stats.aggregations["{method}:{n}"] = forecast_value.

Methods:
    mean:       Plain mean of n trials.
    shrink:     Std-based shrinkage toward 0.5 (James-Stein). Default for predict.py.
    llm-agg:    LLM reads K trial reasoning traces, resolves disagreements.
                Writes to experiments/forecasts_raw/{config}_aggregated/ (separate files).
                See Section 7.2 of Karger et al. (2025), arXiv:2511.07678.

Usage:
    python3 src/aggregate.py --xid xid-aibq2                                    # all standard variants
    python3 src/aggregate.py --xid xid-aibq2 --variants "mean:1,mean:5,shrink:5"
    python3 src/aggregate.py --xid xid-aibq2 --method llm-agg --max-workers 20

After aggregating, reference variants in the xid's eval field:
    "eval": ["pro-high[shrink:5]", "pro-high[mean:5]", "pro-high[mean:1]"]

Then run eval:
    python3 src/eval.py --xid xid-aibq2 --add-calibration
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.predict import load_exam, load_questions_from_exam


# ---------------------------------------------------------------------------
# Mean/shrinkage aggregation (stored in forecast JSON)
# ---------------------------------------------------------------------------

_SHRINKAGE_FLOOR = 0.3
_SHRINKAGE_SCALE = 0.7


def _aggregate_ps(ps_list, method, shrinkage=None):
    """Aggregate a list of forecasts. Shrinkage operates in logit space."""
    arr = np.array(ps_list)
    if method == "shrink" and len(arr) > 1:
        logits = np.log(np.clip(arr, 0.001, 0.999) / (1 - np.clip(arr, 0.001, 0.999)))
        logit_bar = float(np.mean(logits))
        std_logit = float(np.std(logits))
        if shrinkage is None:
            a = max(_SHRINKAGE_FLOOR, 1.0 - _SHRINKAGE_SCALE * std_logit)
        else:
            a = shrinkage
        return float(1 / (1 + np.exp(-a * logit_bar)))
    return float(np.mean(arr))


def compute_variants(config_name, questions, variants, shrinkage=None):
    """Compute aggregation variants and store in each forecast JSON.

    variants: list of (method, n) tuples, e.g. [("mean", 1), ("mean", 5), ("shrink", 5)]
    Stores results in trial_stats.aggregations["{method}:{n}"] = forecast_value.
    Also stores the expected score for each variant.
    """
    n_updated = 0
    for q in questions:
        source = q.get("source", "unknown")
        qid = q.get("id", "unknown")
        safe_id = re.sub(r'[/\\:]', '_', str(qid))

        fc_path = os.path.join("experiments", "forecasts_raw", config_name,
                                source, f"{safe_id}.json")
        if not os.path.exists(fc_path):
            continue
        with open(fc_path) as f:
            fc = json.load(f)

        ts = fc.get("trial_stats")
        if not ts or ts.get("n_trials", 1) <= 1:
            continue

        trial_fcs = ts.get("forecasts", {})
        all_ps = {int(t): p for t, p in trial_fcs.items()}
        trial_nums = sorted(all_ps.keys())
        K = len(trial_nums)

        outcome = fc.get("resolved_to")
        if isinstance(outcome, list):
            outcome = outcome[0] if outcome else None
        if outcome is not None:
            outcome = float(outcome)

        aggs = ts.get("aggregations", {})

        for method, n in variants:
            n_use = min(n, K)
            key = f"{method}:{n_use}"

            # Compute across all C(K,n) subsets
            subsets = list(combinations(trial_nums, n_use))
            if len(subsets) > 200:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(subsets), 200, replace=False)
                subsets = [subsets[i] for i in idx]

            subset_ps = [_aggregate_ps([all_ps[t] for t in subset], method, shrinkage)
                         for subset in subsets]
            agg_forecast = float(np.mean(subset_ps))

            entry = {"forecast": agg_forecast, "n_subsets": len(subsets)}

            # Compute expected scores if outcome is known
            if outcome is not None:
                def _brier(p, o): return (p - o) ** 2
                def _metaculus(p, o): return 100 * (1 + math.log2(
                    max(0.001, min(0.999, p if o >= 0.5 else (1 - p)))))

                entry["expected_brier"] = float(np.mean(
                    [_brier(p, outcome) for p in subset_ps]))
                entry["expected_metaculus"] = float(np.mean(
                    [_metaculus(p, outcome) for p in subset_ps]))

            aggs[key] = entry

        ts["aggregations"] = aggs
        fc["trial_stats"] = ts

        with open(fc_path, "w") as f:
            json.dump(fc, f, indent=2)
        n_updated += 1

    return n_updated


# ---------------------------------------------------------------------------
# LLM aggregator (writes to separate directory)
# Based on Section 7.2 of Karger et al. (2025), arXiv:2511.07678
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """\
You are an expert meta-forecaster. You have been given a binary forecasting \
question along with {n_trials} independent forecasts from different runs of \
an AI forecasting agent. Each forecast includes a probability and reasoning.

Your task:
1. Review all {n_trials} forecasts and their reasoning.
2. Identify areas of AGREEMENT and DISAGREEMENT.
3. If there are significant disagreements, use your tools (web_search, \
lookup_url, summarize_results) to gather additional evidence.
4. If all forecasts largely agree, submit immediately.
5. Submit your final probability as a well-calibrated synthesis.

Do NOT simply average. Critically evaluate reasoning quality and evidence.

Rules:
- You MUST call submit before step {max_steps}.
- Probabilities must be between 0.05 and 0.95.
- Include an updated_belief with every tool call.
"""


def _build_supervisor_prompt(question, trial_fcs, cutoff_date):
    q = question
    parts = [f"# Question\n{q.get('question', '')}"]
    bg = q.get("background", "")
    rc = q.get("resolution_criteria", "")
    if bg or rc:
        parts.append("## Background\n" + "\n\n".join(x for x in [bg, rc] if x))
    rdate = q.get("resolution_date", "")
    if rdate:
        parts.append(f"## Resolution date\n{rdate}")
    parts.append(f"## Knowledge cutoff\n{cutoff_date}")

    parts.append(f"## Independent Forecasts ({len(trial_fcs)} trials)")
    for t, fc in trial_fcs:
        p = fc.get("forecast", 0.5)
        reasoning = fc.get("reasoning", "")
        if not reasoning or reasoning.startswith("Agent timed out"):
            bh = fc.get("belief_history", [])
            if bh:
                last = bh[-1]
                ev_for = last.get("evidence_for", [])
                ev_against = last.get("evidence_against", [])
                reasoning = (f"(No submit.) Evidence for: {'; '.join(ev_for[:5])}. "
                             f"Against: {'; '.join(ev_against[:5])}.")
            else:
                reasoning = "(No reasoning available.)"
        if len(reasoning) > 1000:
            reasoning = reasoning[:1000] + "..."
        parts.append(f"### Trial {t}: p = {p:.3f}\n{reasoning}")

    ps = [fc.get("forecast", 0.5) for _, fc in trial_fcs]
    parts.append(f"## Stats\nMean: {np.mean(ps):.3f} | Std: {np.std(ps):.3f}")
    parts.append("## Instructions\nReview, investigate disagreements if needed, submit.")
    return "\n\n".join(parts)


def _run_supervisor_one(question, trial_fcs, config_name, supervisor_config, output_dir, verbose=False):
    import litellm
    from agent.belief_state import BeliefState, compact_belief
    from config.config import AgentConfig
    from agent.tools import get_tool_schemas, dispatch_tool

    litellm.suppress_debug_info = True
    qid = question.get("id", "unknown")
    source = question.get("source", "unknown")
    safe_id = re.sub(r'[/\\:]', '_', str(qid))
    cutoff = question.get("forecast_due_date", "")[:10]

    cfg = AgentConfig(
        name=f"{config_name}_aggregated",
        llm=supervisor_config.get("llm", "openrouter/google/gemini-3.1-pro-preview"),
        max_tokens=supervisor_config.get("max_tokens", 16000),
        reasoning_effort=supervisor_config.get("reasoning_effort"),
        search_engine=supervisor_config.get("search_engine", "brave"),
        max_results_per_search=supervisor_config.get("max_results_per_search", 10),
        max_steps=supervisor_config.get("max_steps", 6),
        question_timeout=supervisor_config.get("question_timeout", 180),
        backtesting=supervisor_config.get("backtesting", True),
    )

    system = SUPERVISOR_SYSTEM_PROMPT.format(n_trials=len(trial_fcs), max_steps=cfg.max_steps)
    user_prompt = _build_supervisor_prompt(question, trial_fcs, cutoff)

    agg_source_dir = os.path.join(output_dir, source)
    os.makedirs(agg_source_dir, exist_ok=True)

    state = BeliefState(p=0.5)
    search_cache = {}
    tool_log = []
    belief_history = [state.to_dict()]
    total_in = total_out = 0
    t0 = time.time()
    deadline = t0 + cfg.question_timeout

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_prompt}]
    submitted = False

    for step in range(cfg.max_steps):
        state.step = step + 1
        if time.time() > deadline:
            tool_log.append({"step": step + 1, "type": "timeout"})
            break
        tools = get_tool_schemas(cfg, source=source)
        try:
            kwargs = dict(model=cfg.llm, messages=messages, tools=tools,
                          max_tokens=cfg.max_tokens,
                          timeout=min(120, max(10, deadline - time.time())),
                          num_retries=2)
            if cfg.reasoning_effort:
                kwargs["reasoning_effort"] = cfg.reasoning_effort
            response = litellm.completion(**kwargs)
        except Exception as e:
            tool_log.append({"step": step + 1, "type": "error", "error": str(e)})
            break

        choice = response.choices[0]
        total_in += response.usage.prompt_tokens
        total_out += response.usage.completion_tokens
        text = choice.message.content or ""
        tool_calls = choice.message.tool_calls

        if not tool_calls:
            tool_log.append({"step": step + 1, "type": "no_tool_call", "text": text})
            break

        tc = tool_calls[0]
        fn_name = tc.function.name
        try:
            fn_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            fn_args = {}

        if verbose:
            print(f"    [{config_name}_agg/{qid}] step {step+1}: {fn_name}")

        try:
            result_text, new_state, meta = dispatch_tool(
                fn_name, fn_args, state, cfg, search_cache, agg_source_dir,
                safe_id, question, cutoff, deadline=deadline)
        except Exception as e:
            result_text = f"Tool error: {e}"
            new_state = state
            meta = {"tool": fn_name, "error": str(e)}

        state = new_state
        state = compact_belief(state, cfg)
        belief_history.append(state.to_dict())
        tool_log.append({"step": step + 1, "type": "tool_call", **meta, "belief_p": state.p})
        messages.append(choice.message)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
        messages.append({"role": "user",
                         "content": f"[Belief state updated]\n{state.to_prompt_str(cfg.max_steps)}"})
        if fn_name == "submit":
            submitted = True
            break

    final_p = max(0.05, min(0.95, state.p))
    reasoning = ""
    for entry in reversed(tool_log):
        if entry.get("tool") == "submit" and "reasoning" in entry:
            reasoning = entry["reasoning"]
            break

    result = {
        "id": qid, "source": source,
        "question": question.get("question", ""),
        "background": question.get("background", ""),
        "resolution_criteria": question.get("resolution_criteria", ""),
        "forecast_due_date": question.get("forecast_due_date", ""),
        "forecast": final_p,
        "resolution_date": question.get("resolution_date", ""),
        "resolved_to": question.get("resolved_to"),
        "reasoning": reasoning or f"Supervisor p={final_p:.3f}",
        "system_prompt": system, "question_prompt": user_prompt,
        "belief_history": belief_history, "tool_log": tool_log,
        "n_steps": state.step, "submitted": submitted,
        "tokens_in": total_in, "tokens_out": total_out,
        "elapsed_seconds": round(time.time() - t0, 1),
        "aggregation_method": "llm-aggregator",
        "trial_forecasts": {t: fc.get("forecast", 0.5) for t, fc in trial_fcs},
        "config": cfg.to_dict(),
    }
    out_path = os.path.join(agg_source_dir, f"{safe_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_llm_aggregator(config_name, questions, ntrials, max_workers=20, verbose=False):
    output_dir = os.path.join("experiments", "forecasts_raw", f"{config_name}_aggregated")
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join("experiments", "forecasts_raw", config_name, "config.json")
    supervisor_config = json.load(open(config_path)) if os.path.exists(config_path) else {}
    supervisor_config.setdefault("max_steps", 6)
    supervisor_config.setdefault("question_timeout", 180)

    n_ok = n_fail = n_skip = 0
    t0 = time.time()

    def process_one(q):
        source = q.get("source", "unknown")
        qid = q.get("id", "unknown")
        safe_id = re.sub(r'[/\\:]', '_', str(qid))
        out_path = os.path.join(output_dir, source, f"{safe_id}.json")
        if os.path.exists(out_path):
            return qid, "skip", None
        trial_fcs = []
        for t in range(1, ntrials + 1):
            tp = os.path.join("experiments", "forecasts_raw", config_name,
                              f"trial_{t}", source, f"{safe_id}.json")
            if os.path.exists(tp):
                with open(tp) as f:
                    fc = json.load(f)
                if fc.get("forecast") is not None:
                    trial_fcs.append((t, fc))
        if len(trial_fcs) < 2:
            return qid, "skip", None
        result = _run_supervisor_one(q, trial_fcs, config_name,
                                     supervisor_config, output_dir, verbose)
        return (qid, "ok", result.get("forecast")) if result else (qid, "fail", None)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_one, q): q for q in questions}
        for future in as_completed(futures):
            qid, status, p = future.result()
            if status == "ok": n_ok += 1
            elif status == "fail": n_fail += 1
            else: n_skip += 1
            if (n_ok + n_fail) % 10 == 0:
                print(f"  [{config_name}_agg] {n_ok + n_fail}/{len(questions) - n_skip}")

    print(f"  [{config_name}_agg] Done: {n_ok} ok, {n_fail} fail, {n_skip} skip, "
          f"{time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_xid(xid):
    path = f"experiments/xids/{xid}.json"
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compute aggregation variants from multi-trial forecasts")
    parser.add_argument("--xid", required=True)
    parser.add_argument("--method", default="mean",
                        choices=["mean", "llm-agg"],
                        help="mean: store variants in forecast JSON. "
                             "llm-agg: run LLM aggregator (writes to _aggregated/)")
    parser.add_argument("--variants", default="mean:1,mean:3,mean:5,shrink:3,shrink:5",
                        help="Comma-separated method:n pairs for mean method "
                             "(default: mean:1,mean:3,mean:5,shrink:3,shrink:5)")
    parser.add_argument("--shrinkage", type=float, default=None,
                        help="Fixed shrinkage value (default: std-based)")
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    xid_data = load_xid(args.xid)
    exam_name = xid_data["exam"]
    configs = xid_data.get("config", [])
    if isinstance(configs, str):
        configs = [configs]

    exam = load_exam(exam_name)
    questions = load_questions_from_exam(exam)

    print(f"Aggregating: {args.xid} (method={args.method})")
    print(f"  Exam: {exam_name} ({len(questions)} questions)")

    from config.config import resolve_config, pprint_path
    for config in configs:
        if config in ("sota", "baseline"):
            continue
        # Resolve delta strings to directory names
        if "/" in config and ":" in config:
            cfg = resolve_config(config)
            config = pprint_path(cfg)

        trial_dirs = sorted(glob.glob(
            os.path.join("experiments", "forecasts_raw", config, "trial_*")))
        if not trial_dirs:
            print(f"  [{config}] no trials, skipping")
            continue

        ntrials = len(trial_dirs)

        if args.method == "mean":
            variants = []
            for part in args.variants.split(","):
                m, n = part.strip().split(":")
                variants.append((m.strip(), int(n.strip())))
            print(f"  [{config}] computing {len(variants)} variants (ntrials={ntrials})")
            n = compute_variants(config, questions, variants, shrinkage=args.shrinkage)
            print(f"    updated {n} forecast files")

        elif args.method == "llm-agg":
            print(f"  [{config}] LLM aggregator (ntrials={ntrials})")
            run_llm_aggregator(config, questions, ntrials,
                               max_workers=args.max_workers, verbose=args.verbose)

    print("\nDone.")
    if args.method == "mean":
        print(f"  Reference variants in eval field like: \"pro-high[mean:1]\"")
    else:
        print(f"  Reference as: \"pro-high_aggregated\" in eval field")


if __name__ == "__main__":
    main()
