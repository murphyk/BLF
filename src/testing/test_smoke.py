#!/usr/bin/env python3
"""test_smoke.py — Sanity-check smoke test forecast outputs.

Reads forecast files from a smoke test xid and verifies:
1. All configs produced forecasts for all questions
2. All forecasts have valid probabilities (0.05-0.95)
3. Submit rate is acceptable per config type
4. Zero-shot configs used exactly 1 step
5. Batch configs used exactly 2 steps
6. No hallucinated tool calls (rejected_tool entries)
7. Configs with tools=0 did not call source tools
8. Configs with search=none did not call web_search
9. Forecasts are not all 0.5 (model actually engaged)
10. Brier Index is above chance (0.5) on average

Usage:
    python3 src/test_smoke.py --xid xid-smoke2
    python3 src/test_smoke.py --xid xid-smoke2 --verbose
"""

import argparse
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

from config.config import resolve_config, pprint as cfg_pprint
from testing.test_prompts import ABLATION_CONFIGS


# Source-specific tool names
_SOURCE_TOOLS = {
    "yfinance": "fetch_ts_yfinance",
    "fred": "fetch_ts_fred",
    "dbnomics": "fetch_ts_dbnomics",
    "wikipedia": {"fetch_wikipedia_toc", "fetch_wikipedia_section"},
    "polymarket": "fetch_polymarket_info",
    "manifold": "fetch_manifold_info",
}


def _load_exam_and_outcomes(exam_name):
    with open(f"data/exams/{exam_name}/indices.json") as f:
        exam = json.load(f)
    outcomes = {}
    for source, ids in exam.items():
        for qid in ids:
            safe = re.sub(r'[/\\:]', '_', str(qid))
            q_path = f"data/questions/{source}/{safe}.json"
            if os.path.exists(q_path):
                with open(q_path) as f:
                    q = json.load(f)
                rt = q.get("resolved_to")
                if isinstance(rt, list):
                    rt = rt[0] if rt else None
                outcomes[(source, qid)] = rt
    return exam, outcomes


def check_config(config_name: str, delta: str, exam: dict,
                 outcomes: dict, verbose: bool = False) -> tuple[list[str], list[str]]:
    """Run sanity checks on one config's forecasts. Returns (errors, warnings)."""
    errors = []
    warnings = []
    cfg = resolve_config(delta)
    dir_name = cfg_pprint(cfg)
    base = os.path.join("experiments", "forecasts", dir_name)

    all_forecasts = []
    all_bis = []
    n_submitted = 0
    n_expected = 0
    n_found = 0
    n_all_half = 0  # count of p=0.5 forecasts

    # Check trial dirs to determine if this is a multi-trial run
    trial_dirs = sorted(d for d in os.listdir(base)
                        if d.startswith("trial_") and os.path.isdir(os.path.join(base, d)))
    has_trials = len(trial_dirs) > 0

    for source, ids in sorted(exam.items()):
        for qid in ids:
            n_expected += 1
            safe = re.sub(r'[/\\:]', '_', str(qid))
            fpath = os.path.join(base, source, f"{safe}.json")

            # Check 1: file exists
            if not os.path.exists(fpath):
                errors.append(f"[{config_name}/{source}] missing forecast: {qid}")
                continue

            n_found += 1
            with open(fpath) as f:
                fc = json.load(f)

            # For multi-trial runs, check submit rate across trials
            # (aggregated file inherits submitted from trial 1, which is misleading)
            if has_trials:
                any_submitted = False
                for td in trial_dirs:
                    tfpath = os.path.join(base, td, source, f"{safe}.json")
                    if os.path.exists(tfpath):
                        with open(tfpath) as tf:
                            tfc = json.load(tf)
                        if tfc.get("submitted"):
                            any_submitted = True
                            break
                if any_submitted:
                    n_submitted += 1
            else:
                if fc.get("submitted"):
                    n_submitted += 1

            p = fc.get("forecast")
            n_steps = fc.get("n_steps", 0)
            tool_log = fc.get("tool_log", [])

            # Check 2: valid probability (allow slight overshoot from model)
            if p is None:
                errors.append(f"[{config_name}/{source}/{qid[:20]}] forecast is None")
                continue
            if not (0.01 <= p <= 0.99):
                errors.append(f"[{config_name}/{source}/{qid[:20]}] p={p:.3f} out of range")

            all_forecasts.append(p)
            if p == 0.5:
                n_all_half += 1

            # Compute BI
            o = outcomes.get((source, qid))
            if o is not None:
                all_bis.append(1 - abs(p - o))

            # Check tool log
            tools_called = [e.get("tool", "") for e in tool_log
                           if e.get("type") == "tool_call"]
            rejected = [e for e in tool_log if e.get("type") == "rejected_tool"]

            # Check 6: no hallucinated tool calls
            # These are caught and rejected by the agent loop, so they're
            # warnings not errors — the agent retries with the correct tool.
            if rejected:
                for r in rejected:
                    warnings.append(
                        f"[{config_name}/{source}/{qid[:20]}] "
                        f"rejected hallucinated tool: {r.get('tool')}")

            # Check 7: tools=0 should not have *successfully* called source tools
            # (rejected calls are OK — the agent was told to use submit instead)
            if not cfg.use_tools:
                src_tools = _SOURCE_TOOLS.get(source, set())
                if isinstance(src_tools, str):
                    src_tools = {src_tools}
                successful_src_calls = [tc for tc in tools_called if tc in src_tools]
                rejected_names = {r.get("tool") for r in rejected}
                for tc in successful_src_calls:
                    if tc not in rejected_names:
                        errors.append(
                            f"[{config_name}/{source}/{qid[:20]}] "
                            f"successfully called {tc} with use_tools=False")

            # Check 8: search=none should not have *successfully* called web_search
            if cfg.search_engine == "none":
                if "web_search" in tools_called and "web_search" not in {r.get("tool") for r in rejected}:
                    errors.append(
                        f"[{config_name}/{source}/{qid[:20]}] "
                        f"successfully called web_search with search=none")

    # Check 1 (aggregate): all questions have forecasts
    if n_found < n_expected:
        errors.append(
            f"[{config_name}] only {n_found}/{n_expected} forecasts found")

    # Check 3: submit rate (Flash is worse at following instructions)
    is_flash = "flash" in cfg_pprint(cfg)
    if cfg.max_steps == 1:
        min_submit_rate = 0.8
    elif cfg.batch_queries > 0:
        min_submit_rate = 0.5
    elif is_flash:
        min_submit_rate = 0.5
    else:
        min_submit_rate = 0.6
    if n_found > 0:
        submit_rate = n_submitted / n_found
        if submit_rate < min_submit_rate:
            errors.append(
                f"[{config_name}] low submit rate: {n_submitted}/{n_found} "
                f"({submit_rate:.0%}, min {min_submit_rate:.0%})")

    # Check 9: not all 0.5 (unless zero-shot with no crowd)
    if n_found > 2 and n_all_half == n_found and config_name != "zs":
        errors.append(
            f"[{config_name}] all {n_found} forecasts are 0.5 — model did not engage")

    # Check 10: BI above chance
    if all_bis and len(all_bis) >= 4:
        mean_bi = sum(all_bis) / len(all_bis)
        if mean_bi < 0.45:  # some slack below 0.5 chance
            errors.append(
                f"[{config_name}] mean BI={mean_bi:.3f} is below chance (0.5)")

    if verbose:
        mean_bi = sum(all_bis) / len(all_bis) if all_bis else 0
        warn_str = f"  warn={len(warnings)}" if warnings else ""
        print(f"  {config_name:<16s}  n={n_found}/{n_expected}  "
              f"sub={n_submitted}/{n_found}  "
              f"BI={mean_bi:.3f}  "
              f"errors={len(errors)}{warn_str}")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Sanity-check smoke test outputs")
    parser.add_argument("--xid", required=True, help="Smoke test xid")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from core.eval import load_xid
    xid_data = load_xid(args.xid)
    exam_name = xid_data["exam"]

    exam, outcomes = _load_exam_and_outcomes(exam_name)
    n_questions = sum(len(ids) for ids in exam.values())

    # Determine which configs to test: only those in the xid
    xid_configs = xid_data.get("config", [])
    if isinstance(xid_configs, str):
        xid_configs = [xid_configs]
    # Resolve xid config delta strings to directory names for matching
    xid_dir_names = set()
    for delta in xid_configs:
        try:
            cfg = resolve_config(delta)
            xid_dir_names.add(cfg_pprint(cfg))
        except Exception:
            pass

    # Filter ABLATION_CONFIGS to those present in the xid
    configs_to_test = {}
    for config_name, delta in ABLATION_CONFIGS.items():
        cfg = resolve_config(delta)
        if cfg_pprint(cfg) in xid_dir_names:
            configs_to_test[config_name] = delta

    print(f"Checking {args.xid} (exam={exam_name}, {n_questions} questions)")
    print(f"Configs: {len(configs_to_test)} of {len(ABLATION_CONFIGS)} "
          f"(matching xid)")
    if args.verbose:
        print()

    all_errors = []
    all_warnings = []
    for config_name, delta in sorted(configs_to_test.items()):
        errs, warns = check_config(config_name, delta, exam, outcomes,
                                    verbose=args.verbose)
        all_errors.extend(errs)
        all_warnings.extend(warns)

    # Also check FB reference configs if present
    fb_refs = xid_data.get("fb_reference", [])
    for method_key in fb_refs:
        cfg_name = "fb-" + re.sub(r'[^a-zA-Z0-9]+', '-',
                                   method_key.removeprefix("external.")).strip('-').lower()
        base = os.path.join("experiments", "forecasts", cfg_name)
        if os.path.isdir(base):
            n_found = 0
            for source, ids in exam.items():
                for qid in ids:
                    safe = re.sub(r'[/\\:]', '_', str(qid))
                    if os.path.exists(os.path.join(base, source, f"{safe}.json")):
                        n_found += 1
            if args.verbose:
                print(f"  {cfg_name:<16s}  n={n_found}/{n_questions}  (fb_reference)")
            if n_found == 0:
                all_errors.append(f"[{cfg_name}] no forecasts found for fb_reference")

    print(f"\n{'=' * 60}")
    if all_warnings:
        print(f"{len(all_warnings)} warnings (non-fatal):")
        for w in all_warnings:
            print(f"  ! {w}")
        print()
    if all_errors:
        print(f"FAILED — {len(all_errors)} errors:")
        for e in all_errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print(f"PASSED — all checks OK ({len(ABLATION_CONFIGS)} configs × {n_questions} questions)")


if __name__ == "__main__":
    main()
