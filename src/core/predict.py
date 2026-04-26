#!/usr/bin/env python3
"""predict.py — Run the agentic forecaster on an exam.

Usage:
    python3 src/predict.py --xid myxid
    python3 src/predict.py --exam my_exam --config flash-nothink-brave-crowd0-tools0
    python3 src/predict.py --exam my_exam --config a,b,c --live  # include unresolved

Reads questions from data/questions/{source}/{id}.json via exam indices.
Output goes to experiments/forecasts_raw/{config_name}/{source}/{id}.json.
Multiple configs are run in parallel (--config a,b,c).
"""

import argparse
import glob
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.agent import run_agent, run_batch_agent, _question_stem
from config.config import AgentConfig, resolve_config


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_config_path(config: str) -> str:
    """Resolve config name to a path: try direct, then experiments/configs/{config}.json."""
    if os.path.exists(config):
        return config
    path = f"experiments/configs/{config}.json"
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Config not found: {config} (tried {path})")


def load_xid(xid: str) -> dict:
    """Load an experiment definition from experiments/xids/{xid}.json."""
    path = f"experiments/xids/{xid}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"XID not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_exam(exam_name: str) -> dict[str, list[str]]:
    """Load exam indices, auto-building from mixture.json if indices.json is absent."""
    indices_path = os.path.join("data", "exams", exam_name, "indices.json")
    mixture_path = os.path.join("data", "exams", exam_name, "mixture.json")
    if not os.path.exists(indices_path):
        if os.path.exists(mixture_path):
            import subprocess
            print(f"  [exam '{exam_name}'] indices.json missing — running make_exam.py...")
            r = subprocess.run([sys.executable, "src/make_exam.py", "--name", exam_name])
            if r.returncode != 0:
                sys.exit(f"ERROR: make_exam.py failed for exam '{exam_name}'")
        else:
            raise FileNotFoundError(
                f"Exam '{exam_name}' not found: missing both {indices_path} and {mixture_path}")
    with open(indices_path) as f:
        return json.load(f)


def load_questions_from_exam(exam: dict[str, list[str]]) -> list[dict]:
    """Load all question JSONs referenced by an exam's indices.

    Returns list of question dicts, each with source and id fields.
    """
    questions = []
    for source, ids in sorted(exam.items()):
        for qid in ids:
            safe_id = re.sub(r'[/\\:]', '_', str(qid))
            path = os.path.join("data", "questions", source, f"{safe_id}.json")
            if not os.path.exists(path):
                print(f"  WARNING: question file not found: {path}")
                continue
            with open(path) as f:
                q = json.load(f)
            questions.append(q)
    return questions


def forecast_path(config_name: str, source: str, qid: str) -> str:
    """Build output path: experiments/forecasts_raw/{config}/{source}/{id}.json."""
    safe_id = re.sub(r'[/\\:]', '_', str(qid))
    return os.path.join("experiments", "forecasts_raw", config_name, source, f"{safe_id}.json")


# ---------------------------------------------------------------------------
# Progress monitor
# ---------------------------------------------------------------------------

class ProgressMonitor:
    """Thread-safe progress tracker that writes a self-refreshing HTML page."""

    def __init__(self, output_dir: str, config_names: list[str], n_questions: int):
        self._lock = threading.Lock()
        self._output_dir = output_dir
        self._config_names = config_names
        self._n_questions = n_questions
        self._t0 = time.time()
        self._done = {c: 0 for c in config_names}
        self._ok = {c: 0 for c in config_names}
        self._fail = {c: 0 for c in config_names}
        self._existing = {c: 0 for c in config_names}
        self._recent = []
        self._stop = threading.Event()
        self._thread = None

    def set_existing(self, config_name: str, n: int):
        with self._lock:
            self._existing[config_name] = n

    def record(self, config_name: str, qid: str, ok: bool, p: float | None = None):
        with self._lock:
            self._done[config_name] = self._done.get(config_name, 0) + 1
            if ok:
                self._ok[config_name] = self._ok.get(config_name, 0) + 1
            else:
                self._fail[config_name] = self._fail.get(config_name, 0) + 1
            entry = f"{config_name} | {qid} | {'p=' + f'{p:.3f}' if p is not None else 'FAIL'}"
            self._recent.append((time.time(), entry))
            self._recent = self._recent[-20:]

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._write_html()

    def _loop(self):
        while not self._stop.is_set():
            self._write_html()
            self._stop.wait(timeout=5)

    def _write_html(self):
        with self._lock:
            elapsed = time.time() - self._t0
            os.makedirs(self._output_dir, exist_ok=True)

            rows = []
            total_done = 0
            total_todo = 0
            for c in self._config_names:
                existing = self._existing.get(c, 0)
                done = self._done.get(c, 0)
                ok = self._ok.get(c, 0)
                fail = self._fail.get(c, 0)
                completed = existing + done
                total_for_config = max(self._n_questions, completed)
                pct = int(completed / total_for_config * 100) if total_for_config else 100
                total_done += completed
                total_todo += total_for_config

                bar_width = min(pct, 100)
                bar_color = "#27ae60" if fail == 0 else "#e67e22"
                rows.append(
                    f'<tr><td style="font-weight:bold;">{_esc(c)}</td>'
                    f'<td style="width:300px;">'
                    f'<div style="background:#eee; border-radius:4px; overflow:hidden;">'
                    f'<div style="width:{bar_width}%; background:{bar_color}; '
                    f'padding:4px 8px; color:white; font-size:12px; white-space:nowrap;">'
                    f'{completed}/{total_for_config}</div></div></td>'
                    f'<td>{ok} ok, {fail} fail</td>'
                    f'<td>{existing} cached</td></tr>'
                )

            total_pct = int(total_done / total_todo * 100) if total_todo else 0

            activity = []
            for ts, entry in reversed(self._recent):
                age = elapsed - (ts - self._t0)
                activity.append(f'<div style="font-size:12px; color:#666;">'
                                f'{age:.0f}s ago: {_esc(entry)}</div>')

            html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>Progress — {total_pct}%</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 800px; margin: 0 auto; padding: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
</style></head><body>
<h1>Run Progress</h1>
<p>Elapsed: {elapsed:.0f}s | Overall: {total_done}/{total_todo} ({total_pct}%)</p>
<table>
<tr><th>Config</th><th>Progress</th><th>Status</th><th>Cached</th></tr>
{"".join(rows)}
</table>
<h2 style="margin-top:24px;">Recent Activity</h2>
{"".join(activity) if activity else '<div style="color:#999;">Waiting...</div>'}
</body></html>"""

            path = os.path.join(self._output_dir, "progress.html")
            with open(path, "w") as f:
                f.write(html)


def _esc(text):
    """HTML-escape for progress page."""
    import html
    return html.escape(str(text)) if text else ""


# ---------------------------------------------------------------------------
# Run one config
# ---------------------------------------------------------------------------

def _dbnomics_harmonic_forecast(question: dict, config: AgentConfig,
                                verbose: bool = False) -> dict:
    """Compute dbnomics forecast using harmonic model directly (no LLM).

    Fetches the time series, fits a harmonic model, and returns per-resolution-date
    probabilities. This bypasses the agent loop entirely for dbnomics questions,
    since the statistical model outperforms LLM reasoning on seasonal data.
    """
    import re as _re
    import time
    from agent.data_tools import fetch_dbnomics
    from agent.source_tools import _harmonic_forecast

    t0 = time.time()
    qid = question.get("id", "unknown")
    source = question.get("source", "dbnomics")
    fdd = question.get("forecast_due_date", "")
    res_dates = question.get("resolution_dates", [])

    # Extract URL from resolution criteria or background
    url = None
    for field in ["resolution_criteria", "background"]:
        m = _re.search(r'https://db\.nomics\.world/\S+', question.get(field, ""))
        if m:
            url = m.group(0).rstrip('.,;)')
            break

    if not url:
        # Fallback: construct from question ID
        qid_base = qid.split("_")[0] if "_" in qid else qid
        url = f"https://db.nomics.world/{qid_base.replace('_', '/')}"

    # Fetch data (use full history for better empirical estimates)
    if config.clairvoyant:
        from datetime import date
        cutoff = str(date.today())
    else:
        cutoff = fdd[:10] if fdd else ""
    df = fetch_dbnomics(url, cutoff)
    # fetch_dbnomics limits to 2 years; try to get more via dbnomics directly
    try:
        import dbnomics as _dbn
        path = url.split("db.nomics.world/")[1].rstrip("/")
        provider, dataset, series = path.split("/", 2)
        df_full = _dbn.fetch_series(provider, dataset, series)
        if df_full is not None and not df_full.empty:
            period_col = next((c for c in ("period", "original_period") if c in df_full.columns), None)
            if period_col:
                df_full = df_full[[period_col, "value"]].rename(columns={period_col: "ds", "value": "y"})
                df_full["ds"] = pd.to_datetime(df_full["ds"], errors="coerce").dt.tz_localize(None)
                df_full["y"] = pd.to_numeric(df_full["y"], errors="coerce")
                df_full = df_full.dropna(subset=["ds", "y"])
                df_full = df_full[df_full["ds"] <= pd.to_datetime(cutoff)]
                if len(df_full) > len(df):
                    df = df_full
    except Exception:
        pass

    if df is None or df.empty:
        if verbose:
            print(f"  [dbnomics/{qid}] no data, returning 0.5")
        final_p = 0.5
        forecasts = [0.5] * len(res_dates)
    else:
        # Get threshold (value on the FORECAST date, not the latest date)
        import pandas as pd
        fdd_dt = pd.to_datetime(fdd)
        df_before_fdd = df[df["ds"] <= fdd_dt]
        if df_before_fdd.empty:
            threshold = float(df["y"].iloc[-1])
        else:
            threshold = float(df_before_fdd["y"].iloc[-1])

        # Run harmonic model
        harmonic = _harmonic_forecast(df, threshold, res_dates)
        forecasts = [max(0.05, min(0.95, p)) for _, _, p in harmonic]
        final_p = forecasts[0] if forecasts else 0.5

        if verbose:
            pred_str = ", ".join(f"{rd}: p={p:.2f}" for rd, _, p in harmonic)
            print(f"  [dbnomics/{qid}] harmonic: threshold={threshold:.2f}, {pred_str}")

    result = {
        "id": qid,
        "source": source,
        "question": question.get("question", ""),
        "forecast_due_date": fdd,
        "forecast": final_p,
        "reasoning": f"Harmonic seasonal model: P(exceed) = {forecasts}",
        "resolution_date": question.get("resolution_date", ""),
        "resolution_dates": res_dates,
        "resolved_to": question.get("resolved_to"),
        "n_steps": 1,
        "submitted": True,
        "mode": "harmonic",
        "tokens_in": 0,
        "tokens_out": 0,
        "elapsed_seconds": round(time.time() - t0, 1),
        "config": config.to_dict(),
        "tool_log": [{"step": 1, "type": "harmonic_model",
                      "forecasts": forecasts}],
    }
    if len(forecasts) > 1:
        result["forecasts"] = forecasts

    return result


def _prior_only_forecast(question: dict) -> dict:
    """Submit empirical prior (dataset) or market price (market) directly, no LLM.

    For dataset questions, uses the source/subtype empirical base rate.
    For market questions, uses the market price as the probability.
    """
    from config.empirical_prior import get_empirical_prior, get_prior_explanation

    qid = question.get("id", "unknown")
    source = question.get("source", "unknown")
    res_dates = question.get("resolution_dates", [])

    # Dataset questions: empirical prior
    prior = get_empirical_prior(question)
    if prior is not None:
        explanation = get_prior_explanation(question)
        p = max(0.01, min(0.99, prior))  # clamp
    else:
        # Market questions: use market price
        market_val = question.get("market_value")
        if market_val and str(market_val).strip() not in ("", "unknown", "None"):
            try:
                p = float(market_val)
                explanation = "Market price"
            except (ValueError, TypeError):
                p = 0.5
                explanation = "Fallback (unparseable market value)"
        else:
            p = 0.5
            explanation = "Fallback (no prior available)"

    # Use same prior for all resolution dates
    n_rdates = len(res_dates) if res_dates else 1
    forecasts = [round(p, 4)] * n_rdates
    final_p = forecasts[0]

    result = {
        "id": qid,
        "source": source,
        "question": question.get("question", ""),
        "forecast_due_date": question.get("forecast_due_date", ""),
        "forecast": final_p,
        "reasoning": f"Prior-only baseline: {explanation} = {p:.4f}",
        "resolution_date": question.get("resolution_date", ""),
        "resolution_dates": res_dates,
        "resolved_to": question.get("resolved_to"),
        "n_steps": 0,
        "submitted": True,
        "mode": "prior_only",
        "tokens_in": 0,
        "tokens_out": 0,
        "elapsed_seconds": 0.0,
        "tool_log": [],
    }
    if n_rdates > 1:
        result["forecasts"] = forecasts

    return result


# Qtype-based policy overrides (None = use config default)
# Disabled for now: using uniform max_steps across all Qtypes
# for fair ablation comparisons. Re-enable after paper experiments.
_QTYPE_MAX_STEPS = {
    # "timeseries": 3,   # fetch_ts + ≤1 search + submit
    # "acled": 3,         # web search + submit (no specialized tool)
    # "wikipedia" and "market" use config.max_steps (default 10)
}

# Disabled for now: using uniform ntrials across all Qtypes
# for fair ablation comparisons. Re-enable after paper experiments.
_QTYPE_NTRIALS = {
    # "timeseries": 1,   # deterministic tool data, LLM variance is low
    # "acled": 1,         # tight loop, low variance
    # "wikipedia" and "market" use the --ntrials CLI value
}


def _run_one_pass(config: AgentConfig, questions: list[dict],
                  max_workers: int, verbose: bool,
                  base_dir: str, label: str,
                  monitor: ProgressMonitor | None = None) -> tuple[int, int]:
    """Run one pass of a config on all questions. Returns (n_ok, n_fail).

    Output: {base_dir}/{source}/{id}.json
    Searches: {base_dir}/{source}/searches/{id}/
    """
    # Check for existing forecasts (resume support)
    existing = set()
    for q in questions:
        source = q.get("source", "unknown")
        qid = q.get("id", "unknown")
        safe_id = re.sub(r'[/\\:]', '_', str(qid))
        if os.path.exists(os.path.join(base_dir, source, f"{safe_id}.json")):
            existing.add((source, qid))

    todo = [q for q in questions
            if (q.get("source", "unknown"), q.get("id", "unknown")) not in existing]
    print(f"[{label}] {len(existing)} done, {len(todo)} remaining")

    if monitor:
        monitor.set_existing(label, len(existing))

    if not todo:
        return len(existing), 0

    # Save config for reproducibility
    config_out = os.path.join(base_dir, "config.json")
    os.makedirs(os.path.dirname(config_out), exist_ok=True)
    config.save(config_out)

    t0 = time.time()
    n_ok = n_fail = 0

    def process_one(q):
        source = q.get("source", "unknown")
        qid = q.get("id", "unknown")
        safe_id = re.sub(r'[/\\:]', '_', str(qid))
        output_dir = os.path.join(base_dir, source)
        os.makedirs(output_dir, exist_ok=True)
        try:
            # Prior-only mode: submit empirical prior or market price, no LLM
            from config.tags import get_tag
            if config.prior_only:
                result = _prior_only_forecast(q)
            # Dbnomics special case: use harmonic model directly (no LLM)
            elif (source == "dbnomics" and config.use_tools):
                result = _dbnomics_harmonic_forecast(q, config, verbose=verbose)
            # Batch mode: non-agentic parallel search
            elif config.batch_queries > 0:
                result = run_batch_agent(q, config, output_dir, verbose=verbose)
            else:
                # Apply Qtype-based policy overrides
                qtype = get_tag(q, "Qtype")
                qtype_steps = _QTYPE_MAX_STEPS.get(qtype)
                if qtype_steps and qtype_steps < config.max_steps:
                    q_config = AgentConfig(**{k: getattr(config, k)
                                             for k in config.__dataclass_fields__})
                    q_config.max_steps = qtype_steps
                    result = run_agent(q, q_config, output_dir, verbose=verbose)
                else:
                    result = run_agent(q, config, output_dir, verbose=verbose)
            out_path = os.path.join(base_dir, source, f"{safe_id}.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            return qid, True, result.get("forecast", 0.5)
        except Exception as e:
            print(f"  [{label}] ERROR {qid}: {e}")
            return qid, False, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_one, q): q for q in todo}
        for future in as_completed(futures):
            qid, ok, p = future.result()
            if ok:
                n_ok += 1
                if verbose:
                    print(f"  [{label}] {qid}: p={p:.3f}")
            else:
                n_fail += 1
            if monitor:
                monitor.record(label, qid, ok, p)
            done = n_ok + n_fail
            if done % 10 == 0 or done == len(todo):
                elapsed = time.time() - t0
                print(f"  [{label}] Progress: {done}/{len(todo)} "
                      f"({elapsed:.0f}s, {elapsed/done:.1f}s/q)")

    elapsed = time.time() - t0
    print(f"  [{label}] Done: {n_ok} ok, {n_fail} fail, {elapsed:.0f}s")
    return len(existing) + n_ok, n_fail


def _average_trials(config_name: str, questions: list[dict], ntrials: int,
                    agg_method: str = "logit-mean",
                    shrinkage_floor: float = 0.3,
                    shrinkage_scale: float = 0.7):
    """Average forecasts across trials.

    agg_method:
        "logit-mean": Logit-space mean (paper §C.9 eq. 8 with α=1).
            p_hat = sigmoid(mean(logits)). No labels needed at inference.
            Default for SOTA — paper Table 3 / Sec C.9 show LOO selects
            α=1 on FB, i.e. this exact formula.
        "std-shrinkage": Logit-space mean × adaptive scaling toward 0.5.
            p_hat = sigmoid(α · mean(logits)),  α = max(f, 1 − c·std(logits)).
            Floor f and scale c are HARDCODED (0.3, 0.7) — they were
            originally chosen by LOO during paper development on AIBQ2;
            on FB the LOO optimum is α=1 (i.e. logit-mean) so std-
            shrinkage is mainly useful on AIBQ2-like exams. No runtime
            LOO; bake-in constants only.
        "plain-mean": deprecated alias for "logit-mean" (kept so old
            sota.json / saved-config dicts still load).

    For runtime LOO on α, use src/core/aggregate.py with the
    "shrink5-loo" variant — that one fits α to labeled data and
    therefore can't be used at live-submission time.
    """
    import numpy as np
    n_written = 0
    if agg_method == "plain-mean":
        agg_method = "logit-mean"  # backwards-compat alias
    method = agg_method

    for q in questions:
        source = q.get("source", "unknown")
        qid = q.get("id", "unknown")
        safe_id = re.sub(r'[/\\:]', '_', str(qid))

        # Collect forecasts from all trials
        trial_fcs = []
        for t in range(1, ntrials + 1):
            path = os.path.join("experiments", "forecasts_raw", config_name,
                                f"trial_{t}", source, f"{safe_id}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                fc = json.load(f)
            if fc.get("forecast") is not None:
                trial_fcs.append((t, fc))

        if not trial_fcs:
            continue

        # Compute statistics
        ps = [fc["forecast"] for _, fc in trial_fcs]
        ps_arr = np.array(ps)
        mean_p = float(np.mean(ps_arr))
        std_p = float(np.std(ps_arr))

        # Aggregate in logit space
        logits = np.log(np.clip(ps_arr, 0.001, 0.999)
                        / (1 - np.clip(ps_arr, 0.001, 0.999)))
        logit_bar = float(np.mean(logits))
        std_logit = float(np.std(logits))

        if agg_method == "std-shrinkage":
            # alpha = max(floor, 1 - scale * std_logit)
            # p_hat = sigmoid(alpha * logit_bar)
            a = max(shrinkage_floor, 1.0 - shrinkage_scale * std_logit)
            final_p = float(1 / (1 + np.exp(-a * logit_bar)))
        else:
            # logit-mean: logit-space average (alpha=1, no shrinkage)
            a = 1.0
            final_p = float(1 / (1 + np.exp(-logit_bar)))

        # Use first trial as base, add trial statistics
        base = dict(trial_fcs[0][1])
        base["forecast"] = final_p
        # Sum token counts and elapsed time across all trials
        base["tokens_in"] = sum(fc.get("tokens_in", 0) or 0 for _, fc in trial_fcs)
        base["tokens_out"] = sum(fc.get("tokens_out", 0) or 0 for _, fc in trial_fcs)
        base["elapsed_seconds"] = sum(fc.get("elapsed_seconds", 0) or 0 for _, fc in trial_fcs)
        base["n_steps"] = sum(fc.get("n_steps", 0) or 0 for _, fc in trial_fcs)
        base["trial_stats"] = {
            "n_trials": len(trial_fcs),
            "aggregation": agg_method,
            "shrinkage_alpha": float(a),
            "mean": mean_p,
            "median": float(np.median(ps_arr)),
            "min": float(np.min(ps_arr)),
            "max": float(np.max(ps_arr)),
            "std": std_p,
            "forecast_before_shrinkage": mean_p,
            "forecasts": {t: fc["forecast"] for t, fc in trial_fcs},
        }

        out_path = os.path.join("experiments", "forecasts_raw", config_name,
                                source, f"{safe_id}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(base, f, indent=2)
        n_written += 1

    print(f"  [{config_name}] Averaged {ntrials} trials ({method}) -> {n_written} forecasts")

    # Compute per-trial aggregate scores and save meta.json
    import math
    trial_nums = sorted(set(t for q in questions for t in range(1, ntrials + 1)))
    trial_scores = {}  # {trial: {metric: [scores]}}
    for t in trial_nums:
        brier_scores = []
        metaculus_scores = []
        for q in questions:
            source = q.get("source", "unknown")
            qid = q.get("id", "unknown")
            safe_id = re.sub(r'[/\\:]', '_', str(qid))
            path = os.path.join("experiments", "forecasts_raw", config_name,
                                f"trial_{t}", source, f"{safe_id}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                fc = json.load(f)
            p = fc.get("forecast")
            o = fc.get("resolved_to")
            if p is None or o is None:
                continue
            if isinstance(o, list):
                o = o[0] if o else None
            if o is None:
                continue
            o = float(o)
            brier_scores.append((p - o) ** 2)
            metaculus_scores.append(
                100 * (1 + math.log2(max(0.001, min(0.999,
                    p if o >= 0.5 else (1 - p))))))
        if brier_scores:
            trial_scores[t] = {
                "brier-score": sum(brier_scores) / len(brier_scores),
                "metaculus-score": sum(metaculus_scores) / len(metaculus_scores),
                "n": len(brier_scores),
            }

    meta = {
        "config": config_name,
        "ntrials": ntrials,
        "n_questions": n_written,
        "trial_scores": trial_scores,
    }
    meta_path = os.path.join("experiments", "forecasts_raw", config_name, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Print summary
    if trial_scores:
        for metric_name in ["metaculus-score", "brier-score"]:
            vals = [ts[metric_name] for ts in trial_scores.values() if metric_name in ts]
            if vals:
                mean_v = sum(vals) / len(vals)
                best_v = max(vals) if metric_name == "metaculus-score" else min(vals)
                best_t = [t for t, ts in trial_scores.items()
                          if ts.get(metric_name) == best_v][0]
                fmt = ".1f" if metric_name == "metaculus-score" else ".4f"
                print(f"  [{config_name}] {metric_name}: "
                      f"mean={mean_v:{fmt}}, best=trial_{best_t} ({best_v:{fmt}})")

    return n_written


def run_one_config(config: AgentConfig, questions: list[dict],
                   max_workers: int, verbose: bool,
                   monitor: ProgressMonitor | None = None,
                   ntrials: int = 1,
                   agg_method: str = "logit-mean") -> tuple[str, int, int]:
    """Run a config on all questions (possibly multiple trials). Returns (config_name, n_ok, n_fail).

    ntrials=1: writes to experiments/forecasts_raw/{config.name}/{source}/{id}.json
    ntrials>1: writes each trial to experiments/forecasts_raw/{config.name}/trial_{t}/{source}/{id}.json
               then averages into experiments/forecasts_raw/{config.name}/{source}/{id}.json
    """
    # Save full config (with ntrials) at the top level for reproducibility
    top_dir = os.path.join("experiments", "forecasts_raw", config.name)
    os.makedirs(top_dir, exist_ok=True)
    full_config = config.to_dict()
    full_config["ntrials"] = ntrials
    full_config["aggregation"] = agg_method
    with open(os.path.join(top_dir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=2)

    # Split questions by effective ntrials (Qtype policy overrides)
    from config.tags import get_tag
    single_trial_qs = []  # ntrials=1
    multi_trial_qs = []   # ntrials>1 (use CLI ntrials)
    for q in questions:
        qtype = get_tag(q, "Qtype")
        q_ntrials = _QTYPE_NTRIALS.get(qtype, ntrials)
        if q_ntrials == 1 or ntrials == 1:
            single_trial_qs.append(q)
        else:
            multi_trial_qs.append(q)

    total_ok = total_fail = 0

    # Single-trial questions: one pass, write directly to top_dir
    if single_trial_qs:
        n_ok, n_fail = _run_one_pass(config, single_trial_qs, max_workers, verbose,
                                      top_dir, config.name, monitor)
        total_ok += n_ok
        total_fail += n_fail

    # Multi-trial questions: run each trial, then average
    if multi_trial_qs:
        for t in range(1, ntrials + 1):
            trial_dir = os.path.join(top_dir, f"trial_{t}")
            label = f"{config.name}/trial_{t}"
            n_ok, n_fail = _run_one_pass(config, multi_trial_qs, max_workers, verbose,
                                          trial_dir, label, monitor)
            total_ok += n_ok
            total_fail += n_fail

    # Average trials (only for multi-trial questions)
    if multi_trial_qs:
        _average_trials(config.name, multi_trial_qs, ntrials, agg_method=agg_method)
    return config.name, total_ok // ntrials, total_fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run agentic forecaster on an exam")
    parser.add_argument("--xid", default=None,
                        help="Experiment ID: reads exam and configs from experiments/xids/{xid}.json")
    parser.add_argument("--exam", default=None,
                        help="Exam name (reads experiments/exams/{exam}/indices.json)")
    parser.add_argument("--config", default=None,
                        help="Config name(s), comma-separated "
                             "(delta strings like 'pro/thk:high/crowd:1' OK)")
    parser.add_argument("--config-file", default=None, metavar="PATH",
                        help="Load a full AgentConfig from a JSON file (e.g. "
                             "experiments/configs/sota.json). Mutually exclusive "
                             "with --config; takes precedence over an xid's "
                             "'config' field when given with --xid.")
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Max parallel workers per config (default: 50)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--monitor", action="store_true",
                        help="Write live progress.html to results/")
    parser.add_argument("--live", action="store_true",
                        help="Live mode (not backtesting): allow source URLs, skip redaction")
    parser.add_argument("--clairvoyant", action="store_true",
                        help="Set cutoff to today — agent can see future info (upper bound)")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to the first N questions from the exam "
                             "(useful to incrementally process a large exam)")
    parser.add_argument("--ntrials", type=int, default=1,
                        help="Run each config N times and average forecasts (default: 1)")
    parser.add_argument("--agg-method", default=None,
                        help="Trial aggregation: 'logit-mean' (default; α=1) or "
                             "'std-shrinkage' (hardcoded f=0.3, c=0.7). "
                             "'plain-mean' kept as deprecated alias for logit-mean.")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing forecasts before running")
    args = parser.parse_args()

    # Resolve exam name, config source, and --config-file takes precedence
    if args.config and args.config_file:
        parser.error("Pass only one of --config or --config-file, not both.")

    xid_data = {}
    if args.xid:
        xid_data = load_xid(args.xid)
        exam_name = xid_data["exam"]
    elif args.exam:
        exam_name = args.exam
    else:
        parser.error("Either --xid or --exam is required")

    configs: list = []

    if args.config_file:
        # Explicit file path → load a single AgentConfig dict.
        with open(args.config_file) as f:
            cfg_dict = json.load(f)
        if "name" not in cfg_dict or not cfg_dict["name"]:
            parser.error(f"Config file {args.config_file} must have a non-empty 'name' field.")
        from config.config import AgentConfig
        configs.append(AgentConfig(**{k: v for k, v in cfg_dict.items()
                                       if k in AgentConfig.__dataclass_fields__}))
    else:
        if args.xid and not args.config:
            cfg_list = xid_data.get("config", [])
            if isinstance(cfg_list, str):
                cfg_list = [cfg_list]
        elif args.config:
            cfg_list = [c.strip() for c in args.config.split(",") if c.strip()]
        else:
            parser.error("Need --config, --config-file, or --xid with a 'config' field")
        _EVAL_ONLY = {"sota", "baseline", "superhuman"}
        for cname in cfg_list:
            if cname in _EVAL_ONLY:
                continue
            configs.append(resolve_config(cname))

    # Load exam
    exam = load_exam(exam_name)
    questions = load_questions_from_exam(exam)

    if args.n is not None:
        questions = questions[:args.n]

    n_suffix = f" (first {args.n})" if args.n is not None else ""
    print(f"\nExam '{exam_name}': {len(questions)} questions{n_suffix}")

    # xid-level overrides
    if args.xid and "use_tools" in xid_data:
        for cfg in configs:
            cfg.use_tools = bool(xid_data["use_tools"])

    # CLI overrides
    if args.live:
        for cfg in configs:
            cfg.backtesting = False
    if args.clairvoyant:
        for cfg in configs:
            cfg.clairvoyant = True

    ntrials = args.ntrials
    trial_suffix = f" x {ntrials} trials (market/wiki), 1 trial (timeseries/acled)" if ntrials > 1 else ""
    print(f"Configs: {[c.name for c in configs]}{trial_suffix}")

    # Cost/time estimate based on config properties
    def _estimate_per_q(cfg):
        """Estimate (cost_per_q, sec_per_q) from config properties."""
        # Base cost per 1M tokens (input/output) by model family
        llm = cfg.llm.lower()
        if "flash" in llm:
            cost_in, cost_out = 0.15e-6, 0.6e-6
            base_sec = 15
        elif "kimi" in llm:
            cost_in, cost_out = 0.6e-6, 2.4e-6
            base_sec = 15
        else:  # pro, opus, gpt, etc.
            cost_in, cost_out = 2.5e-6, 10e-6
            base_sec = 20

        # Estimate steps and tokens based on config
        if cfg.max_steps == 1:
            # Zero-shot: ~2K tokens in, ~500 out
            tok_in, tok_out = 2000, 500
            sec = base_sec
        elif cfg.batch_queries > 0:
            # Batch: query gen + N searches + reasoning
            tok_in, tok_out = 25000, 5000
            sec = base_sec * 4
        elif cfg.search_engine == "none" and cfg.use_tools:
            # Tools only, no search: ~2 steps (tool + submit)
            tok_in, tok_out = 6000, 2000
            sec = base_sec * 2
        elif cfg.search_engine == "none":
            # No search, no tools: just submit
            tok_in, tok_out = 2000, 500
            sec = base_sec
        else:
            # Full agentic: ~4 steps avg
            tok_in, tok_out = 20000, 5000
            sec = base_sec * 5

        cost = tok_in * cost_in + tok_out * cost_out
        return cost, sec

    n_q = len(questions)
    est_cost = 0
    est_seq_sec = 0
    for cfg in configs:
        cpq, spq = _estimate_per_q(cfg)
        est_cost += cpq * n_q * ntrials
        est_seq_sec += spq * n_q * ntrials
    max_workers = min(args.max_workers, n_q * len(configs))
    est_wall_sec = est_seq_sec / max(1, max_workers)
    print(f"Estimate: ${est_cost:.0f} | "
          f"wall ~{est_wall_sec/3600:.1f}h ({max_workers} workers) | "
          f"seq ~{est_seq_sec/3600:.0f}h")

    # Delete existing forecasts for exam questions if --force
    if args.force:
        import shutil
        n_deleted = 0
        for cfg in configs:
            base = os.path.join("experiments", "forecasts_raw", cfg.name)
            for source, ids in exam.items():
                for qid in ids:
                    safe_id = re.sub(r'[/\\:]', '_', str(qid))
                    # Delete aggregated forecast + trace
                    for ext in [".json", "_trace.html"]:
                        p = os.path.join(base, source, f"{safe_id}{ext}")
                        if os.path.exists(p):
                            os.remove(p)
                            n_deleted += 1
                    # Delete searches
                    sd = os.path.join(base, source, "searches", safe_id)
                    if os.path.isdir(sd):
                        shutil.rmtree(sd)
                    # Delete per-trial files
                    for t_dir in glob.glob(os.path.join(base, "trial_*")):
                        for ext in [".json", "_trace.html"]:
                            p = os.path.join(t_dir, source, f"{safe_id}{ext}")
                            if os.path.exists(p):
                                os.remove(p)
                                n_deleted += 1
        print(f"Cleared {n_deleted} existing forecast files (--force)")

    # Build monitor labels (include trial labels if ntrials > 1)
    if ntrials > 1:
        monitor_labels = []
        for c in configs:
            for t in range(1, ntrials + 1):
                monitor_labels.append(f"{c.name}/trial_{t}")
    else:
        monitor_labels = [c.name for c in configs]

    # Start progress monitor
    monitor = None
    if args.monitor:
        run_label = args.xid or exam_name
        monitor_dir = os.path.join("experiments", "progress", run_label)
        monitor = ProgressMonitor(monitor_dir, monitor_labels, len(questions))
        monitor.start()
        print(f"Monitor: {os.path.join(monitor_dir, 'progress.html')}")

    # Resolve aggregation method: CLI flag overrides config default
    agg_method = args.agg_method or configs[0].agg_method if configs else "logit-mean"

    try:
        if len(configs) == 1:
            run_one_config(configs[0], questions, args.max_workers, args.verbose,
                           monitor, ntrials, agg_method=agg_method)
        else:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=len(configs)) as pool:
                futures = {
                    pool.submit(run_one_config, cfg, questions,
                                args.max_workers, args.verbose, monitor,
                                ntrials, agg_method): cfg.name
                    for cfg in configs
                }
                for future in as_completed(futures):
                    name, n_ok, n_fail = future.result()
                    print(f"[{name}] Finished: {n_ok} ok, {n_fail} fail")
            elapsed = time.time() - t0
            print(f"\nAll configs done in {elapsed:.0f}s")
    finally:
        if monitor:
            monitor.stop()


if __name__ == "__main__":
    main()
