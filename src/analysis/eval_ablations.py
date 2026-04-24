#!/usr/bin/env python3
"""eval_ablations.py — Ablation table: gains from each post-processing step.

For each base config in an xid, computes metric scores under different
aggregation and calibration settings, using the raw trial data.

Rows:
    1. Single trial (expected score = mean of K individual trial scores)
    2. Plain mean (k=N, shrinkage=1.0)
    3. Shrinkage mean (k=N, std-based shrinkage)
    4. Shrinkage + calibration
    5. Best single trial (best-of-N)

Usage:
    python3 src/eval_ablations.py --xid xid-aibq2
    python3 src/eval_ablations.py --xid xid-aibq2 --metric brier-score
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eval import (
    SCORING_FUNCTIONS, METRIC_LABELS, HIGHER_IS_BETTER,
    load_xid, load_exam, forecast_path, _resolve_outcome,
)


def _score_fn(metric):
    return SCORING_FUNCTIONS[metric]


def _load_trial_data(config, exam, ntrials):
    """Load per-question, per-trial forecasts and outcomes.

    Returns list of (qid, outcome, {trial_num: forecast}).
    """
    questions = []
    for source, ids in exam.items():
        for qid in ids:
            safe_id = re.sub(r'[/\\:]', '_', str(qid))
            # Load outcome from any trial
            outcome = None
            trial_forecasts = {}
            for t in range(1, ntrials + 1):
                path = os.path.join("experiments", "forecasts", config,
                                    f"trial_{t}", source, f"{safe_id}.json")
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    fc = json.load(f)
                p = fc.get("forecast")
                o = _resolve_outcome(fc.get("resolved_to"))
                if p is not None and o is not None:
                    trial_forecasts[t] = p
                    outcome = float(o)
            if outcome is not None and trial_forecasts:
                questions.append((f"{source}/{qid}", outcome, trial_forecasts))
    return questions


def _load_calibrated_scores(config, exam, metric):
    """Load scores from calibrated forecasts."""
    cal_config = f"{config}_calibrated"
    score_fn = _score_fn(metric)
    scores = []
    for source, ids in exam.items():
        for qid in ids:
            path = forecast_path(cal_config, source, qid)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                fc = json.load(f)
            p = fc.get("forecast")
            o = _resolve_outcome(fc.get("resolved_to"))
            if p is not None and o is not None:
                scores.append(score_fn(p, float(o)))
    return np.mean(scores) if scores else None


def compute_ablation(config, exam, metric, ntrials):
    """Compute ablation scores for a config.

    Returns dict with keys: single_trial, plain_mean, shrinkage_mean,
    shrinkage_cal, best_trial, ntrials, per_trial_scores.
    """
    questions = _load_trial_data(config, exam, ntrials)
    if not questions:
        return None

    score_fn = _score_fn(metric)

    # 1. Per-trial scores
    all_trials = set()
    for _, _, tf in questions:
        all_trials.update(tf.keys())
    all_trials = sorted(all_trials)

    per_trial = {}
    for t in all_trials:
        t_scores = []
        for _, outcome, tf in questions:
            if t in tf:
                t_scores.append(score_fn(tf[t], outcome))
        if t_scores:
            per_trial[t] = float(np.mean(t_scores))

    # Single trial = mean of per-trial scores
    single_trial = float(np.mean(list(per_trial.values()))) if per_trial else None

    # Best single trial
    higher = metric in HIGHER_IS_BETTER
    best_trial_score = (max(per_trial.values()) if higher
                        else min(per_trial.values())) if per_trial else None
    best_trial_num = ([t for t, s in per_trial.items() if s == best_trial_score][0]
                      if best_trial_score is not None else None)

    # 2. Plain mean (k=N, shrinkage=1.0)
    plain_scores = []
    for _, outcome, tf in questions:
        ps = list(tf.values())
        plain_scores.append(score_fn(np.mean(ps), outcome))
    plain_mean = float(np.mean(plain_scores)) if plain_scores else None

    # 3. Shrinkage mean (k=N, logit-space std-based)
    shrink_scores = []
    for _, outcome, tf in questions:
        ps = np.array(list(tf.values()))
        logits = np.log(np.clip(ps, 0.001, 0.999) / (1 - np.clip(ps, 0.001, 0.999)))
        logit_bar = float(np.mean(logits))
        std_logit = float(np.std(logits)) if len(ps) > 1 else 0.0
        a = max(0.3, 1.0 - 0.7 * std_logit)
        p_final = float(1 / (1 + np.exp(-a * logit_bar)))
        shrink_scores.append(score_fn(p_final, outcome))
    shrinkage_mean = float(np.mean(shrink_scores)) if shrink_scores else None

    # 4. Shrinkage + calibration (read from calibrated forecasts)
    shrinkage_cal = _load_calibrated_scores(config, exam, metric)

    return {
        "ntrials": len(all_trials),
        "n_questions": len(questions),
        "per_trial_scores": per_trial,
        "single_trial": single_trial,
        "best_trial": best_trial_score,
        "best_trial_num": best_trial_num,
        "plain_mean": plain_mean,
        "shrinkage_mean": shrinkage_mean,
        "shrinkage_cal": shrinkage_cal,
    }


def generate_ablation_html(output_dir, xid_name, configs, results, metric):
    """Generate ablations.html."""
    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    higher = metric in HIGHER_IS_BETTER
    fmt = ".1f" if metric == "metaculus-score" else ".4f"

    def _fmt(v, ref=None):
        if v is None:
            return '<td class="num">—</td>'
        s = f"{v:{fmt}}"
        gain = ""
        if ref is not None and ref is not v:
            diff = v - ref
            color = "#27ae60" if (diff > 0) == higher else "#c0392b"
            gain = f' <span style="color:{color};font-size:0.85em">({diff:+{fmt}})</span>'
        return f'<td class="num">{s}{gain}</td>'

    # Determine K from first config with results
    K = next((results[c]["ntrials"] for c in configs if c in results), "?")

    # Build rows
    row_labels = [
        (f"Single trial (mean of K={K})", "single_trial"),
        ("Best single trial (oracle upper bound)", "best_trial"),
        ("Plain mean", "plain_mean"),
        ("Shrinkage mean", "shrinkage_mean"),
        ("Shrinkage + calibration", "shrinkage_cal"),
    ]

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Ablations — {xid_name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 900px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 1.3em; }}
table {{ border-collapse: collapse; font-size: 14px; margin: 16px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 14px; }}
th {{ background: #f5f5f5; text-align: left; }}
td.num {{ text-align: right; font-family: monospace; }}
.note {{ font-size: 12px; color: #666; margin-top: 8px; }}
</style></head><body>

<h1>Ablation: {label}</h1>
<p>XID: {xid_name} | {subtitle} |
<a href="leaderboard.html">Leaderboard</a></p>

<table>
<thead>
<tr><th>Post-processing step</th>"""

    for config in configs:
        r = results.get(config)
        n = r["ntrials"] if r else "?"
        html += f'<th>{config}<br><span style="color:#888;font-size:0.85em">(k={n})</span></th>'
    html += "</tr></thead><tbody>"

    for row_label, key in row_labels:
        html += f"<tr><td>{row_label}</td>"
        for config in configs:
            r = results.get(config)
            if r is None:
                html += '<td class="num">—</td>'
                continue
            val = r.get(key)
            ref = r.get("single_trial")
            if key == "single_trial":
                html += _fmt(val)
            else:
                html += _fmt(val, ref)
        html += "</tr>"

    html += "</tbody></table>"

    # Per-trial breakdown
    html += "<h2>Per-trial scores</h2><table><thead><tr><th>Trial</th>"
    for config in configs:
        html += f"<th>{config}</th>"
    html += "</tr></thead><tbody>"

    max_trials = max((len(results[c]["per_trial_scores"])
                      for c in configs if c in results), default=0)
    for t in range(1, max_trials + 1):
        html += f"<tr><td>Trial {t}</td>"
        for config in configs:
            r = results.get(config)
            if r and t in r.get("per_trial_scores", {}):
                v = r["per_trial_scores"][t]
                best_t = r.get("best_trial_num")
                bold = " font-weight:bold;" if t == best_t else ""
                html += f'<td class="num" style="{bold}">{v:{fmt}}</td>'
            else:
                html += '<td class="num">—</td>'
        html += "</tr>"

    html += "</tbody></table>"

    # Incremental effect sizes
    html += "<h2>Incremental effect sizes</h2>"
    html += "<p>Each row shows the gain from that step over the previous step.</p>"
    html += '<table><thead><tr><th>Step</th>'
    for config in configs:
        html += f"<th>{config}</th>"
    html += "</tr></thead><tbody>"

    steps = [
        ("Single trial", "single_trial", None),
        ("Best single trial (oracle)", "best_trial", "single_trial"),
        ("+ Plain mean", "plain_mean", "single_trial"),
        ("+ Shrinkage (vs plain)", "shrinkage_mean", "plain_mean"),
        ("+ Calibration (vs shrinkage)", "shrinkage_cal", "shrinkage_mean"),
    ]
    for step_label, key, prev_key in steps:
        html += f"<tr><td>{step_label}</td>"
        for config in configs:
            r = results.get(config)
            if r is None:
                html += '<td class="num">—</td>'
                continue
            val = r.get(key)
            if val is None:
                html += '<td class="num">—</td>'
                continue
            if prev_key is None:
                html += f'<td class="num">{val:{fmt}}</td>'
            else:
                prev_val = r.get(prev_key)
                if prev_val is not None:
                    diff = val - prev_val
                    color = "#27ae60" if (diff > 0) == higher else "#c0392b"
                    html += (f'<td class="num">{val:{fmt}} '
                             f'<span style="color:{color};font-size:0.85em">'
                             f'({diff:+{fmt}})</span></td>')
                else:
                    html += f'<td class="num">{val:{fmt}}</td>'
        html += "</tr>"
    html += "</tbody></table>"

    html += f"""
<p class="note">
Single trial (mean of K={K}) = mean of K individual trial scores (expected score of a single run).<br>
Plain mean = average of K trial forecasts per question (&alpha; = 1, no shrinkage).<br>
Shrinkage mean = logit-space shrinkage toward 0.5:
  p&#x0302; = sigmoid(&alpha; &middot; logit&#x0304;),
  where &alpha; = max(0.3, 1&minus;0.7&middot;std<sub>logit</sub>).<br>
Gains shown relative to single trial baseline.<br>
See <a href="../../docs/shrinkage.tex">docs/shrinkage.tex</a> for derivation.
</p>
</body></html>"""

    out_path = os.path.join(output_dir, "ablations.html")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute ablation table for post-processing steps")
    parser.add_argument("--xid", required=True)
    parser.add_argument("--metric", default="metaculus-score",
                        choices=list(SCORING_FUNCTIONS.keys()),
                        help="Metric to ablate (default: metaculus-score)")
    args = parser.parse_args()

    xid_data = load_xid(args.xid)
    exam_name = xid_data["exam"]
    configs = xid_data.get("config", [])
    if isinstance(configs, str):
        configs = [configs]
    # Filter to base configs only
    configs = [c for c in configs if c not in ("sota", "baseline")]

    exam = load_exam(exam_name)
    label, _ = METRIC_LABELS.get(args.metric, (args.metric, ""))

    print(f"Ablation: {args.xid} ({label})")
    print(f"  Exam: {exam_name}")
    print(f"  Configs: {configs}")

    results = {}
    for config in configs:
        # Discover ntrials
        trial_dirs = glob.glob(os.path.join("experiments", "forecasts", config, "trial_*"))
        ntrials = len(trial_dirs)
        if ntrials == 0:
            print(f"  [{config}] no trials, skipping")
            continue

        r = compute_ablation(config, exam, args.metric, ntrials)
        if r:
            results[config] = r
            fmt = ".1f" if args.metric == "metaculus-score" else ".4f"
            print(f"  [{config}] k={r['ntrials']}, n={r['n_questions']}")
            cal_str = f"{r['shrinkage_cal']:{fmt}}" if r['shrinkage_cal'] is not None else "—"
            print(f"    single={r['single_trial']:{fmt}}, "
                  f"plain={r['plain_mean']:{fmt}}, "
                  f"shrink={r['shrinkage_mean']:{fmt}}, "
                  f"shrink+cal={cal_str}, "
                  f"best={r['best_trial']:{fmt}} (t{r['best_trial_num']})")

    if results:
        output_dir = os.path.join("experiments", "eval", args.xid)
        out = generate_ablation_html(output_dir, args.xid, configs, results, args.metric)
        print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
