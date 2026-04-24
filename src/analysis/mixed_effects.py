#!/usr/bin/env python3
"""mixed_effects.py — Mixed-effects analysis of multi-trial forecasting results.

Fits the model:
    score(q, m, t) = mu + alpha_m + gamma_q + epsilon_{q,m,t}

where:
    mu       = grand mean
    alpha_m  = method effect (what we want to estimate)
    gamma_q  = question difficulty (nuisance, factored out)
    epsilon  = residual (cross-trial noise)

Reports:
    - Method effects (alpha_m) with bootstrap CI
    - Question difficulty distribution
    - Variance decomposition: between-method, between-question, within-trial
    - Pairwise method comparisons (paired differences on same questions)

Usage:
    python3 src/mixed_effects.py --xid xid-aibq2
    python3 src/mixed_effects.py --xid xid-aibq2 --metric brier-score
    python3 src/mixed_effects.py --xid xid-aibq2 --reference pro-high-brave-crowd0-tools0_calibrated
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


def _load_trial_scores(config, exam, metric):
    """Load per-question, per-trial metric scores.

    Returns {qid: [score_t1, score_t2, ...]} and {qid: mean_score}.
    Also returns the aggregated (shrunk) score per question.
    """
    score_fn = SCORING_FUNCTIONS[metric]
    trial_scores = {}  # qid -> [scores]
    agg_scores = {}    # qid -> aggregated score

    for source, ids in exam.items():
        for qid in ids:
            safe_id = re.sub(r'[/\\:]', '_', str(qid))
            key = f"{source}/{qid}"

            # Load aggregated forecast
            fc_path = forecast_path(config, source, qid)
            if not os.path.exists(fc_path):
                continue
            with open(fc_path) as f:
                fc = json.load(f)
            outcome = _resolve_outcome(fc.get("resolved_to"))
            if outcome is None:
                continue
            outcome = float(outcome)

            # Aggregated score
            p_agg = fc.get("forecast")
            if p_agg is not None:
                agg_scores[key] = score_fn(p_agg, outcome)

            # Per-trial scores
            ts = fc.get("trial_stats", {})
            trial_fcs = ts.get("forecasts", {})
            if trial_fcs:
                t_scores = [score_fn(p, outcome) for p in trial_fcs.values()]
                trial_scores[key] = t_scores
            elif p_agg is not None:
                # Single trial
                trial_scores[key] = [score_fn(p_agg, outcome)]

    return trial_scores, agg_scores


def _alternating_projections(long_data, tol=1e-8, max_iter=500):
    """Estimate method and question effects via alternating projections.

    long_data: list of (method, question, score) tuples.
    Returns (method_effects, question_effects, grand_mean).
    """
    methods = sorted(set(m for m, _, _ in long_data))
    questions = sorted(set(q for _, q, _ in long_data))
    m_idx = {m: i for i, m in enumerate(methods)}
    q_idx = {q: i for i, q in enumerate(questions)}

    n_m = len(methods)
    n_q = len(questions)
    mi = np.array([m_idx[m] for m, _, _ in long_data])
    qi = np.array([q_idx[q] for _, q, _ in long_data])
    scores = np.array([s for _, _, s in long_data])

    grand_mean = float(np.mean(scores))

    a = np.zeros(n_m)  # method effects
    b = np.zeros(n_q)  # question effects

    m_count = np.bincount(mi, minlength=n_m).astype(float)
    q_count = np.bincount(qi, minlength=n_q).astype(float)

    for _ in range(max_iter):
        b_prev = b.copy()

        # b_j = mean_i(score_ij - a_i)
        resid_q = scores - grand_mean - a[mi]
        b = np.where(q_count > 0,
                     np.bincount(qi, weights=resid_q, minlength=n_q) / np.maximum(q_count, 1),
                     0.0)

        # a_i = mean_j(score_ij - b_j)
        resid_m = scores - grand_mean - b[qi]
        a = np.where(m_count > 0,
                     np.bincount(mi, weights=resid_m, minlength=n_m) / np.maximum(m_count, 1),
                     0.0)

        if np.max(np.abs(b - b_prev)) < tol:
            break

    method_effects = {m: float(a[i]) for m, i in m_idx.items()}
    question_effects = {q: float(b[i]) for q, i in q_idx.items()}
    return method_effects, question_effects, grand_mean


def main():
    parser = argparse.ArgumentParser(
        description="Mixed-effects analysis of multi-trial forecasting results")
    parser.add_argument("--xid", required=True)
    parser.add_argument("--metric", default="metaculus-score",
                        choices=list(SCORING_FUNCTIONS.keys()))
    parser.add_argument("--reference", default=None,
                        help="Reference method for pairwise comparison")
    args = parser.parse_args()

    xid_data = load_xid(args.xid)
    exam_name = xid_data["exam"]
    configs = xid_data.get("eval", xid_data.get("config", []))
    if isinstance(configs, str):
        configs = [configs]
    # Filter out non-method entries
    configs = [c for c in configs if c not in ("sota", "baseline") and "[" not in c]
    # Resolve delta strings to filesystem directory names
    from config.config import resolve_config, pprint_path
    resolved = []
    for c in configs:
        if "/" in c and ":" in c:
            cfg = resolve_config(c)
            resolved.append(pprint_path(cfg))
        else:
            resolved.append(c)
    configs = resolved
    # Also resolve reference
    if args.reference and "/" in args.reference and ":" in args.reference:
        cfg = resolve_config(args.reference)
        args.reference = pprint_path(cfg)

    exam = load_exam(exam_name)
    metric = args.metric
    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    higher = metric in HIGHER_IS_BETTER

    print(f"Mixed-Effects Analysis: {args.xid}")
    print(f"  Metric: {label} ({subtitle})")
    print(f"  Configs: {configs}")
    print()

    # Load all trial scores
    all_trial_scores = {}  # config -> {qid: [scores]}
    all_agg_scores = {}    # config -> {qid: score}
    for config in configs:
        ts, agg = _load_trial_scores(config, exam, metric)
        all_trial_scores[config] = ts
        all_agg_scores[config] = agg
        n_q = len(ts)
        n_trials = sum(len(v) for v in ts.values())
        mean_score = np.mean(list(agg.values())) if agg else 0
        fmt = ".1f" if metric == "metaculus-score" else ".4f"
        print(f"  [{config}] {n_q} questions, {n_trials} trial-scores, "
              f"mean(agg)={mean_score:{fmt}}")

    # =====================================================
    # 1. Alternating projections: estimate method + question effects
    # =====================================================
    print(f"\n{'='*60}")
    print("1. Method Effects (alternating projections)")
    print(f"{'='*60}")

    # Build long-form data using AGGREGATED scores (one per question per method)
    long_agg = []
    for config in configs:
        for qid, score in all_agg_scores[config].items():
            long_agg.append((config, qid, score))

    if len(set(m for m, _, _ in long_agg)) >= 2:
        method_fx, question_fx, grand_mean = _alternating_projections(long_agg)

        fmt = ".1f" if metric == "metaculus-score" else ".4f"
        print(f"\n  Grand mean: {grand_mean:{fmt}}")
        print(f"\n  {'Method':<50s} {'Effect':>10s} {'Adjusted':>10s}")
        print("  " + "-" * 72)
        sorted_methods = sorted(method_fx.items(),
                                key=lambda x: x[1], reverse=higher)
        for m, fx in sorted_methods:
            adj = grand_mean + fx
            print(f"  {m:<50s} {fx:>+10{fmt}} {adj:>10{fmt}}")

        # Question difficulty stats
        q_vals = list(question_fx.values())
        print(f"\n  Question effects: mean={np.mean(q_vals):{fmt}}, "
              f"std={np.std(q_vals):{fmt}}, "
              f"min={np.min(q_vals):{fmt}}, max={np.max(q_vals):{fmt}}")
    else:
        print("  Need >= 2 methods for alternating projections")
        method_fx = {}
        question_fx = {}

    # =====================================================
    # 2. Variance decomposition
    # =====================================================
    print(f"\n{'='*60}")
    print("2. Variance Decomposition")
    print(f"{'='*60}")

    # Using trial-level data
    long_trials = []
    for config in configs:
        for qid, t_scores in all_trial_scores[config].items():
            for t, score in enumerate(t_scores):
                long_trials.append((config, qid, t, score))

    if long_trials:
        # Proper Type III sum-of-squares decomposition.
        # For balanced data (same K trials for each (q,m) pair):
        #   SS_total   = sum_{q,m,t} (s_{qmt} - s_...)^2
        #   SS_method  = K*Q * sum_m (s_.m. - s_...)^2
        #   SS_question = K*M * sum_q (s_q.. - s_...)^2
        #   SS_residual = SS_total - SS_method - SS_question

        all_scores_arr = np.array([s for _, _, _, s in long_trials])
        grand_mean_trial = float(np.mean(all_scores_arr))
        N = len(all_scores_arr)
        ss_total = float(np.sum((all_scores_arr - grand_mean_trial) ** 2))

        # Method means and SS
        method_trial_means = {}
        for config in configs:
            vals = [s for m, _, _, s in long_trials if m == config]
            method_trial_means[config] = np.mean(vals) if vals else 0
        n_per_method = N // len(configs) if configs else 1
        ss_method = float(n_per_method * np.sum(
            [(m - grand_mean_trial) ** 2 for m in method_trial_means.values()]))

        # Question means and SS
        q_trial_means = {}
        for _, q, _, s in long_trials:
            q_trial_means.setdefault(q, []).append(s)
        q_trial_means = {q: np.mean(v) for q, v in q_trial_means.items()}
        n_per_question = N // len(q_trial_means) if q_trial_means else 1
        ss_question = float(n_per_question * np.sum(
            [(m - grand_mean_trial) ** 2 for m in q_trial_means.values()]))

        ss_residual = max(0, ss_total - ss_method - ss_question)

        fmt = ".1f" if metric == "metaculus-score" else ".4f"
        pct = lambda x: f"{x / ss_total * 100:.1f}%" if ss_total > 0 else "—"
        print(f"\n  SS Total:              {ss_total:{fmt}}")
        print(f"  SS Method:             {ss_method:{fmt}} ({pct(ss_method)})")
        print(f"  SS Question:           {ss_question:{fmt}} ({pct(ss_question)})")
        print(f"  SS Residual:           {ss_residual:{fmt}} ({pct(ss_residual)})")

        # Also report as variance (MS = SS / df)
        M = len(configs)
        Q = len(q_trial_means)
        K_avg = N / (M * Q) if M * Q > 0 else 1
        df_method = M - 1
        df_question = Q - 1
        df_residual = max(1, N - M - Q + 1)
        ms_method = ss_method / max(1, df_method)
        ms_question = ss_question / max(1, df_question)
        ms_residual = ss_residual / df_residual

        print(f"\n  Mean Squares (MS = SS/df):")
        print(f"    Method:   {ms_method:{fmt}} (df={df_method})")
        print(f"    Question: {ms_question:{fmt}} (df={df_question})")
        print(f"    Residual: {ms_residual:{fmt}} (df={df_residual})")
        if ms_residual > 0:
            f_method = ms_method / ms_residual
            f_question = ms_question / ms_residual
            print(f"  F-ratios: method={f_method:.2f}, question={f_question:.2f}")

    # =====================================================
    # 3. Pairwise comparisons (paired by question)
    # =====================================================
    print(f"\n{'='*60}")
    print("3. Pairwise Comparisons (paired by question)")
    print(f"{'='*60}")

    ref = args.reference
    if ref and ref not in configs:
        print(f"  WARNING: reference '{ref}' not in configs, using best method")
        ref = None
    if not ref:
        # Use best method as reference
        if method_fx:
            ref = max(method_fx, key=method_fx.get) if higher else min(method_fx, key=method_fx.get)
        else:
            ref = configs[0]

    print(f"\n  Reference: {ref}")
    fmt = ".1f" if metric == "metaculus-score" else ".4f"
    print(f"\n  {'Method':<50s} {'Δ mean':>8s} {'Δ std':>8s} {'p(Δ>0)':>8s} {'n':>5s}")
    print("  " + "-" * 82)

    for config in configs:
        if config == ref:
            continue
        # Paired differences on common questions (using aggregated scores)
        common = set(all_agg_scores[config]) & set(all_agg_scores[ref])
        if not common:
            continue
        diffs = [all_agg_scores[config][q] - all_agg_scores[ref][q] for q in common]
        diffs_arr = np.array(diffs)
        mean_diff = float(np.mean(diffs_arr))
        std_diff = float(np.std(diffs_arr))
        # Bootstrap p-value: proportion of bootstrap means > 0 (or < 0)
        rng = np.random.default_rng(42)
        boot = rng.choice(diffs_arr, size=(2000, len(diffs_arr)), replace=True).mean(axis=1)
        if higher:
            p_better = float((boot > 0).mean())
        else:
            p_better = float((boot < 0).mean())
        print(f"  {config:<50s} {mean_diff:>+8{fmt}} {std_diff:>8{fmt}} "
              f"{p_better:>8.3f} {len(common):>5d}")

    # =====================================================
    # Save results
    # =====================================================
    output_dir = os.path.join("experiments", "eval", args.xid)
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "metric": metric,
        "grand_mean": grand_mean if method_fx else None,
        "method_effects": method_fx,
        "reference": ref,
        "configs": configs,
    }
    out_path = os.path.join(output_dir, "mixed_effects.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
