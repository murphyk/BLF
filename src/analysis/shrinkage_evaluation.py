#!/usr/bin/env python3
"""shrinkage_evaluation.py — Compare aggregation methods on multi-trial forecasts.

Reproduces the empirical comparison table from docs/shrinkage.tex and
searches for optimal heuristic parameters.

Usage:
    python3 src/shrinkage_evaluation.py --xid xid-aibq2
    python3 src/shrinkage_evaluation.py --xid xid-aibq2 --config pro-high-brave-crowd0-tools0
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
from scipy.special import expit as sigmoid
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def logit(p):
    return np.log(np.clip(p, 0.001, 0.999) / (1 - np.clip(p, 0.001, 0.999)))


def ms(p, o):
    return 100 * (1 + math.log2(max(0.001, min(0.999, p if o >= 0.5 else (1 - p)))))


def load_trial_data(config):
    """Load per-question trial forecasts and outcomes."""
    questions = []
    for f in sorted(glob.glob(f"experiments/forecasts/{config}/aibq2/*.json")):
        fc = json.load(open(f))
        ts = fc.get("trial_stats")
        o = fc.get("resolved_to")
        if not ts or o is None:
            continue
        if isinstance(o, list):
            o = o[0]
        ps = list(ts["forecasts"].values())
        if len(ps) < 2:
            continue
        questions.append((float(o), ps))
    return questions


def score_method(questions, agg_fn):
    """Score an aggregation function. Returns (mean_metaculus, n_catastrophic)."""
    scores = []
    for o, ps in questions:
        p_hat = agg_fn(ps)
        scores.append(ms(p_hat, o))
    arr = np.array(scores)
    return float(np.mean(arr)), int((arr < -200).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xid", required=True)
    parser.add_argument("--config", default=None,
                        help="Specific config (default: all in xid)")
    args = parser.parse_args()

    xid_data = json.load(open(f"experiments/xids/{args.xid}.json"))
    configs = [args.config] if args.config else [
        c for c in xid_data.get("config", []) if c not in ("sota", "baseline")]

    for config in configs:
        questions = load_trial_data(config)
        if not questions:
            print(f"[{config}] no trial data")
            continue
        K = len(questions[0][1])

        print(f"\n{'='*70}")
        print(f"Config: {config}")
        print(f"Questions: {len(questions)}, K={K} trials")
        print(f"{'='*70}")

        # --- Standard methods ---
        print(f"\n{'Method':<45s} {'Metaculus':>10s} {'Catastrophic':>12s}")
        print("-" * 70)

        # Plain mean
        mean_ms, mean_cat = score_method(questions,
            lambda ps: np.mean(ps))
        print(f"{'Plain mean (alpha=1)':<45s} {mean_ms:>10.1f} {mean_cat:>12d}")

        # EB logit-normal (various tau^2)
        for tau2 in [0.05, 0.10, 0.15, 0.25, 0.5]:
            def eb_agg(ps, _tau2=tau2):
                logits = [logit(p) for p in ps]
                y_bar = np.mean(logits)
                s2 = max(0.001, np.var(logits, ddof=1))
                v_j = 1 / (K / s2 + 1 / _tau2)
                m_j = v_j * (K * y_bar / s2)
                return float(sigmoid(m_j))
            eb_ms, eb_cat = score_method(questions, eb_agg)
            print(f"{'EB logit-normal (tau2=' + f'{tau2:.2f})':<45s} {eb_ms:>10.1f} {eb_cat:>12d}")

        # Heuristic (current default)
        def heuristic(ps):
            a = max(0.3, 1 - 3 * np.std(ps))
            return a * np.mean(ps) + (1 - a) * 0.5
        h_ms, h_cat = score_method(questions, heuristic)
        print(f"{'Heuristic (floor=0.3, scale=3)':<45s} {h_ms:>10.1f} {h_cat:>12d}")

        # --- Grid search over heuristic parameters ---
        print(f"\n--- Grid search: alpha = max(floor, 1 - scale * std) ---")
        print(f"{'floor':<8s} {'scale':<8s} {'Metaculus':>10s} {'Catastrophic':>12s}")
        print("-" * 42)

        best_ms = -1e9
        best_params = (0.3, 3.0)
        results = []

        for floor in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for scale in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
                def h(ps, _f=floor, _s=scale):
                    a = max(_f, 1 - _s * np.std(ps))
                    return a * np.mean(ps) + (1 - a) * 0.5
                m, c = score_method(questions, h)
                results.append((floor, scale, m, c))
                if m > best_ms:
                    best_ms = m
                    best_params = (floor, scale)

        # Show top 10
        results.sort(key=lambda x: -x[2])
        for floor, scale, m, c in results[:10]:
            marker = " <-- current" if (floor == 0.3 and scale == 3.0) else ""
            marker = " <-- BEST" if (floor == best_params[0] and scale == best_params[1]) and not marker else marker
            print(f"{floor:<8.1f} {scale:<8.1f} {m:>10.1f} {c:>12d}{marker}")

        print(f"\nBest: floor={best_params[0]}, scale={best_params[1]}, "
              f"metaculus={best_ms:.1f}")

        # --- Minimum catastrophes ---
        print(f"\n--- Minimum catastrophic failures ---")
        min_cat = min(r[3] for r in results)
        cat_results = [(f, s, m, c) for f, s, m, c in results if c == min_cat]
        cat_results.sort(key=lambda x: -x[2])
        print(f"Minimum catastrophic = {min_cat}")
        print(f"{'floor':<8s} {'scale':<8s} {'Metaculus':>10s} {'Catastrophic':>12s}")
        for floor, scale, m, c in cat_results[:5]:
            print(f"{floor:<8.1f} {scale:<8.1f} {m:>10.1f} {c:>12d}")

        # --- Also try: logit-space heuristic ---
        print(f"\n--- Logit-space heuristic: sigmoid(max(floor, 1-scale*std_logit) * logit_bar) ---")
        for floor in [0.2, 0.3, 0.4]:
            for scale in [0.3, 0.5, 0.7, 1.0]:
                def h_logit(ps, _f=floor, _s=scale):
                    logits = [logit(p) for p in ps]
                    logit_bar = np.mean(logits)
                    std_logit = np.std(logits)
                    a = max(_f, 1 - _s * std_logit)
                    return float(sigmoid(a * logit_bar))
                m, c = score_method(questions, h_logit)
                print(f"  floor={floor:.1f}, scale={scale:.1f}: "
                      f"metaculus={m:.1f}, catastrophic={c}")


if __name__ == "__main__":
    main()
