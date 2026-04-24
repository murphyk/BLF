#!/usr/bin/env python3
"""Re-aggregate trial forecasts using logit-space mean or LOO-tuned shrinkage.

Default: logit-space mean (parameter-free, alpha=1).
With --loo: LOO-tune (floor, scale) to minimize mean Brier score, then apply.
  This requires outcomes to be available in the question files.
  On FB, LOO typically finds alpha=1 (logit-space mean).
  On harder datasets (AIBQ2), LOO finds genuine shrinkage toward p=0.5.

No LLM calls are made — this is purely a post-processing step.

Usage:
    python3 src/core/reaggregate.py                          # logit-space mean
    python3 src/core/reaggregate.py --loo                    # LOO-tuned shrinkage
    python3 src/core/reaggregate.py --config pro-high-brave-c0-t1
    python3 src/core/reaggregate.py --dry-run
"""
import argparse
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
import numpy as np


FORECAST_DIR = "experiments/forecasts_raw"


def find_configs_with_trials():
    """Find all config directories that have trial_* subdirs."""
    configs = []
    for name in sorted(os.listdir(FORECAST_DIR)):
        d = os.path.join(FORECAST_DIR, name)
        if not os.path.isdir(d):
            continue
        if name.endswith("_calibrated"):
            continue
        trials = [x for x in os.listdir(d) if x.startswith("trial_") and os.path.isdir(os.path.join(d, x))]
        if trials:
            configs.append(name)
    return configs


def _logit_shrink(ps_arr, floor, scale):
    """Compute sigmoid(alpha * mean(logits)), alpha = max(floor, 1 - scale * std)."""
    clipped = np.clip(ps_arr, 0.001, 0.999)
    logits = np.log(clipped / (1 - clipped))
    logit_bar = float(np.mean(logits))
    std_logit = float(np.std(logits)) if len(logits) > 1 else 0.0
    alpha = max(floor, 1.0 - scale * std_logit)
    return float(1 / (1 + np.exp(-alpha * logit_bar))), alpha


def _collect_trials(config_name):
    """Collect all (source, fn, trial_fcs) for a config."""
    config_dir = os.path.join(FORECAST_DIR, config_name)
    trial_dirs = sorted([x for x in os.listdir(config_dir)
                         if x.startswith("trial_") and os.path.isdir(os.path.join(config_dir, x))])
    if not trial_dirs:
        return []

    questions = set()
    for td in trial_dirs:
        td_path = os.path.join(config_dir, td)
        for source in os.listdir(td_path):
            source_path = os.path.join(td_path, source)
            if not os.path.isdir(source_path):
                continue
            for fn in os.listdir(source_path):
                if fn.endswith(".json"):
                    questions.add((source, fn))

    result = []
    for source, fn in sorted(questions):
        trial_fcs = []
        for td in trial_dirs:
            fp = os.path.join(config_dir, td, source, fn)
            if not os.path.exists(fp):
                continue
            with open(fp) as f:
                fc = json.load(f)
            if fc.get("forecast") is not None:
                trial_num = int(td.split("_")[1])
                trial_fcs.append((trial_num, fc))
        if trial_fcs:
            result.append((source, fn, trial_fcs))
    return result


def loo_tune_shrinkage(config_name):
    """Find optimal (floor, scale) via LOO on resolved questions, minimizing BS.

    Returns (floor, scale). With floor=1, scale=0 this is logit-space mean.
    """
    records = _collect_trials(config_name)
    # Collect (logit_bar, std_logit, outcomes) for resolved questions
    data = []
    for source, fn, trial_fcs in records:
        # Load question to get outcome
        qpath = os.path.join("data", "questions", source, fn)
        if not os.path.exists(qpath):
            continue
        with open(qpath) as f:
            q = json.load(f)
        res = q.get("resolved_to")
        if res is None:
            continue

        ps = [fc["forecast"] for _, fc in trial_fcs]
        ps_arr = np.array(ps)
        clipped = np.clip(ps_arr, 0.001, 0.999)
        logits = np.log(clipped / (1 - clipped))
        logit_bar = float(np.mean(logits))
        std_logit = float(np.std(logits)) if len(logits) > 1 else 0.0

        outcomes = res if isinstance(res, list) else [res]
        for o in outcomes:
            if o is not None:
                data.append((logit_bar, std_logit, float(o)))

    if not data:
        print(f"    No resolved questions — using logit-space mean (floor=1, scale=0)")
        return 1.0, 0.0

    # Grid search
    floors = np.arange(0.0, 1.05, 0.1)
    scales = np.arange(0.0, 2.05, 0.2)
    best_bs, best_floor, best_scale = np.inf, 1.0, 0.0

    for floor in floors:
        for scale in scales:
            total_bs = 0.0
            for logit_bar, std_logit, o in data:
                alpha = max(floor, 1.0 - scale * std_logit)
                p = 1 / (1 + np.exp(-alpha * logit_bar))
                total_bs += (p - o) ** 2
            mean_bs = total_bs / len(data)
            if mean_bs < best_bs:
                best_bs = mean_bs
                best_floor = float(floor)
                best_scale = float(scale)

    # Compare to logit-space mean (floor=1, scale=0 => alpha=1)
    logit_bs = sum((1/(1+np.exp(-lb)) - o)**2 for lb, _, o in data) / len(data)
    print(f"    LOO tuning: {len(data)} resolved entries")
    print(f"    Logit-mean BS: {logit_bs:.6f}")
    print(f"    Best shrinkage BS: {best_bs:.6f} (floor={best_floor:.1f}, scale={best_scale:.1f})")
    print(f"    Improvement: {logit_bs - best_bs:+.6f}")

    return best_floor, best_scale


def reaggregate_config(config_name, floor=1.0, scale=0.0, dry_run=False):
    """Re-aggregate all questions for a config.

    floor=1.0, scale=0.0: logit-space mean (alpha=1, no shrinkage).
    Other values: shrinkage toward p=0.5 when trials disagree.
    """
    config_dir = os.path.join(FORECAST_DIR, config_name)
    records = _collect_trials(config_name)
    if not records:
        return 0, 0

    agg_label = "logit-mean" if (floor >= 1.0 and scale <= 0.0) else \
                f"loo-shrink(f={floor:.1f},s={scale:.1f})"
    n_updated = 0
    n_unchanged = 0

    for source, fn, trial_fcs in records:
        ps = [fc["forecast"] for _, fc in trial_fcs]
        ps_arr = np.array(ps)
        mean_p, alpha = _logit_shrink(ps_arr, floor, scale)
        arith_mean_p = float(np.mean(ps_arr))
        std_p = float(np.std(ps_arr))

        out_path = os.path.join(config_dir, source, fn)
        old_forecast = None
        if os.path.exists(out_path):
            with open(out_path) as f:
                old = json.load(f)
            old_forecast = old.get("forecast")

        base = dict(trial_fcs[0][1])
        base["forecast"] = mean_p
        base["tokens_in"] = sum(fc.get("tokens_in", 0) or 0 for _, fc in trial_fcs)
        base["tokens_out"] = sum(fc.get("tokens_out", 0) or 0 for _, fc in trial_fcs)
        base["elapsed_seconds"] = sum(fc.get("elapsed_seconds", 0) or 0 for _, fc in trial_fcs)
        base["n_steps"] = sum(fc.get("n_steps", 0) or 0 for _, fc in trial_fcs)
        base["trial_stats"] = {
            "n_trials": len(trial_fcs),
            "aggregation": agg_label,
            "shrinkage_alpha": alpha,
            "mean": mean_p,
            "arithmetic_mean": arith_mean_p,
            "median": float(np.median(ps_arr)),
            "min": float(np.min(ps_arr)),
            "max": float(np.max(ps_arr)),
            "std": std_p,
            "forecast_before_shrinkage": float(1 / (1 + np.exp(-np.mean(
                np.log(np.clip(ps_arr, 0.001, 0.999) / (1 - np.clip(ps_arr, 0.001, 0.999))))))),
            "forecasts": {t: fc["forecast"] for t, fc in trial_fcs},
        }

        changed = old_forecast is None or abs(old_forecast - mean_p) > 1e-10
        if changed:
            n_updated += 1
            if not dry_run:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(base, f, indent=2)
            if old_forecast is not None:
                delta = mean_p - old_forecast
                if abs(delta) > 0.001:
                    print(f"    {source}/{fn}: {old_forecast:.4f} -> {mean_p:.4f} (Δ={delta:+.4f})")
        else:
            n_unchanged += 1

    return n_updated, n_unchanged


def main():
    parser = argparse.ArgumentParser(
        description="Re-aggregate trial forecasts (logit-space mean or LOO-tuned shrinkage)")
    parser.add_argument("--config", type=str, help="Specific config to reaggregate")
    parser.add_argument("--loo", action="store_true",
                        help="LOO-tune (floor, scale) to minimize BS using resolved outcomes. "
                             "Without --loo, uses logit-space mean (alpha=1, parameter-free).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    if args.config:
        configs = [args.config]
    else:
        configs = find_configs_with_trials()

    method = "LOO-tuned shrinkage" if args.loo else "logit-space mean"
    print(f"Re-aggregating {len(configs)} configs with {method}...")
    if args.dry_run:
        print("(DRY RUN — no files will be written)")

    total_updated = 0
    total_unchanged = 0
    for config in configs:
        print(f"  {config}:")
        if args.loo:
            floor, scale = loo_tune_shrinkage(config)
        else:
            floor, scale = 1.0, 0.0
        n_up, n_unch = reaggregate_config(config, floor=floor, scale=scale,
                                           dry_run=args.dry_run)
        print(f"    {n_up} updated, {n_unch} unchanged")
        total_updated += n_up
        total_unchanged += n_unch

    print(f"\nTotal: {total_updated} updated, {total_unchanged} unchanged")


if __name__ == "__main__":
    main()
