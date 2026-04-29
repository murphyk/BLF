#!/usr/bin/env python3
"""aggregate.py — Compute trial-aggregation variants against forecasts_final/.

Reads `raw_trials` from each entry in experiments/forecasts_final/{date}/
{config}.json, computes one forecast per (variant, entry), and stores the
result at the file level:

    "forecasts_aggregated": {
        "mean:1":              [0.71, 0.58, ...],
        "mean:5":              [0.72, 0.63, ...],
        "logit-mean:5":        [0.73, 0.64, ...],
        "shrink-std-aibq2:5":  [0.68, 0.61, ...],
        "shrink-std-loo:5":    [0.69, 0.62, ...],   // label-dependent
        "shrink-alpha-loo:5":  [0.70, 0.62, ...]    // label-dependent
    }

The base `forecast` field (set by collate.py) is NOT overwritten — it stays
the default aggregation. Aggregate-variant columns are opt-in via
`eval.py --add-aggregation` or bracket syntax (`"pro-high[mean:5]"`).

Variants:
    mean:n              arithmetic mean of trial probabilities over n-sized subsets
    logit-mean:n        sigmoid(mean(logits)) over n-sized subsets
    shrink-std-aibq2:n  std-based shrinkage in logit space with HARDCODED
                        (f=0.3, c=0.7), as used for AIBQ2 in the paper:
                        α = max(0.3, 1 − 0.7·std(logits)),
                        p_hat = sigmoid(α · mean(logits)). The constants were
                        chosen during AIBQ2 paper development to approximate
                        what shrink-std-loo would pick.
    shrink-std-loo:n    same per-question α(q) = max(f, 1 − c · std_logit(q))
                        but with (f, c) tuned on the labeled set via an
                        11×11 grid search (f ∈ {0..1, step 0.1};
                        c ∈ {0..2, step 0.2}), minimizing mean Brier.
                        Subsumes shrink-std-aibq2 (which is one operating
                        point) and shrink-alpha-loo when LOO selects c=0.
                        Requires resolved outcomes.
    shrink-alpha-loo:n  single GLOBAL α ∈ (0.05, 1.0] fit by a 20-point 1-D
                        grid search to minimize mean Brier on the labeled
                        set; no per-question std dependence. On FB this
                        variant historically picked α=1 (= logit-mean).
                        Requires resolved outcomes.
    median:n            median of trial probabilities over n-sized subsets

Usage:
    python3 src/core/aggregate.py --xid my-xid
    python3 src/core/aggregate.py --xid my-xid --variants "mean:1,mean:5,shrink-std-loo:5"
    python3 src/core/aggregate.py --config pro-high-brave-c1-t1
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import sys
from typing import Iterable
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import numpy as np

from config.paths import FORECASTS_FINAL_DIR, XIDS_DIR


# ---------------------------------------------------------------------------
# Aggregation primitives
# ---------------------------------------------------------------------------

_EPS = 1e-4
# AIBQ2 hardcoded operating point for shrink-std-aibq2: chosen during
# paper development to approximate what shrink-std-loo selects on AIBQ2.
# On FB the LOO optimum is (1, 0) (= logit-mean) so these constants
# are mainly useful on AIBQ2-style noisier exams.
_AIBQ2_SHRINK_FLOOR = 0.3
_AIBQ2_SHRINK_SCALE = 0.7
_MAX_SUBSETS = 200  # cap to avoid combinatorial blowup when K is large


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _agg_mean(ps: list[float]) -> float:
    return float(np.mean(ps))


def _agg_logit_mean(ps: list[float]) -> float:
    xs = [_logit(p) for p in ps]
    return _sigmoid(float(np.mean(xs)))


def _agg_shrink_std(ps: list[float],
                    floor: float = _AIBQ2_SHRINK_FLOOR,
                    scale: float = _AIBQ2_SHRINK_SCALE,
                    alpha_override: float | None = None) -> float:
    """Per-question std-based shrinkage in logit space.

    Computes p_hat = sigmoid(α · mean(logits)), where:
      - if alpha_override is set, α = alpha_override (used by
        shrink-alpha-loo, which fits one global α and bypasses std);
      - otherwise α = max(floor, 1 − scale · std(logits)).

    Defaults to the AIBQ2-paper operating point (floor=0.3, scale=0.7).
    """
    if len(ps) <= 1:
        return float(ps[0]) if ps else 0.5
    xs = np.array([_logit(p) for p in ps], dtype=float)
    mean = float(xs.mean())
    std = float(xs.std())
    a = (alpha_override if alpha_override is not None
         else max(floor, 1.0 - scale * std))
    return _sigmoid(a * mean)


def _agg_median(ps: list[float]) -> float:
    return float(np.median(ps))


_AGGS = {
    "mean":             _agg_mean,
    "logit-mean":       _agg_logit_mean,
    "shrink-std-aibq2": _agg_shrink_std,  # uses default (0.3, 0.7)
    "median":           _agg_median,
    # shrink-std-loo and shrink-alpha-loo go through _agg_shrink_std
    # with fitted parameters; not entered directly into _AGGS.
}


def _enumerate_subsets(k: int, n: int) -> list[tuple[int, ...]]:
    """Return up to _MAX_SUBSETS size-n combinations of range(k)."""
    if n >= k:
        return [tuple(range(k))]
    combos = list(itertools.combinations(range(k), n))
    if len(combos) > _MAX_SUBSETS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(combos), _MAX_SUBSETS, replace=False)
        combos = [combos[i] for i in idx]
    return combos


def _aggregate_entry(raw_trials: list[float], method: str, n: int,
                     floor: float | None = None,
                     scale: float | None = None,
                     alpha_override: float | None = None) -> float | None:
    """Compute expected variant forecast over subsets of raw_trials.

    For the shrinkage methods (shrink-std-aibq2, shrink-std-loo,
    shrink-alpha-loo) the call site supplies the operating point:
      - shrink-std-aibq2: defaults (floor=0.3, scale=0.7) used.
      - shrink-std-loo: (floor, scale) come from _fit_std_shrinkage.
      - shrink-alpha-loo: alpha_override comes from _fit_global_shrinkage.
    """
    trials = [p for p in raw_trials if p is not None]
    if not trials:
        return None
    k = len(trials)
    n_use = max(1, min(n, k))
    combos = _enumerate_subsets(k, n_use)
    if method in ("shrink-std-aibq2", "shrink-std-loo", "shrink-alpha-loo"):
        f = floor if floor is not None else _AIBQ2_SHRINK_FLOOR
        s = scale if scale is not None else _AIBQ2_SHRINK_SCALE
        vals = [_agg_shrink_std([trials[i] for i in c], f, s, alpha_override)
                for c in combos]
    else:
        fn = _AGGS.get(method)
        if fn is None:
            return None
        vals = [fn([trials[i] for i in c]) for c in combos]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# LOO shrinkage
# ---------------------------------------------------------------------------

def _fit_global_shrinkage(records: list[tuple[list[float], float]], n: int) -> float:
    """Single global α ∈ (0.05, 1.0] that minimizes mean Brier on the
    pooled labeled set. Used by the shrink-alpha-loo variant.

    Each (raw_trials, outcome) record is expanded into all C(K, n)
    subsets of trials. For each candidate λ in a 20-point grid, we
    compute p_i = sigmoid(λ · mean(logits_i)) on every (record,
    subset) pair and take the mean Brier across all of them; pick λ
    minimizing that.

    Despite the "loo" name in the variant key, this fits on the full
    labeled set rather than leave-one-out per held-out question. The
    name reflects "labeled-set α fit" rather than strict LOO; refit
    per held-out question is a TODO if needed for paper claims.

    records: list of (raw_trials, outcome) with outcome in {0, 1}.
    """
    # Collect (trials-subset-aggregate-given-λ, outcome) contributions.
    # Pre-enumerate subsets once per record.
    per_entry: list[tuple[list[float], float]] = []  # (logits_list, outcome)
    for trials, outcome in records:
        if outcome is None:
            continue
        ts = [p for p in trials if p is not None]
        if not ts:
            continue
        for c in _enumerate_subsets(len(ts), min(n, len(ts))):
            sub = [ts[i] for i in c]
            if len(sub) <= 1:
                per_entry.append(([_logit(sub[0])], float(outcome)))
            else:
                xs = [_logit(p) for p in sub]
                per_entry.append((xs, float(outcome)))

    if not per_entry:
        return 1.0

    def brier_at(lam: float) -> float:
        total = 0.0
        for xs, o in per_entry:
            if len(xs) == 1:
                p = _sigmoid(lam * xs[0])
            else:
                mean = sum(xs) / len(xs)
                p = _sigmoid(lam * mean)
            total += (p - o) ** 2
        return total / len(per_entry)

    # Simple 1-D grid search — cheap and robust.
    best_l, best_b = 1.0, float("inf")
    for lam_i in np.linspace(0.05, 1.0, 20):
        b = brier_at(lam_i)
        if b < best_b:
            best_b, best_l = b, float(lam_i)
    return best_l


def _fit_std_shrinkage(records: list[tuple[list[float], float]], n: int
                       ) -> tuple[float, float]:
    """Fit (floor, scale) for shrink-std-loo via 11×11 grid search on
    the labeled set, minimizing mean Brier of the per-question
    α(q) = max(floor, 1 − scale · std_logit(q)) estimator.

    floor ∈ {0.0, 0.1, ..., 1.0}, scale ∈ {0.0, 0.2, ..., 2.0}.
    Returns (best_floor, best_scale). When the optimum is (1.0, 0.0)
    the variant collapses to logit-space averaging; (0.3, 0.7)
    recovers the AIBQ2 hardcoded operating point.

    Like _fit_global_shrinkage above, this is a labeled-set fit
    (not strict per-question leave-one-out) — the historical "LOO"
    naming reflects "fit on the held-out backtest set" rather than
    per-question CV.
    """
    # Pre-compute (logits-list, outcome) for every (record, subset).
    per_entry: list[tuple[list[float], float]] = []
    for trials, outcome in records:
        if outcome is None:
            continue
        ts = [p for p in trials if p is not None]
        if not ts:
            continue
        for c in _enumerate_subsets(len(ts), min(n, len(ts))):
            sub = [ts[i] for i in c]
            per_entry.append(([_logit(p) for p in sub], float(outcome)))

    if not per_entry:
        return 1.0, 0.0

    def brier_at(floor: float, scale: float) -> float:
        total = 0.0
        for xs, o in per_entry:
            if len(xs) <= 1:
                p = _sigmoid(xs[0]) if xs else 0.5
            else:
                arr = np.asarray(xs)
                m = float(arr.mean())
                s = float(arr.std())
                a = max(floor, 1.0 - scale * s)
                p = _sigmoid(a * m)
            total += (p - o) ** 2
        return total / len(per_entry)

    floors = np.arange(0.0, 1.05, 0.1)
    scales = np.arange(0.0, 2.05, 0.2)
    best_b = float("inf")
    best_f, best_s = 1.0, 0.0
    for f in floors:
        for s in scales:
            b = brier_at(float(f), float(s))
            if b < best_b:
                best_b = b
                best_f, best_s = float(f), float(s)
    return best_f, best_s


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

CANONICAL_METHODS = {
    "mean",
    "logit-mean",
    "shrink-std-aibq2",
    "shrink-std-loo",
    "shrink-alpha-loo",
    "median",
}

# Renames applied 2026-04-29 — old keys are silently dropped from
# already-written forecasts_aggregated dicts on the next aggregate run.
_LEGACY_RENAMES = {
    "shrink-std": "shrink-std-aibq2",
    "shrink-fit": "shrink-alpha-loo",
}


def _parse_variant(v: str) -> tuple[str, int]:
    """Parse a variant key into (method, n). Canonical names only."""
    method, _, num = v.partition(":")
    method = method.strip()
    if method not in CANONICAL_METHODS:
        raise ValueError(
            f"Unknown variant method {method!r} (in {v!r}). "
            f"Valid: {sorted(CANONICAL_METHODS)}.")
    return method, int(num) if num else 1


def _gather_label_records(config_name: str, n: int,
                          root: str = FORECASTS_FINAL_DIR
                          ) -> list[tuple[list[float], float]]:
    """Collect (raw_trials, outcome) pairs across all dates for a config."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", f"{config_name}.json"))):
        with open(p) as f:
            payload = json.load(f)
        for e in payload.get("forecasts", []):
            o = e.get("resolved_to")
            if isinstance(o, list):
                o = o[0] if o else None
            if o is None:
                continue
            rt = e.get("raw_trials") or []
            if rt:
                out.append((rt, float(o)))
    return out


def aggregate_config(config_name: str, variants: list[str],
                     root: str = FORECASTS_FINAL_DIR,
                     verbose: bool = False) -> int:
    """Compute every variant for config_name, store in each date-file at
    top level as forecasts_aggregated: {key: [p0, p1, ...]}.

    Returns number of date-files updated.
    """
    paths = sorted(glob.glob(os.path.join(root, "*", f"{config_name}.json")))
    if not paths:
        print(f"  SKIP: no forecasts_final files for {config_name}")
        return 0

    # Fit any LOO-style variants on the labeled set up-front.
    # shrink-alpha-loo: single global α via 1-D 20-point grid.
    # shrink-std-loo:   (floor, scale) via 11×11 grid over the
    #                   per-question α(q) = max(f, 1 − c · std_logit(q)).
    fit_alpha: dict[str, float] = {}      # variant key -> α
    fit_fc: dict[str, tuple[float, float]] = {}  # variant key -> (f, c)
    for v in variants:
        method, n = _parse_variant(v)
        if method == "shrink-alpha-loo":
            records = _gather_label_records(config_name, n, root=root)
            if len(records) < 5:
                print(f"  SKIP (fit): only {len(records)} labeled records for {config_name}/{v}")
                fit_alpha[v] = 1.0
            else:
                fit_alpha[v] = _fit_global_shrinkage(records, n)
                if verbose:
                    print(f"  {config_name}/{v}: α = {fit_alpha[v]:.3f} "
                          f"(global, from {len(records)} records)")
        elif method == "shrink-std-loo":
            records = _gather_label_records(config_name, n, root=root)
            if len(records) < 5:
                print(f"  SKIP (fit): only {len(records)} labeled records for {config_name}/{v}")
                fit_fc[v] = (1.0, 0.0)
            else:
                fit_fc[v] = _fit_std_shrinkage(records, n)
                if verbose:
                    f, s = fit_fc[v]
                    print(f"  {config_name}/{v}: (f, c) = ({f:.1f}, {s:.1f}) "
                          f"(LOO grid, from {len(records)} records)")

    n_updated = 0
    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        entries = payload.get("forecasts", [])
        # Migrate / drop legacy variant keys lingering from older
        # aggregate.py runs (e.g. "shrink-std:5" → "shrink-std-aibq2:5",
        # "shrink-fit:5" → "shrink-alpha-loo:5", or unrecognized).
        existing = payload.get("forecasts_aggregated", {})
        aggs: dict = {}
        for k, val in existing.items():
            try:
                method, n = _parse_variant(k)
                if method in CANONICAL_METHODS:
                    aggs[k] = val
                continue
            except ValueError:
                pass
            # Try a legacy rename: "shrink-std:5" -> "shrink-std-aibq2:5"
            old_method, _, num = k.partition(":")
            if old_method in _LEGACY_RENAMES:
                new_key = f"{_LEGACY_RENAMES[old_method]}:{num}" if num else _LEGACY_RENAMES[old_method]
                aggs[new_key] = val
            # else: silently drop unrecognized
        for v in variants:
            method, n = _parse_variant(v)
            key = v
            col: list[float | None] = []
            for e in entries:
                raw = e.get("raw_trials") or []
                if method == "shrink-alpha-loo":
                    agg = _aggregate_entry(raw, method, n,
                                           alpha_override=fit_alpha.get(v, 1.0))
                elif method == "shrink-std-loo":
                    f, s = fit_fc.get(v, (1.0, 0.0))
                    agg = _aggregate_entry(raw, method, n, floor=f, scale=s)
                else:
                    agg = _aggregate_entry(raw, method, n)
                col.append(agg)
            aggs[key] = col
        payload["forecasts_aggregated"] = aggs
        with open(p, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        n_updated += 1

    if verbose:
        print(f"  {config_name}: updated {n_updated} date-files with {len(variants)} variants")
    return n_updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configs_in_xid(xid: str) -> list[str]:
    from config.config import resolve_config, pprint as cfg_pprint
    with open(os.path.join(XIDS_DIR, f"{xid}.json")) as f:
        xid_data = json.load(f)
    labels: list[str] = []
    for field in ("config", "eval", "calibrate"):
        for entry in xid_data.get(field, []):
            name = entry.split("[")[0]
            # Delta strings use ":" and "/" as separators. Hyphenated directory
            # names are passed through unchanged.
            if ":" in name or "/" in name:
                try:
                    cfg = resolve_config(name)
                    labels.append(cfg_pprint(cfg))
                    continue
                except Exception:
                    pass
            labels.append(name)
    return sorted(set(labels))


DEFAULT_VARIANTS = ("mean:1,mean:3,mean:5,logit-mean:5,"
                    "shrink-std-aibq2:5,shrink-std-loo:5,shrink-alpha-loo:5")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute trial-aggregation variants in forecasts_final/ files.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--xid", help="Aggregate every config referenced by this xid")
    g.add_argument("--config", help="Aggregate a single config by directory name")
    g.add_argument("--all", action="store_true",
                   help="Aggregate every top-level config that has forecasts_final files")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help=f"Comma-separated variant keys (default: {DEFAULT_VARIANTS})")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.xid:
        configs = _configs_in_xid(args.xid)
    elif args.config:
        configs = [args.config]
    else:
        all_files = glob.glob(os.path.join(FORECASTS_FINAL_DIR, "*", "*.json"))
        configs = sorted({os.path.splitext(os.path.basename(p))[0] for p in all_files})

    total = 0
    for c in configs:
        total += aggregate_config(c, variants, verbose=args.verbose)
    print(f"aggregate: {len(configs)} configs, {total} date-files updated, "
          f"{len(variants)} variants written.")


if __name__ == "__main__":
    main()
