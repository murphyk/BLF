#!/usr/bin/env python3
"""aggregate.py — Compute trial-aggregation variants against forecasts_final/.

Reads `raw_trials` from each entry in experiments/forecasts_final/{date}/
{config}.json, computes one forecast per (variant, entry), and stores the
result at the file level:

    "forecasts_aggregated": {
        "mean:1":       [0.71, 0.58, ...],   // one value per entry, aligned with forecasts[]
        "mean:5":       [0.72, 0.63, ...],
        "logit-mean:5": [0.73, 0.64, ...],
        "shrink:5":     [0.68, 0.61, ...],
        "shrink5-loo":  [0.65, 0.58, ...]    // label-dependent, only when resolved
    }

The base `forecast` field (set by collate.py) is NOT overwritten — it stays
the default aggregation. Aggregate-variant columns are opt-in via
`eval.py --add-aggregation` or bracket syntax (`"pro-high[mean:5]"`).

Variants:
    mean:n          arithmetic mean over n-sized subsets of trials
    logit-mean:n    arithmetic mean in logit space over n-sized subsets
    shrink:n        std-shrinkage-to-0.5 over n-sized subsets (James-Stein-ish;
                    shrinkage strength λ = 1 − 0.7·std(logits), floored at 0.3)
    shrink<n>-loo   global λ fit by leave-one-out on all resolved entries
                    (requires labels — NOT usable at live-submission time)

Usage:
    python3 src/core/aggregate.py --xid my-xid
    python3 src/core/aggregate.py --xid my-xid --variants "mean:1,mean:5,shrink:5"
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
_SHRINK_FLOOR = 0.3
_SHRINK_SCALE = 0.7
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


def _agg_shrink(ps: list[float], lam: float | None = None) -> float:
    if len(ps) <= 1:
        return float(ps[0]) if ps else 0.5
    xs = np.array([_logit(p) for p in ps], dtype=float)
    mean = float(xs.mean())
    std = float(xs.std())
    a = lam if lam is not None else max(_SHRINK_FLOOR, 1.0 - _SHRINK_SCALE * std)
    return _sigmoid(a * mean)


_AGGS = {
    "mean":       _agg_mean,
    "logit-mean": _agg_logit_mean,
    "shrink":     _agg_shrink,
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
                     lam: float | None = None) -> float | None:
    """Compute expected variant forecast over subsets of raw_trials."""
    trials = [p for p in raw_trials if p is not None]
    if not trials:
        return None
    k = len(trials)
    n_use = max(1, min(n, k))
    fn = _AGGS.get(method)
    if fn is None:
        return None
    combos = _enumerate_subsets(k, n_use)
    if method == "shrink":
        vals = [fn([trials[i] for i in c], lam) for c in combos]
    else:
        vals = [fn([trials[i] for i in c]) for c in combos]
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# LOO shrinkage
# ---------------------------------------------------------------------------

def _fit_loo_shrinkage(records: list[tuple[list[float], float]], n: int) -> float:
    """Find λ ∈ (0.05, 1.0] that minimizes LOO mean Brier over all entries.

    records: list of (raw_trials, outcome) with outcome in {0, 1}.
    Uses subsets of size min(n, len(trials)) per entry and evaluates the
    mean Brier score across all (entry, subset) pairs.
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


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def _parse_variant(v: str) -> tuple[str, int]:
    """Parse 'mean:5' → ('mean', 5), 'shrink5-loo' → ('shrink-loo', 5)."""
    if v.endswith("-loo"):
        head = v[:-4]
        method = "shrink-loo"
        n = int("".join(c for c in head if c.isdigit()) or "5")
        return method, n
    method, _, num = v.partition(":")
    return method.strip(), int(num) if num else 1


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

    # Any *-loo variant is fit once globally against labeled data.
    loo_lams: dict[str, float] = {}
    for v in variants:
        method, n = _parse_variant(v)
        if method == "shrink-loo":
            records = _gather_label_records(config_name, n, root=root)
            if len(records) < 5:
                print(f"  SKIP (loo): only {len(records)} labeled records for {config_name}/{v}")
                loo_lams[v] = 1.0
            else:
                loo_lams[v] = _fit_loo_shrinkage(records, n)
                if verbose:
                    print(f"  {config_name}/{v}: λ = {loo_lams[v]:.3f} (from {len(records)} records)")

    n_updated = 0
    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        entries = payload.get("forecasts", [])
        aggs = payload.get("forecasts_aggregated", {})
        for v in variants:
            method, n = _parse_variant(v)
            key = v
            col: list[float | None] = []
            for e in entries:
                raw = e.get("raw_trials") or []
                if method == "shrink-loo":
                    agg = _aggregate_entry(raw, "shrink", n, lam=loo_lams.get(v, 1.0))
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


DEFAULT_VARIANTS = "mean:1,mean:3,mean:5,logit-mean:5,shrink:5"


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
