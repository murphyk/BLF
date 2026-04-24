#!/usr/bin/env python3
"""test_fb_leaderboard.py — Validate ABI computation against FB leaderboard.

Downloads a method's forecasts from ForecastBench, computes our ABI score,
and compares to the published leaderboard value. This validates that our
adjusted Brier score computation matches ForecastBench's methodology.

Usage:
    python3 src/testing/test_fb_leaderboard.py
    python3 src/testing/test_fb_leaderboard.py --method "external.Cassi-AI.2"
    python3 src/testing/test_fb_leaderboard.py --tolerance 1.0
"""

import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import os
import argparse
import json
import math
from datetime import datetime, timedelta
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FB_CACHE = os.path.join("data", "fb_cache", "forecastbench-processed-forecast-sets")
_FE_CACHE = "datasets/question_fixed_effects"

MARKET_SOURCES = {"infer", "manifold", "metaculus", "polymarket"}
DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}

# Published ABI values from the FB tournament leaderboard as of 2026-04-20
# https://www.forecastbench.org/leaderboards/#tournament
LEADERBOARD_ABI = {
    "external.Cassi-AI.2": 67.9,
    "OpenAI.gpt-5-2025-08-07_zero_shot_with_freeze_values": 67.2,
    "external.xai.1": 67.8,
    "external.Lightning-Rod-Labs": 67.2,
}


# ---------------------------------------------------------------------------
# Load forecasts
# ---------------------------------------------------------------------------

def load_all_forecasts(method_pattern: str) -> list[dict]:
    """Load all resolved binary forecasts for a method across all dates."""
    if not os.path.isdir(_FB_CACHE):
        print(f"ERROR: FB cache not found at {_FB_CACHE}")
        print(f"Run: python3 src/data/fb_leaderboard.py --xid <any-xid> to download")
        sys.exit(1)

    all_forecasts = []
    for date_dir in sorted(os.listdir(_FB_CACHE)):
        d = os.path.join(_FB_CACHE, date_dir)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if method_pattern not in fn:
                continue
            with open(os.path.join(d, fn)) as f:
                data = json.load(f)
            for fc in data["forecasts"]:
                if not fc.get("resolved"):
                    continue
                o = fc.get("resolved_to")
                if o is None or o not in (0, 1, 0.0, 1.0):
                    continue
                fc["_forecast_due_date"] = date_dir
                all_forecasts.append(fc)

    return all_forecasts


# ---------------------------------------------------------------------------
# Question fixed effects
# ---------------------------------------------------------------------------

def _date_str_to_ms(date_str: str) -> int:
    """Convert YYYY-MM-DD to Unix timestamp in milliseconds (UTC midnight)."""
    from datetime import timezone
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def load_fe() -> dict:
    """Load question fixed effects, keyed by (fdd_ms, source, id, horizon_days)."""
    today = datetime.today().date()

    # Try local cache first
    for delta in range(30):
        date_str = (today - timedelta(days=delta)).strftime("%Y-%m-%d")
        path = os.path.join(_FE_CACHE, f"question_fixed_effects.{date_str}.json")
        if os.path.exists(path):
            return _parse_fe_file(path)

    # Try downloading
    try:
        import requests
        for delta in range(14):
            date_str = (today - timedelta(days=delta)).strftime("%Y-%m-%d")
            url = f"https://www.forecastbench.org/assets/data/question-fixed-effects/question_fixed_effects.{date_str}.json"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                os.makedirs(_FE_CACHE, exist_ok=True)
                path = os.path.join(_FE_CACHE, f"question_fixed_effects.{date_str}.json")
                with open(path, "w") as f:
                    json.dump(resp.json(), f)
                print(f"  Downloaded FE for {date_str}")
                return _parse_fe_file(path)
    except Exception as e:
        print(f"  Warning: could not download FE: {e}")

    return {}


def _parse_fe_file(path: str) -> dict:
    """Parse FE file into lookup dict.

    FE records have:
        forecast_due_date: Unix timestamp in ms (int)
        source: str
        id: str
        horizon: float (days)
        question_fixed_effect: float
    """
    with open(path) as f:
        records = json.load(f)

    fe = {}
    for rec in records:
        fdd_ms = rec.get("forecast_due_date")
        source = rec.get("source", "")
        qid = str(rec.get("id", ""))
        horizon = rec.get("horizon")
        gamma = rec.get("question_fixed_effect", 0)

        if fdd_ms is None:
            continue

        # Key by (fdd_ms, source, id, horizon_int)
        h_int = int(round(horizon)) if horizon is not None else None
        fe[(int(fdd_ms), source, qid, h_int)] = gamma

    print(f"  Loaded {len(fe)} FE entries from {os.path.basename(path)}")
    return fe


def lookup_fe(fe: dict, fdd_str: str, source: str, qid: str,
              resolution_date_str: str) -> float | None:
    """Look up gamma for a dataset question."""
    if not fe:
        return None

    fdd_ms = _date_str_to_ms(fdd_str)

    # Compute horizon in days
    try:
        rd = datetime.strptime(resolution_date_str[:10], "%Y-%m-%d")
        fd = datetime.strptime(fdd_str[:10], "%Y-%m-%d")
        horizon = (rd - fd).days
    except (ValueError, TypeError):
        return None

    key = (fdd_ms, source, str(qid), horizon)
    return fe.get(key)


# ---------------------------------------------------------------------------
# ABI computation
# ---------------------------------------------------------------------------

def compute_abi(forecasts: list[dict], fe: dict, verbose: bool = False) -> dict:
    """Compute ABI matching ForecastBench methodology.

    Market: gamma_j = (market_value - outcome)^2
    Dataset: gamma_j from precomputed FE
    ABI = 100 * (1 - sqrt(ABS))
    ABS = mean(BS - gamma) + mean(gamma)  [rescaled so always-0.5 = 0.25]
    Overall = unweighted average of market and dataset ABI.
    """
    market_bs, market_gamma = [], []
    dataset_bs, dataset_gamma = [], []
    n_fe_miss = 0

    for fc in forecasts:
        source = fc["source"]
        p = fc["forecast"]
        o = float(fc["resolved_to"])
        bs = (p - o) ** 2
        fdd = fc["_forecast_due_date"]

        if source in MARKET_SOURCES:
            mv = fc.get("market_value_on_due_date")
            if mv is None:
                mv = fc.get("market_value_on_due_date_minus_one")
            if mv is None:
                continue
            gamma = (float(mv) - o) ** 2
            market_bs.append(bs)
            market_gamma.append(gamma)

        elif source in DATASET_SOURCES:
            rd = fc.get("resolution_date", "")[:10]
            gamma = lookup_fe(fe, fdd, source, str(fc["id"]), rd)
            if gamma is not None:
                dataset_bs.append(bs)
                dataset_gamma.append(gamma)
            else:
                n_fe_miss += 1

    if verbose and n_fe_miss:
        print(f"  Warning: {n_fe_miss} dataset forecasts missing FE (skipped)")

    results = {}

    if market_bs:
        mkt_abs = np.mean(np.array(market_bs) - np.array(market_gamma)) + np.mean(market_gamma)
        mkt_abi = 100 * (1 - math.sqrt(max(0, mkt_abs)))
        results["market"] = {"n": len(market_bs), "abs": float(mkt_abs), "abi": float(mkt_abi)}

    if dataset_bs:
        dat_abs = np.mean(np.array(dataset_bs) - np.array(dataset_gamma)) + np.mean(dataset_gamma)
        dat_abi = 100 * (1 - math.sqrt(max(0, dat_abs)))
        results["dataset"] = {"n": len(dataset_bs), "abs": float(dat_abs), "abi": float(dat_abi)}

    if "market" in results and "dataset" in results:
        overall_abi = (results["market"]["abi"] + results["dataset"]["abi"]) / 2
    elif "market" in results:
        overall_abi = results["market"]["abi"]
    elif "dataset" in results:
        overall_abi = results["dataset"]["abi"]
    else:
        overall_abi = 0

    results["overall"] = {"abi": float(overall_abi)}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate ABI computation against FB leaderboard")
    parser.add_argument("--method", type=str, default=None,
                        help="Specific method pattern to test (default: all known)")
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="Max allowable ABI difference (default: 3.0). "
                             "Exact match is not expected because: (1) our FE file "
                             "is a snapshot that may not cover all dates, and (2) the "
                             "FB leaderboard recomputes FE as new methods are added.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    methods = LEADERBOARD_ABI if args.method is None else {args.method: LEADERBOARD_ABI.get(args.method)}

    # Load FE once
    print("Loading question fixed effects...")
    fe = load_fe()

    n_pass, n_fail = 0, 0

    for method, expected_abi in methods.items():
        if expected_abi is None:
            print(f"\n{method}: no expected ABI in LEADERBOARD_ABI dict")
            continue

        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"Expected ABI (leaderboard): {expected_abi}")

        forecasts = load_all_forecasts(method)
        print(f"  Loaded {len(forecasts)} resolved binary forecasts")

        if len(forecasts) == 0:
            print(f"  SKIP: no forecasts found")
            continue

        results = compute_abi(forecasts, fe, verbose=args.verbose)

        computed_abi = results["overall"]["abi"]
        diff = abs(computed_abi - expected_abi)

        for split in ["market", "dataset"]:
            if split in results:
                r = results[split]
                print(f"  {split}: n={r['n']}, ABS={r['abs']:.4f}, ABI={r['abi']:.1f}")

        print(f"  Overall ABI: {computed_abi:.1f}")

        if diff <= args.tolerance:
            print(f"  PASS: computed={computed_abi:.1f}, expected={expected_abi:.1f}, diff={diff:.1f}")
            n_pass += 1
        else:
            print(f"  FAIL: computed={computed_abi:.1f}, expected={expected_abi:.1f}, diff={diff:.1f} > tol={args.tolerance}")
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"Results: {n_pass} passed, {n_fail} failed")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
