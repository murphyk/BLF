#!/usr/bin/env python3
"""eval_ts_models.py — Evaluate time-series forecasting models on dbnomics questions.

Tests various statistical models for predicting P(temp on date X > temp on date Y)
using LOO cross-validation on the actual dbnomics data.

Usage:
    python3 src/eval_ts_models.py --exam tranche-a
    python3 src/eval_ts_models.py --exam tranche-a --source dbnomics --verbose
"""

import argparse
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm


def load_dbnomics_questions(exam_name):
    """Load all dbnomics questions from an exam."""
    with open(f"data/exams/{exam_name}/indices.json") as f:
        exam = json.load(f)

    questions = []
    for qid in exam.get("dbnomics", []):
        safe = re.sub(r'[/\\:]', '_', str(qid))
        qpath = f"data/questions/dbnomics/{safe}.json"
        if not os.path.exists(qpath):
            continue
        with open(qpath) as f:
            q = json.load(f)

        # Extract URL from resolution criteria
        rc = q.get("resolution_criteria", "")
        url_match = re.search(r'https://db\.nomics\.world/\S+', rc)
        url = url_match.group(0).rstrip('.,;)') if url_match else None

        # Also try background
        if not url:
            bg = q.get("background", "")
            url_match = re.search(r'https://db\.nomics\.world/\S+', bg)
            url = url_match.group(0).rstrip('.,;)') if url_match else None

        questions.append({
            "id": qid,
            "url": url,
            "forecast_due_date": q["forecast_due_date"],
            "resolution_dates": q.get("resolution_dates", []),
            "resolved_to": q.get("resolved_to", []),
        })

    return questions


def fetch_all_data(url, end_date, max_years=10):
    """Fetch as much history as possible for a dbnomics series."""
    from agent.data_tools import fetch_dbnomics
    # Temporarily patch to get more history
    import agent.data_tools as data_tools
    original_days = 730
    # Monkey-patch to get more years
    start = (pd.to_datetime(end_date) - pd.Timedelta(days=365 * max_years)).strftime("%Y-%m-%d")

    try:
        import dbnomics
    except ImportError:
        return None

    try:
        path = url.split("db.nomics.world/")[1].rstrip("/")
        if "/" in path:
            provider, dataset, series = path.split("/", 2)
        elif "_" in path:
            parts = path.split("_", 2)
            if len(parts) == 3:
                provider, dataset, series = parts
            else:
                return None
        else:
            return None
    except (IndexError, ValueError):
        return None

    try:
        df = dbnomics.fetch_series(provider, dataset, series)
    except Exception as e:
        print(f"  WARNING: fetch failed: {e}")
        return None

    if df is None or df.empty:
        return None

    period_col = next((c for c in ("period", "original_period") if c in df.columns), None)
    if period_col is None:
        return None

    df = df[[period_col, "value"]].rename(columns={period_col: "ds", "value": "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")
    df = df[df["ds"] <= pd.to_datetime(end_date)]

    return df


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def model_always_half(df, forecast_date, resolution_date, threshold):
    """Baseline: always predict 0.5."""
    return 0.5


def model_same_period_prior_years(df, forecast_date, resolution_date, threshold):
    """Look up same calendar date in prior years, compute frequency."""
    rd = pd.to_datetime(resolution_date)
    vals = []
    for year_offset in range(1, 20):
        try:
            target = rd.replace(year=rd.year - year_offset)
        except ValueError:
            continue
        mask = (df["ds"] >= target - pd.Timedelta(days=7)) & \
               (df["ds"] <= target + pd.Timedelta(days=7))
        nearby = df[mask]
        if not nearby.empty:
            closest_idx = (nearby["ds"] - target).abs().idxmin()
            vals.append(float(nearby.loc[closest_idx, "y"]))

    if not vals:
        return 0.5

    n_exceeded = sum(1 for v in vals if v > threshold)
    # Laplace smoothing
    return (n_exceeded + 1) / (len(vals) + 2)


def model_linear_trend(df, forecast_date, resolution_date, threshold):
    """Fit linear trend to last year, extrapolate."""
    fd = pd.to_datetime(forecast_date)
    recent = df[df["ds"] >= fd - pd.Timedelta(days=365)]
    if len(recent) < 10:
        return 0.5

    t0 = recent["ds"].min()
    x = (recent["ds"] - t0).dt.days.values.astype(float)
    y = recent["y"].values.astype(float)

    coeffs = np.polyfit(x, y, 1)
    residuals = y - np.polyval(coeffs, x)
    std = max(float(residuals.std()), 1e-6)

    x_rd = float((pd.to_datetime(resolution_date) - t0).days)
    predicted = float(np.polyval(coeffs, x_rd))

    return float(scipy_norm.sf(threshold, loc=predicted, scale=std))


def model_seasonal_harmonic(df, forecast_date, resolution_date, threshold):
    """Fit linear + annual sinusoidal model."""
    fd = pd.to_datetime(forecast_date)
    # Use all available data (not just last year)
    if len(df) < 30:
        return 0.5

    t0 = df["ds"].min()
    x_days = (df["ds"] - t0).dt.days.values.astype(float)
    y = df["y"].values.astype(float)

    # Features: intercept, linear, sin(2π/365 * t), cos(2π/365 * t)
    omega = 2 * np.pi / 365.25
    X = np.column_stack([
        np.ones(len(x_days)),
        x_days,
        np.sin(omega * x_days),
        np.cos(omega * x_days),
    ])

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.5

    residuals = y - X @ beta
    std = max(float(residuals.std()), 1e-6)

    # Predict at resolution date
    x_rd = float((pd.to_datetime(resolution_date) - t0).days)
    X_rd = np.array([1, x_rd, np.sin(omega * x_rd), np.cos(omega * x_rd)])
    predicted = float(X_rd @ beta)

    return float(scipy_norm.sf(threshold, loc=predicted, scale=std))


def model_seasonal_harmonic2(df, forecast_date, resolution_date, threshold):
    """Fit linear + annual + semi-annual sinusoidal model."""
    fd = pd.to_datetime(forecast_date)
    if len(df) < 50:
        return 0.5

    t0 = df["ds"].min()
    x_days = (df["ds"] - t0).dt.days.values.astype(float)
    y = df["y"].values.astype(float)

    omega1 = 2 * np.pi / 365.25
    omega2 = 4 * np.pi / 365.25
    X = np.column_stack([
        np.ones(len(x_days)),
        x_days,
        np.sin(omega1 * x_days),
        np.cos(omega1 * x_days),
        np.sin(omega2 * x_days),
        np.cos(omega2 * x_days),
    ])

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.5

    residuals = y - X @ beta
    std = max(float(residuals.std()), 1e-6)

    x_rd = float((pd.to_datetime(resolution_date) - t0).days)
    X_rd = np.array([1, x_rd, np.sin(omega1 * x_rd), np.cos(omega1 * x_rd),
                     np.sin(omega2 * x_rd), np.cos(omega2 * x_rd)])
    predicted = float(X_rd @ beta)

    return float(scipy_norm.sf(threshold, loc=predicted, scale=std))


def model_combined(df, forecast_date, resolution_date, threshold):
    """Combined: 50% seasonal_harmonic + 50% same_period."""
    p1 = model_seasonal_harmonic(df, forecast_date, resolution_date, threshold)
    p2 = model_same_period_prior_years(df, forecast_date, resolution_date, threshold)
    return 0.5 * p1 + 0.5 * p2


def model_current(df, forecast_date, resolution_date, threshold):
    """Current system: linear + same_period with source-specific weights."""
    from agent.source_tools import _analyze_trend
    # Use the actual _analyze_trend function
    result = _analyze_trend(df, threshold, resolution_date, "dbnomics")
    # Extract the combined estimate
    import re
    m = re.search(r"Combined estimate:.*?=\s*([\d.]+)", result)
    if m:
        return float(m.group(1))
    return 0.5


MODELS = {
    "always_0.5": model_always_half,
    "same_period": model_same_period_prior_years,
    "linear_trend": model_linear_trend,
    "harmonic_1": model_seasonal_harmonic,
    "harmonic_2": model_seasonal_harmonic2,
    "combined": model_combined,
    "current": model_current,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate TS models on dbnomics")
    parser.add_argument("--exam", default="tranche-a", help="Exam name")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-years", type=int, default=10,
                        help="Max years of history to fetch")
    args = parser.parse_args()

    questions = load_dbnomics_questions(args.exam)
    print(f"Loaded {len(questions)} dbnomics questions from {args.exam}")

    # Fetch data for all questions
    print("Fetching data...")
    data_cache = {}
    for q in questions:
        url = q["url"]
        if url and url not in data_cache:
            df = fetch_all_data(url, q["forecast_due_date"], args.max_years)
            data_cache[url] = df
            if df is not None:
                print(f"  {url.split('/')[-1]}: {len(df)} rows, "
                      f"{df['ds'].min().strftime('%Y')} to {df['ds'].max().strftime('%Y-%m-%d')}")

    print(f"\nEvaluating {len(MODELS)} models on {len(questions)} questions...")

    # Evaluate each model
    model_scores = {name: [] for name in MODELS}

    for q in questions:
        url = q["url"]
        df = data_cache.get(url)
        if df is None or df.empty:
            continue

        fdd = q["forecast_due_date"]
        res_dates = q["resolution_dates"]
        outcomes = q["resolved_to"]
        if not isinstance(outcomes, list):
            outcomes = [outcomes]

        # Get the actual threshold (last value on forecast date)
        fd_mask = df["ds"] <= pd.to_datetime(fdd)
        if not fd_mask.any():
            continue
        threshold = float(df[fd_mask]["y"].iloc[-1])

        # Restrict data to before forecast date
        df_train = df[df["ds"] <= pd.to_datetime(fdd)]

        for model_name, model_fn in MODELS.items():
            question_bis = []
            for rd, outcome in zip(res_dates, outcomes):
                if outcome is None:
                    continue
                try:
                    p = model_fn(df_train, fdd, rd, threshold)
                    p = max(0.05, min(0.95, p))
                except Exception as e:
                    if args.verbose:
                        print(f"  {model_name} failed on {q['id'][:30]}/{rd}: {e}")
                    p = 0.5

                bi = 1 - abs(p - float(outcome))
                question_bis.append(bi)

                if args.verbose and model_name in ("harmonic_1", "current"):
                    print(f"  {model_name} {q['id'][:30]:30s} rd={rd} "
                          f"p={p:.3f} o={outcome} BI={bi:.3f}")

            if question_bis:
                model_scores[model_name].append(np.mean(question_bis))

    # Print results
    print(f"\n{'Model':<20s}  {'Mean BI':>8s}  {'Std':>8s}  {'n':>4s}")
    print("-" * 45)
    for name in sorted(model_scores, key=lambda n: -np.mean(model_scores[n]) if model_scores[n] else 0):
        scores = model_scores[name]
        if scores:
            print(f"{name:<20s}  {np.mean(scores):>8.3f}  {np.std(scores):>8.3f}  {len(scores):>4d}")

    # Also show Cassi for reference
    print("\n--- Reference ---")
    from core.eval import load_exam, load_and_score
    import math
    exam = load_exam(args.exam)
    for label, cfg in [("Cassi", "fb-cassi-ai-2"), ("Our full", "pro-high-brave-c0-t1")]:
        scores = load_and_score(cfg, exam, ["brier-index"])
        if not scores:
            continue
        dbnom = [rec["brier-index"] for (s, q), rec in scores.items()
                 if s == "dbnomics" and "brier-index" in rec
                 and not (isinstance(rec["brier-index"], float) and math.isnan(rec["brier-index"]))]
        if dbnom:
            print(f"{label:<20s}  {np.mean(dbnom):>8.3f}  {np.std(dbnom):>8.3f}  {len(dbnom):>4d}")


if __name__ == "__main__":
    main()
