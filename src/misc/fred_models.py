"""fred_models.py — Per-series statistical models for FRED questions.

Two modes of operation:
1. Standalone bypass (like dbnomics): fred_statistical_forecast()
   — bypasses the LLM entirely, uses pure statistical model
   — Currently WORSE than LLM on FRED (BI 51 vs 57), because the LLM
     gets useful signal from web search even for near-random-walk series.

2. Enhanced tool output: fred_enhanced_analysis()
   — Adds series classification and model-appropriate advice to the
     tool output that the LLM receives, helping it make better decisions.
   — This is the recommended mode.

Usage:
    # Mode 1 (standalone, from predict.py):
    from misc.fred_models import fred_statistical_forecast
    result = fred_statistical_forecast(question, config, verbose=verbose)

    # Mode 2 (enhanced tool, from source_tools.py):
    from misc.fred_models import fred_enhanced_analysis
    analysis = fred_enhanced_analysis(df, threshold, resolution_date, series_id, fdd)
"""
import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import glob
import math
import os
import re

import numpy as np
import pandas as pd

from agent.data_tools import fetch_fred, _CACHE_DIR


# ---------------------------------------------------------------------------
# Series classification
# ---------------------------------------------------------------------------

def _load_fred_cached_any(series_id: str, end_date: str) -> pd.DataFrame:
    """Load FRED data from cache, trying exact match then any cached file.

    Merges all cached files for this series and filters to <= end_date.
    """
    # Try exact match first
    df = fetch_fred(series_id, end_date)
    if df is not None and not df.empty:
        return df

    # Fall back: merge all cached files for this series
    cache_dir = os.path.join(_CACHE_DIR, "fred")
    if not os.path.isdir(cache_dir):
        return pd.DataFrame()

    files = sorted(glob.glob(os.path.join(cache_dir, f"{series_id}_*.csv")))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, parse_dates=['ds'])
            dfs.append(d)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs).drop_duplicates(subset='ds').sort_values('ds')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds', 'y'])
    # Filter to before end_date
    df = df[df['ds'] <= pd.to_datetime(end_date)]
    return df


def _compute_autocorr(series_id: str, end_date: str) -> float:
    """Compute lag-1 autocorrelation of daily returns for a FRED series."""
    df = _load_fred_cached_any(series_id, end_date)
    if df is None or len(df) < 30:
        return 0.0
    returns = df['y'].diff().dropna()
    if returns.std() == 0:
        return 0.0
    return float(returns.autocorr(lag=1))


def classify_series(series_id: str, end_date: str) -> str:
    """Classify a FRED series as 'random-walk', 'trending', or 'mild'.

    Based on lag-1 autocorrelation of daily returns:
    - |AC| < 0.05: random walk (changes are uncorrelated)
    - |AC| > 0.15: trending (changes have momentum)
    - otherwise: mild (intermediate)
    """
    ac = _compute_autocorr(series_id, end_date)
    if abs(ac) < 0.05:
        return "random-walk"
    elif abs(ac) > 0.15:
        return "trending"
    else:
        return "mild"


# ---------------------------------------------------------------------------
# Forecasting models
# ---------------------------------------------------------------------------

def _random_walk_forecast(comparison_value: float, last_value: float,
                          std: float, resolution_date: str,
                          forecast_date: str) -> float:
    """Random walk: future value is current value ± noise.

    P(value > threshold) ≈ P(N(last_value, std*sqrt(horizon)) > threshold)
    But for a true random walk, the best estimate is just based on
    current position relative to threshold, shrunk toward 0.5.
    """
    from scipy.stats import norm

    # Horizon in days
    horizon = (pd.to_datetime(resolution_date) - pd.to_datetime(forecast_date)).days
    if horizon <= 0:
        return 0.5

    # Scale std by sqrt(horizon) for random walk
    rw_std = std * math.sqrt(horizon)
    if rw_std <= 0:
        return 0.5

    # P(value > threshold)
    p = float(norm.sf(comparison_value, loc=last_value, scale=rw_std))

    # Heavy shrinkage toward 0.5 (random walk is hard to predict)
    p = 0.2 * p + 0.8 * 0.5
    return max(0.05, min(0.95, p))


def _trending_forecast(df: pd.DataFrame, comparison_value: float,
                       resolution_date: str) -> float:
    """Trending series: use recent momentum + linear extrapolation.

    More weight on the linear model than for random walks.
    """
    from scipy.stats import norm

    if len(df) < 5:
        return 0.5

    # Use last 30 data points for trend
    recent = df.iloc[-min(30, len(df)):]
    t0 = recent['ds'].min()
    x = (recent['ds'] - t0).dt.days.values.astype(float)
    y = recent['y'].values.astype(float)

    try:
        coeffs = np.polyfit(x, y, 1)
    except Exception:
        return 0.5

    residuals = y - np.polyval(coeffs, x)
    std = float(residuals.std())
    if std <= 0:
        std = max(abs(float(y.mean())) * 0.05, 1e-6)

    # Predict at resolution date
    x_rd = float((pd.to_datetime(resolution_date) - t0).days)
    predicted = float(np.polyval(coeffs, x_rd))

    p = float(norm.sf(comparison_value, loc=predicted, scale=std))

    # Moderate shrinkage (trust the trend more than random walk)
    p = 0.6 * p + 0.4 * 0.5
    return max(0.05, min(0.95, p))


def _mild_forecast(df: pd.DataFrame, comparison_value: float,
                   resolution_date: str) -> float:
    """Intermediate: blend of trending and random-walk approaches."""
    p_trend = _trending_forecast(df, comparison_value, resolution_date)
    # For mild series, blend trend with heavy shrinkage
    return 0.5 * p_trend + 0.5 * 0.5


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
# Enhanced tool output (Mode 2)
# ---------------------------------------------------------------------------

def fred_enhanced_analysis(df: pd.DataFrame, threshold: float,
                           resolution_date: str, series_id: str,
                           forecast_date: str) -> str:
    """Generate enhanced FRED analysis with series classification.

    Appended to the standard _analyze_trend output to give the LLM
    better guidance based on the series type.
    """
    if df is None or df.empty:
        return ""

    stype = classify_series(series_id, forecast_date)

    # Compute recent stats
    returns = df['y'].diff().dropna()
    daily_std = float(returns.std()) if len(returns) > 5 else 0
    ac1 = float(returns.autocorr(lag=1)) if len(returns) > 10 else 0

    # Compute model probability
    if stype == "random-walk":
        p = _random_walk_forecast(threshold, float(df['y'].iloc[-1]),
                                  daily_std, resolution_date, forecast_date)
        advice = ("This series behaves like a RANDOM WALK (autocorrelation = "
                  f"{ac1:.2f}). Short-term predictions are unreliable. "
                  f"Statistical estimate: P(increase) = {p:.2f}. "
                  "Consider predicting close to 0.5 unless you have strong "
                  "evidence from recent news or policy changes.")
    elif stype == "trending":
        p = _trending_forecast(df, threshold, resolution_date)
        # Determine direction
        recent = df.iloc[-10:]
        slope = np.polyfit(range(len(recent)), recent['y'].values, 1)[0]
        direction = "upward" if slope > 0 else "downward"
        advice = (f"This series shows a TRENDING pattern (autocorrelation = "
                  f"{ac1:.2f}, recent trend: {direction}). "
                  f"Statistical estimate: P(increase) = {p:.2f}. "
                  "The trend signal is relatively reliable for this series.")
    else:
        p = _mild_forecast(df, threshold, resolution_date)
        advice = (f"This series has MILD autocorrelation ({ac1:.2f}). "
                  f"Statistical estimate: P(increase) = {p:.2f}. "
                  "Use both the trend and any relevant news.")

    lines = [
        "",
        "=== Series Classification (FRED) ===",
        f"Series: {series_id}",
        f"Type: {stype} (lag-1 autocorrelation = {ac1:.3f})",
        f"Daily return std: {daily_std:.6f}",
        f"Statistical P(increase): {p:.2f}",
        f"Advice: {advice}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone bypass (Mode 1)
# ---------------------------------------------------------------------------

def fred_statistical_forecast(question: dict, config, verbose: bool = False) -> dict:
    """Compute FRED forecast using per-series statistical model (no LLM).

    Similar to _dbnomics_harmonic_forecast: bypasses the agent loop entirely.
    """
    import time

    t0 = time.time()
    qid = question.get("id", "unknown")
    source = question.get("source", "fred")
    fdd = question.get("forecast_due_date", "")
    res_dates = question.get("resolution_dates", [])

    # Extract series ID
    series_id = qid.split("_")[0] if "_" in qid else qid

    # Determine cutoff
    if config.clairvoyant:
        from datetime import date
        cutoff = str(date.today())
    else:
        cutoff = fdd[:10] if fdd else ""

    # Fetch data (try API, fall back to cache)
    df = _load_fred_cached_any(series_id, cutoff)

    if df is None or df.empty:
        if verbose:
            print(f"  [fred/{qid}] no data, returning 0.5")
        forecasts = [0.5] * len(res_dates)
        return _build_result(question, forecasts, res_dates, t0,
                             series_type="unknown", series_id=series_id)

    # Get threshold (value on forecast date)
    fdd_dt = pd.to_datetime(fdd)
    df_before = df[df['ds'] <= fdd_dt]
    if df_before.empty:
        threshold = float(df['y'].iloc[-1])
    else:
        threshold = float(df_before['y'].iloc[-1])

    # Classify series
    series_type = classify_series(series_id, cutoff)

    # Compute daily return std for random walk model
    returns = df['y'].diff().dropna()
    daily_std = float(returns.std()) if len(returns) > 5 else float(df['y'].std() * 0.1)

    # Forecast each resolution date
    forecasts = []
    for rd in res_dates:
        if series_type == "random-walk":
            p = _random_walk_forecast(threshold, threshold, daily_std, rd, fdd)
        elif series_type == "trending":
            p = _trending_forecast(df, threshold, rd)
        else:  # mild
            p = _mild_forecast(df, threshold, rd)
        forecasts.append(max(0.05, min(0.95, p)))

    if verbose:
        pred_str = ", ".join(f"{rd}: p={p:.2f}" for rd, p in zip(res_dates[:3], forecasts[:3]))
        print(f"  [fred/{qid}] type={series_type}, threshold={threshold:.4g}, "
              f"forecasts: {pred_str}")

    return _build_result(question, forecasts, res_dates, t0,
                         series_type=series_type, series_id=series_id)


def _build_result(question, forecasts, res_dates, t0, **meta):
    """Build a result dict compatible with predict.py output format."""
    import time

    final_p = forecasts[0] if forecasts else 0.5
    resolved_to = question.get("resolved_to", [])
    if not isinstance(resolved_to, list):
        resolved_to = [resolved_to]

    return {
        "id": question.get("id"),
        "source": question.get("source", "fred"),
        "question": question.get("question", ""),
        "background": question.get("background", ""),
        "resolution_criteria": question.get("resolution_criteria", ""),
        "forecast_due_date": question.get("forecast_due_date", ""),
        "url": question.get("url", ""),
        "forecast": final_p,
        "reasoning": f"Statistical FRED model ({meta.get('series_type', 'unknown')})",
        "resolution_date": res_dates[0] if res_dates else "",
        "resolution_dates": res_dates,
        "resolved_to": resolved_to[0] if len(resolved_to) == 1 else resolved_to,
        "system_prompt": "",
        "question_prompt": "",
        "belief_history": [{"p": final_p, "step": 0}],
        "tool_log": [{"step": 0, "type": "statistical_model", **meta}],
        "n_steps": 0,
        "submitted": True,
        "tokens_in": 0,
        "tokens_out": 0,
        "elapsed_seconds": time.time() - t0,
        "config": {"name": "fred-statistical"},
        "trial_stats": {
            "n_trials": 1,
            "aggregation": "statistical",
            "forecasts": {"trial_1": final_p},
        },
        # Per-resolution-date forecasts
        "forecast_raw": final_p,
        "forecasts_per_rdate": {rd: p for rd, p in zip(res_dates, forecasts)},
    }
