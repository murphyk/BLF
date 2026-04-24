"""data_tools.py — Data source lookups for dataset questions (yfinance, fred, dbnomics, wikipedia, acled).

Used by fb_make_data.py to preprocess dataset questions:
- Substitute {resolution_date} and {forecast_due_date} in question text
- Look up the reference value on forecast_due_date
- Split multi-resolution-date questions into separate records
"""

import hashlib
import os
import re

import pandas as pd


# ---------------------------------------------------------------------------
# Disk cache for data fetches
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join("data", "fb_cache", "ts_cache")


def _cache_key(source: str, identifier: str, end_date: str) -> str:
    """Build a filesystem-safe cache key."""
    safe_id = re.sub(r'[/\\:?&=]', '_', identifier)
    return os.path.join(_CACHE_DIR, source, f"{safe_id}_{end_date}.csv")


def _cache_get(source: str, identifier: str, end_date: str) -> pd.DataFrame | None:
    """Read cached DataFrame, or None if not cached."""
    path = _cache_key(source, identifier, end_date)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=["ds"])
            return df if not df.empty else None
        except Exception:
            return None
    return None


def _cache_put(source: str, identifier: str, end_date: str, df: pd.DataFrame):
    """Write DataFrame to cache."""
    if df is None or df.empty:
        return
    path = _cache_key(source, identifier, end_date)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Data fetchers (ported from v4/predict_baseline.py)
# ---------------------------------------------------------------------------

def fetch_yfinance(ticker: str, end_date: str) -> pd.DataFrame | None:
    """Download daily closing prices up to end_date."""
    cached = _cache_get("yfinance", ticker, end_date)
    if cached is not None:
        return cached
    try:
        import yfinance as yf
    except ImportError:
        _warn_once("yfinance not installed — pip install yfinance")
        return None
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    start = (pd.to_datetime(end_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    end = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hist = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if hist is None or hist.empty:
        return None
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist = hist[["Close"]].reset_index()
    hist.columns = ["ds", "y"]
    hist["ds"] = pd.to_datetime(hist["ds"]).dt.tz_localize(None)
    hist["y"] = pd.to_numeric(hist["y"].squeeze(), errors="coerce")
    result = hist.dropna()
    _cache_put("yfinance", ticker, end_date, result)
    return result


_WARNED = set()

def _warn_once(msg):
    if msg not in _WARNED:
        print(f"  WARNING: {msg}")
        _WARNED.add(msg)


def fetch_fred(series_id: str, end_date: str) -> pd.DataFrame | None:
    """Download a FRED series up to end_date (last 1 year)."""
    cached = _cache_get("fred", series_id, end_date)
    if cached is not None:
        return cached
    try:
        import fredapi
    except ImportError:
        _warn_once("fredapi not installed — pip install fredapi")
        return None
    key = os.getenv("FRED_API_KEY")
    if not key:
        _warn_once("FRED_API_KEY not set")
        return None
    fred = fredapi.Fred(api_key=key)
    start = (pd.to_datetime(end_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    try:
        data = fred.get_series(series_id, observation_start=start, observation_end=end_date)
    except Exception as e:
        print(f"  WARNING: FRED fetch failed: {e}")
        return None
    if data is None or data.empty:
        return None
    df = data.reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    result = df.dropna()
    _cache_put("fred", series_id, end_date, result)
    return result


def fetch_dbnomics(question_url: str, end_date: str) -> pd.DataFrame | None:
    """Download a dbnomics series up to end_date."""
    cached = _cache_get("dbnomics", question_url, end_date)
    if cached is not None:
        return cached
    try:
        import dbnomics
    except ImportError:
        _warn_once("dbnomics not installed — pip install dbnomics")
        return None
    try:
        path = question_url.split("db.nomics.world/")[1].rstrip("/")
        if "/" in path:
            provider, dataset, series = path.split("/", 2)
        elif "_" in path:
            # FB uses underscore format: provider_DATASET_series
            # e.g. "meteofrance_TEMPERATURE_celsius.07280.D"
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
        print(f"  WARNING: dbnomics fetch failed: {e}")
        return None
    if df is None or df.empty:
        return None
    period_col = next((c for c in ("period", "original_period") if c in df.columns), None)
    if period_col is None:
        return None
    df = df[[period_col, "value"]].rename(columns={period_col: "ds", "value": "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df = df[df["ds"] <= pd.to_datetime(end_date)]
    # Limit to last 2 years (enough for seasonal patterns)
    start = pd.to_datetime(end_date) - pd.Timedelta(days=730)
    df = df[df["ds"] >= start]
    result = df if not df.empty else None
    _cache_put("dbnomics", question_url, end_date, result)
    return result


def get_value_on_date(df: pd.DataFrame, target_date: str) -> float | None:
    """Get the value on or just before target_date from a time series DataFrame."""
    if df is None or df.empty:
        return None
    target = pd.to_datetime(target_date)
    mask = df["ds"] <= target
    if not mask.any():
        return None
    return float(df.loc[mask, "y"].iloc[-1])


# ---------------------------------------------------------------------------
# Dataset question preprocessing
# ---------------------------------------------------------------------------

DATASET_SOURCES = {"yfinance", "fred", "dbnomics", "acled", "wikipedia"}


def _extract_id_for_source(q: dict) -> str:
    """Extract the data source identifier from the question."""
    source = q.get("source", "")
    qid = str(q.get("id", ""))
    if source == "yfinance":
        return qid  # ticker symbol
    elif source == "fred":
        return qid  # FRED series ID
    elif source == "dbnomics":
        # URL is in resolution_criteria or url field
        url = q.get("url", "") or q.get("resolution_criteria", "")
        m = re.search(r'db\.nomics\.world/\S+', url)
        return f"https://{m.group(0)}" if m else qid
    return qid


def lookup_reference_value(q: dict) -> float | None:
    """Look up the reference value for a dataset question on its forecast_due_date."""
    source = q.get("source", "")
    forecast_date = q.get("forecast_due_date", "")
    if not forecast_date:
        return None

    data_id = _extract_id_for_source(q)

    if source == "yfinance":
        df = fetch_yfinance(data_id, forecast_date)
        return get_value_on_date(df, forecast_date)
    elif source == "fred":
        df = fetch_fred(data_id, forecast_date)
        return get_value_on_date(df, forecast_date)
    elif source == "dbnomics":
        df = fetch_dbnomics(data_id, forecast_date)
        return get_value_on_date(df, forecast_date)
    # acled and wikipedia don't have simple numeric lookups
    return None


def preprocess_dataset_question(q: dict, *, live_lookup: bool = False) -> dict:
    """Preprocess a dataset question into a ready-to-forecast record.

    - Substitutes {forecast_due_date} in question text (keeps {resolution_date}
      as a placeholder since there are multiple resolution dates)
    - Looks up reference value on forecast_due_date (falls back to freeze value)
    - Adds reference value context to background
    - Strips leaky data source URLs from resolution_criteria
    - Keeps resolution_dates and resolved_to as lists

    If live_lookup is False (default), skips the live API call and uses the
    freeze value from the question set directly.

    Returns a single preprocessed record.
    """
    source = q.get("source", "")
    forecast_date = q.get("forecast_due_date", "")
    resolution_dates = q.get("resolution_dates", [])
    resolved_to_list = q.get("resolved_to", [])

    if isinstance(resolution_dates, str):
        resolution_dates = [resolution_dates]
    if not isinstance(resolved_to_list, list):
        resolved_to_list = [resolved_to_list]

    # Look up reference value on forecast_due_date
    ref_value = lookup_reference_value(q) if live_lookup else None
    freeze_date = q.get("freeze_datetime", "")[:10]
    freeze_val = q.get("freeze_datetime_value", "")
    freeze_explanation = q.get("freeze_datetime_value_explanation", "")

    # Build reference value note
    if ref_value is not None:
        ref_str = f"{ref_value:.3f}"
        ref_date = forecast_date
        ref_note = f"The value on {ref_date} was {ref_str}."
    elif freeze_val and freeze_val != "unknown":
        try:
            ref_str = f"{float(freeze_val):.3f}"
        except (ValueError, TypeError):
            ref_str = str(freeze_val)
        ref_date = freeze_date
        ref_note = (f"The most recent known value (as of {ref_date}) was {ref_str}."
                    f" ({freeze_explanation})" if freeze_explanation else
                    f"The most recent known value (as of {ref_date}) was {ref_str}.")
    else:
        ref_str = "unknown"
        ref_date = ""
        ref_note = ""

    # Keep URLs in resolution criteria — the agent needs series identifiers
    # for tool calls. Web search blocking prevents visiting live data pages.
    rc = q.get("resolution_criteria", "")

    # Substitute {forecast_due_date} in question text, keep {resolution_date}
    question_text = q.get("question", "")
    question_text = question_text.replace("{forecast_due_date}", str(forecast_date))
    # Replace {resolution_date} with a description of the dates
    if len(resolution_dates) == 1:
        question_text = question_text.replace("{resolution_date}", str(resolution_dates[0]))
    else:
        question_text = question_text.replace("{resolution_date}", "each resolution date listed below")

    # Build resolution_criteria: original text + reference value
    if ref_value is not None:
        ref_note = f"The value on {forecast_date} was {ref_str}."
    elif freeze_val and freeze_val != "unknown":
        ref_note = (
            f"The exact value on {forecast_date} is unknown, "
            f"but the most recent known value (as of {freeze_date}) was {ref_str}."
            + (f" ({freeze_explanation})" if freeze_explanation else ""))
    else:
        ref_note = ""
    # Keep original RC and append the reference value note
    resolution_criteria = rc
    if ref_note:
        resolution_criteria = f"{rc}\n\n{ref_note}" if rc else ref_note

    # Background: original background text (may be boilerplate)
    # Keep data-source URLs (e.g. db.nomics.world) — the agent needs them
    # to call fetch_ts_dbnomics correctly. The tool enforces date filtering.
    background = q.get("background", "")

    return {
        "id": f"{q['id']}_{forecast_date}" if forecast_date else q["id"],
        "source": source,
        "question": question_text,
        "resolution_criteria": resolution_criteria,
        "background": background,
        "url": q.get("url", ""),
        "forecast_due_date": forecast_date,
        "resolution_dates": resolution_dates,
        "resolved_to": resolved_to_list,
    }
