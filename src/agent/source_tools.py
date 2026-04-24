"""source_tools.py — Source-specific LLM-callable tools for data and market sources.

Ported from forecast-bench-v4/tools.py. Provides tool schemas, dispatch, and
implementations for yfinance, fred, dbnomics, wikipedia, polymarket, manifold.

Each tool takes an end_date parameter that is clamped to the forecast cutoff
date to prevent data leakage.
"""

import os
import re
import urllib.parse

import numpy as np
import pandas as pd
import requests

from agent.data_tools import fetch_yfinance, fetch_fred, fetch_dbnomics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_ROWS_DISPLAY = 30

DATA_TOOL_SOURCES = {"yfinance", "fred", "dbnomics", "wikipedia"}
MARKET_TOOL_SOURCES = {"polymarket", "manifold"}
NUMERIC_TS_SOURCES = {"yfinance", "fred", "dbnomics"}
ALL_TOOL_SOURCES = DATA_TOOL_SOURCES | MARKET_TOOL_SOURCES

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _df_to_csv(df, end_date):
    """Return up to _MAX_ROWS_DISPLAY rows as CSV, subsampled if needed."""
    if df is None or df.empty:
        return "No data available."
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df[df["ds"] <= pd.to_datetime(end_date)]
    if df.empty:
        return "No data available up to the specified date."
    if len(df) > _MAX_ROWS_DISPLAY:
        n_recent = _MAX_ROWS_DISPLAY // 2
        n_hist = _MAX_ROWS_DISPLAY - n_recent
        recent = df.iloc[-n_recent:]
        older = df.iloc[:-n_recent]
        if len(older) > n_hist:
            idx = np.round(np.linspace(0, len(older) - 1, n_hist)).astype(int)
            older = older.iloc[np.unique(idx)]
        df = pd.concat([older, recent]).drop_duplicates(subset="ds")
    df = df.copy()
    df["ds"] = df["ds"].dt.strftime("%Y-%m-%d")
    lines = ["date,value"] + [f"{row['ds']},{row['y']}" for _, row in df.iterrows()]
    return "\n".join(lines)


def _clamp_end_date(end_date, cutoff_date):
    """Clamp end_date to not exceed cutoff_date."""
    if cutoff_date and end_date:
        return min(end_date, cutoff_date)
    return cutoff_date or end_date


# ---------------------------------------------------------------------------
# Statistical trend analysis for numeric time-series sources
# ---------------------------------------------------------------------------

def _harmonic_forecast(df: pd.DataFrame, comparison_value: float,
                       resolution_dates: list[str]) -> list[tuple[str, float, float]]:
    """Predict P(value > threshold) per resolution date using empirical exceedance.

    Uses historical data from the same time of year (±10 days) across all
    available years, with Laplace smoothing. This outperforms parametric
    models (harmonic, linear) on seasonal temperature data because it
    directly estimates the exceedance probability without distributional
    assumptions.

    Returns list of (resolution_date, n_historical, p_exceed) tuples.
    """
    if df is None or df.empty or len(df) < 30:
        return [(rd, 0, 0.5) for rd in resolution_dates]

    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").dropna(subset=["y"])

    _WINDOW_DAYS = 10  # ±10 days around same day-of-year

    results = []
    for rd in resolution_dates:
        rd_doy = pd.to_datetime(rd).dayofyear
        doys = df["ds"].dt.dayofyear.values

        # Same day-of-year ± window across all years
        mask = np.abs(doys - rd_doy) <= _WINDOW_DAYS
        # Handle year wrap-around (e.g. day 5 vs day 360)
        mask |= np.abs(doys - rd_doy + 365) <= _WINDOW_DAYS
        mask |= np.abs(doys - rd_doy - 365) <= _WINDOW_DAYS

        historical = df[mask]["y"].values
        if len(historical) < 5:
            p_exceed = 0.5
        else:
            n_exceed = int((historical > comparison_value).sum())
            # Laplace smoothing
            p_exceed = (n_exceed + 1) / (len(historical) + 2)
            p_exceed = max(0.05, min(0.95, p_exceed))

        results.append((rd, len(historical), p_exceed))

    return results


def _analyze_trend(df: pd.DataFrame, comparison_value: float,
                   resolution_date: str, source: str) -> str:
    """Analyze a time series and return P(value > threshold).

    For seasonal sources (dbnomics), uses a harmonic model (linear + annual
    + semi-annual sinusoidal) which captures seasonal patterns.
    For other sources, uses linear trend + same-period-in-prior-years.

    Returns a human-readable analysis string.
    """
    from scipy.stats import norm as scipy_norm

    if df is None or df.empty:
        return "No data available for trend analysis."

    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").dropna(subset=["y"])

    if len(df) < 3:
        return "Insufficient data points for trend analysis (need >= 3)."

    # Source-dependent window: use recent data to avoid overwhelming with history
    # dbnomics (temperature etc.) can be very long with strong seasonality
    if source == "yfinance":
        # Stock prices: ~60 trading days (~3 months)
        window = min(len(df), 60)
    elif source == "fred":
        # Economic indicators: ~30 observations (varies by frequency)
        window = min(len(df), 30)
    elif source == "dbnomics":
        # Use ~365 days for seasonal data to capture annual cycles
        cutoff_dt = df["ds"].max() - pd.Timedelta(days=365)
        seasonal_df = df[df["ds"] >= cutoff_dt]
        window = max(len(seasonal_df), min(len(df), 30))
    else:
        window = min(len(df), 30)

    recent = df.iloc[-window:]

    # Linear fit
    t0 = recent["ds"].min()
    x = (recent["ds"] - t0).dt.days.values.astype(float)
    y = recent["y"].values.astype(float)

    coeffs = np.polyfit(x, y, 1)
    residuals = y - np.polyval(coeffs, x)
    std = float(residuals.std())
    if std <= 0:
        std = max(abs(float(y.mean())) * 0.05, 1e-6)

    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # Predict at resolution date
    x_rd = float((pd.to_datetime(resolution_date) - t0).days)
    predicted = float(np.polyval(coeffs, x_rd))

    # P(value > comparison_value) via normal CDF
    prob_raw = float(scipy_norm.sf(comparison_value, loc=predicted, scale=std))

    # Source-specific shrinkage toward 0.5: stock weekly returns are nearly
    # unpredictable, so dampen the signal. FRED/dbnomics have more structure.
    # How much to trust the linear model: alpha * model + (1-alpha) * 0.5
    # These are rough defaults — should be LOO-tuned per source.
    _SHRINK_TOWARD_HALF = {
        "yfinance": 0.1,   # very heavy — weekly stock moves are ~random walk
        "fred": 0.5,       # moderate — economic indicators have trends
        "dbnomics": 0.4,   # moderate — has seasonal patterns but linear model is crude
    }
    alpha = _SHRINK_TOWARD_HALF.get(source, 0.7)
    prob_linear = alpha * prob_raw + (1 - alpha) * 0.5
    prob_linear = max(0.05, min(0.95, prob_linear))

    # Recent momentum (last 5 points)
    last_n = min(5, len(recent))
    recent_vals = recent["y"].iloc[-last_n:].values
    if len(recent_vals) >= 2:
        recent_changes = np.diff(recent_vals)
        n_up = int((recent_changes > 0).sum())
        n_down = int((recent_changes < 0).sum())
        momentum = "upward" if n_up > n_down else "downward" if n_down > n_up else "flat"
    else:
        momentum = "unknown"
        n_up = n_down = 0

    last_value = float(recent["y"].iloc[-1])
    last_date = recent["ds"].iloc[-1].strftime("%Y-%m-%d")

    # Same-period-in-prior-years analysis (crucial for seasonal data)
    rd = pd.to_datetime(resolution_date)
    rd_month = rd.month
    rd_day = rd.day
    same_period_vals = []
    same_period_lines = []
    for year_offset in range(1, 6):  # look back up to 5 years
        target_date = rd.replace(year=rd.year - year_offset)
        # Find closest data point within ±7 days
        window_mask = (df["ds"] >= target_date - pd.Timedelta(days=7)) & \
                      (df["ds"] <= target_date + pd.Timedelta(days=7))
        window_data = df[window_mask]
        if not window_data.empty:
            # Pick the closest date
            closest_idx = (window_data["ds"] - target_date).abs().idxmin()
            val = float(window_data.loc[closest_idx, "y"])
            date_str = window_data.loc[closest_idx, "ds"].strftime("%Y-%m-%d")
            same_period_vals.append(val)
            exceeded = "YES" if val > comparison_value else "NO"
            same_period_lines.append(
                f"  {date_str}: {val:.4g} (exceeded threshold: {exceeded})")

    # Seasonal probability estimate
    prob_seasonal = None
    seasonal_str = ""
    if same_period_vals:
        n_exceeded = sum(1 for v in same_period_vals if v > comparison_value)
        n_total = len(same_period_vals)
        # Laplace smoothing
        prob_seasonal = (n_exceeded + 1) / (n_total + 2)
        prob_seasonal = max(0.05, min(0.95, prob_seasonal))
        seasonal_str = (f"Seasonal estimate: {n_exceeded}/{n_total} prior years "
                        f"exceeded threshold → P(exceed) = {prob_seasonal:.2f}")

    # Combined estimate: prefer seasonal for seasonal data, linear for trend data
    if prob_seasonal is not None and source in ("dbnomics",):
        # For strongly seasonal data, weight seasonal estimate heavily
        prob_increase = 0.7 * prob_seasonal + 0.3 * prob_linear
    elif prob_seasonal is not None:
        prob_increase = 0.5 * prob_seasonal + 0.5 * prob_linear
    else:
        prob_increase = prob_linear
    prob_increase = max(0.05, min(0.95, prob_increase))

    lines = [
        f"=== Trend Analysis ===",
        f"Data points used: {len(recent)} (of {len(df)} total)",
        f"Period: {recent['ds'].iloc[0].strftime('%Y-%m-%d')} to {last_date}",
        f"Last value: {last_value:.4g} (on {last_date})",
        f"Comparison value (threshold): {comparison_value:.4g}",
        f"",
        f"Linear trend: slope = {slope:.4g} per day",
        f"Predicted value at {resolution_date}: {predicted:.4g}",
        f"Residual std: {std:.4g}",
        f"Linear model estimate: P(value > {comparison_value:.4g}) = {prob_linear:.2f}",
        f"",
        f"Recent momentum (last {last_n} points): {momentum} "
        f"({n_up} up, {n_down} down)",
    ]
    if same_period_lines:
        lines.extend([
            f"",
            f"=== Same Period in Prior Years ===",
            f"Values near {rd.strftime('%b %d')} in prior years:",
        ] + same_period_lines + [
            f"",
            f"{seasonal_str}",
        ])
    lines.extend([
        f"",
        f">>> Combined estimate: P(value > {comparison_value:.4g}) = {prob_increase:.2f} <<<",
        f"",
        f"Use this as a strong prior for your probability estimate. "
        f"The seasonal analysis (same time of year in prior years) is "
        f"especially informative for data with annual cycles.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_MAX_WIKI_CHARS = 20000
_WIKI_HEADERS = {
    "User-Agent": "ForecastBench/1.0 (https://github.com/forecastingresearch/forecastbench; murphyk@cs.ubc.ca)"
}


def _wiki_get_revision(url, end_date):
    """Find latest Wikipedia revision at or before end_date."""
    title = url
    m = re.search(r"/wiki/([^#?]+)", url)
    if m:
        title = urllib.parse.unquote(m.group(1)).replace("_", " ")
    params_rev = {
        "action": "query", "titles": title, "prop": "revisions",
        "rvprop": "ids|timestamp", "rvstart": f"{end_date}T23:59:59Z",
        "rvlimit": 1, "rvdir": "older", "format": "json",
    }
    resp = requests.get(_WIKI_API, params=params_rev, headers=_WIKI_HEADERS, timeout=30)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    revisions = page.get("revisions", [])
    if not revisions:
        return title, None, None, ""
    revid = revisions[0]["revid"]
    rev_ts = revisions[0]["timestamp"]
    params_parse = {
        "action": "parse", "oldid": revid, "prop": "wikitext", "format": "json",
    }
    resp2 = requests.get(_WIKI_API, params=params_parse, headers=_WIKI_HEADERS, timeout=60)
    resp2.raise_for_status()
    wikitext = resp2.json().get("parse", {}).get("wikitext", {}).get("*", "")
    return title, revid, rev_ts, wikitext


def _wiki_strip_markup(wikitext):
    text = re.sub(r"\{\{[^{}]*\}\}", "", wikitext)
    text = re.sub(r"\[\[(File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[=\*#\|]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _wiki_extract_toc(wikitext):
    headings = re.findall(r"^(={2,})\s*(.+?)\s*\1\s*$", wikitext, re.MULTILINE)
    if not headings:
        return ""
    lines = []
    for eq, heading in headings:
        indent = "  " * (len(eq) - 2)
        lines.append(f"  {indent}{heading}")
    return "\n".join(lines)


def _fetch_wikipedia_toc(url, end_date):
    title, revid, rev_ts, wikitext = _wiki_get_revision(url, end_date)
    if revid is None:
        return f"No Wikipedia revision found for '{title}' at or before {end_date}."
    header = (
        f"Wikipedia: '{title}'\n"
        f"Revision: {revid} ({rev_ts[:10]})\n"
        f"(Content as of {rev_ts[:10]}, requested cutoff {end_date})\n\n"
    )
    toc = _wiki_extract_toc(wikitext)
    if not toc:
        text = _wiki_strip_markup(wikitext)
        body = text[:_MAX_WIKI_CHARS]
        if len(text) > _MAX_WIKI_CHARS:
            body += f"\n\n[... truncated at {_MAX_WIKI_CHARS} chars ...]"
        return header + body
    return (header
            + "Sections (use fetch_wikipedia_section with the section name to get content):\n"
            + toc)


def _fetch_wikipedia_section(url, end_date, section):
    title, revid, rev_ts, wikitext = _wiki_get_revision(url, end_date)
    if revid is None:
        return f"No Wikipedia revision found for '{title}' at or before {end_date}."
    header = (
        f"Wikipedia: '{title}'\n"
        f"Revision: {revid} ({rev_ts[:10]})\n"
        f"(Content as of {rev_ts[:10]}, requested cutoff {end_date})\n\n"
    )
    if section.lower() == "introduction":
        first_heading = re.search(r"^==[^=]", wikitext, re.MULTILINE)
        intro_text = wikitext[:first_heading.start()] if first_heading else wikitext
        text = _wiki_strip_markup(intro_text)
        if not text.strip():
            return header + "(No introduction text found before first section heading.)"
        body = text[:_MAX_WIKI_CHARS]
        if len(text) > _MAX_WIKI_CHARS:
            body += f"\n\n[... truncated at {_MAX_WIKI_CHARS} chars ...]"
        return header + body
    pattern = re.compile(r"(={2,})\s*" + re.escape(section) + r"\s*\1", re.IGNORECASE)
    m2 = pattern.search(wikitext)
    if m2:
        start = m2.start()
        level = len(m2.group(1))
        next_heading = re.compile(r"={" + str(level) + r",}[^=]", re.MULTILINE)
        m3 = next_heading.search(wikitext, m2.end())
        wikitext = wikitext[start: m3.start() if m3 else len(wikitext)]
    else:
        toc = _wiki_extract_toc(wikitext)
        return header + f"Section '{section}' not found.\n\nAvailable sections:\n" + toc
    text = _wiki_strip_markup(wikitext)
    body = text[:_MAX_WIKI_CHARS]
    if len(text) > _MAX_WIKI_CHARS:
        body += f"\n\n[... truncated at {_MAX_WIKI_CHARS} chars ...]"
    return header + body


# ---------------------------------------------------------------------------
# Polymarket helpers
# ---------------------------------------------------------------------------

_GAMMA_API = "https://gamma-api.polymarket.com/markets"
_CLOB_PRICES = "https://clob.polymarket.com/prices-history"
_POLYMARKET_CHUNK_DAYS = 14


def _polymarket_lookup(slug=""):
    params = {"slug": slug}
    resp = requests.get(_GAMMA_API, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    m = data[0]
    import json as _json
    tokens = _json.loads(m.get("clobTokenIds", "[]"))
    outcomes = _json.loads(m.get("outcomes", "[]"))
    return {
        "question": m.get("question", ""),
        "yes_token": tokens[0] if tokens else "",
        "volume": m.get("volumeNum") or m.get("volume") or 0,
        "closed": m.get("closed", False),
    }


def _polymarket_price_history(token_id, end_date, days=60):
    from datetime import datetime, timedelta
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    all_points = {}
    remaining = days
    chunk_end = end_dt
    while remaining > 0:
        chunk_days = min(remaining, _POLYMARKET_CHUNK_DAYS)
        chunk_start = chunk_end - timedelta(days=chunk_days)
        resp = requests.get(_CLOB_PRICES, params={
            "market": token_id,
            "startTs": int(chunk_start.timestamp()),
            "endTs": int(chunk_end.timestamp()),
            "fidelity": chunk_days * 60,
        }, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for pt in data.get("history", []):
                dt = datetime.fromtimestamp(pt["t"])
                date_str = dt.strftime("%Y-%m-%d")
                if date_str not in all_points:
                    all_points[date_str] = pt["p"]
        chunk_end = chunk_start
        remaining -= chunk_days
    return sorted(all_points.items())


def _fetch_polymarket_info(url, end_date, days=60):
    slug = ""
    if "/market/" in url:
        slug = url.split("/market/")[-1].split("?")[0].split("/")[0]
    if not slug:
        return f"Cannot extract market slug from URL: {url}"
    market = _polymarket_lookup(slug=slug)
    if not market:
        return f"Market not found for slug: {slug}"
    lines = [
        f"Polymarket: {market['question']}",
        f"Total volume: ${market['volume']:,.0f}",
        f"Status: {'closed' if market['closed'] else 'active'}",
    ]
    if market["yes_token"]:
        history = _polymarket_price_history(market["yes_token"], end_date, days=days)
        if history:
            prices = [p for _, p in history]
            lines.append(f"\nPrice history ({len(history)} days, up to {end_date}):")
            if len(prices) >= 2:
                daily_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
                lines.append(f"  Current price: {prices[-1]:.3f}")
                lines.append(f"  Min: {min(prices):.3f}, Max: {max(prices):.3f}")
                lines.append(f"  Mean daily change: {np.mean(daily_changes):.4f}")
                if len(prices) >= 7:
                    recent = prices[-7:]
                    trend = recent[-1] - recent[0]
                    lines.append(f"  7-day trend: {'+' if trend >= 0 else ''}{trend:.3f}")
            lines.append("\ndate,price")
            for date_str, price in history:
                lines.append(f"{date_str},{price:.4f}")
        else:
            lines.append(f"\nNo price history available up to {end_date}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manifold helpers
# ---------------------------------------------------------------------------

_MANIFOLD_API = "https://api.manifold.markets/v0"


def _manifold_market_info(market_id):
    resp = requests.get(f"{_MANIFOLD_API}/market/{market_id}", timeout=15)
    resp.raise_for_status()
    m = resp.json()
    return {
        "question": m.get("question", ""),
        "volume": m.get("volume", 0),
        "unique_bettors": m.get("uniqueBettorCount", 0),
        "total_liquidity": m.get("totalLiquidity", 0),
        "probability": m.get("probability"),
        "slug": m.get("slug", ""),
    }


def _manifold_price_history(market_id, end_date, days=60):
    from datetime import datetime, timedelta
    from collections import OrderedDict
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=days)
    before_ms = int((end_dt + timedelta(days=1)).timestamp() * 1000)
    after_ms = int(start_dt.timestamp() * 1000)
    all_bets = []
    cursor_ms = before_ms
    for _ in range(10):
        params = {
            "contractId": market_id,
            "beforeTime": cursor_ms,
            "limit": 1000,
            "order": "desc",
        }
        resp = requests.get(f"{_MANIFOLD_API}/bets", params=params, timeout=15)
        resp.raise_for_status()
        bets = resp.json()
        if not bets:
            break
        for b in bets:
            if b["createdTime"] >= after_ms:
                all_bets.append(b)
        if bets[-1]["createdTime"] < after_ms:
            break
        cursor_ms = bets[-1]["createdTime"]
        if len(bets) < 1000:
            break
    if not all_bets:
        return []
    daily = OrderedDict()
    for b in sorted(all_bets, key=lambda x: x["createdTime"]):
        if "probAfter" not in b:
            continue
        dt = datetime.fromtimestamp(b["createdTime"] / 1000)
        date_str = dt.strftime("%Y-%m-%d")
        if date_str <= end_date:
            daily[date_str] = b["probAfter"]
    return list(daily.items())


def _fetch_manifold_info(market_id, end_date, days=60):
    market = _manifold_market_info(market_id)
    if not market:
        return f"Market not found: {market_id}"
    lines = [
        f"Manifold: {market['question']}",
        f"Total volume: M${market['volume']:,.0f} (play money)",
        f"Unique bettors: {market['unique_bettors']}",
        f"Liquidity pool: M${market['total_liquidity']:,.0f}",
        f"Status: active",
    ]
    history = _manifold_price_history(market_id, end_date, days=days)
    if history:
        prices = [p for _, p in history]
        lines.append(f"\nProbability history ({len(history)} days with activity, up to {end_date}):")
        if len(prices) >= 2:
            daily_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            lines.append(f"  Current probability: {prices[-1]:.4f}")
            lines.append(f"  Min: {min(prices):.4f}, Max: {max(prices):.4f}")
            lines.append(f"  Mean daily change: {np.mean(daily_changes):.4f}")
            if len(prices) >= 7:
                recent = prices[-7:]
                trend = recent[-1] - recent[0]
                lines.append(f"  7-day trend: {'+' if trend >= 0 else ''}{trend:.4f}")
        lines.append("\ndate,probability")
        for date_str, prob in history:
            lines.append(f"{date_str},{prob:.4f}")
    else:
        lines.append(f"\nNo betting activity found up to {end_date}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format, without updated_belief)
# ---------------------------------------------------------------------------

_YFINANCE_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_ts_yfinance",
        "description": (
            "Fetch historical daily closing prices for a Yahoo Finance ticker. "
            f"Returns a CSV of up to {_MAX_ROWS_DISPLAY} (date, price) rows "
            "(15 most recent daily + 15 sampled from the prior year), "
            "plus a statistical trend analysis with same-period-in-prior-years "
            "comparison and P(exceed threshold) estimate. "
            "Use the statistical estimate as your primary anchor."
            "Stock prices are autocorrelated but noisy — look for trends and recent momentum."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Yahoo Finance ticker symbol (e.g. AAPL)"},
                "end_date": {"type": "string", "description": "Set to the forecast due date (YYYY-MM-DD)"},
            },
            "required": ["ticker", "end_date"],
        },
    },
}

_FRED_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_ts_fred",
        "description": (
            "Fetch a FRED (Federal Reserve Economic Data) series. "
            f"Returns a CSV of up to {_MAX_ROWS_DISPLAY} (date, value) rows "
            "(15 most recent + 15 sampled from older history), "
            "plus a statistical trend analysis with same-period-in-prior-years "
            "comparison and P(exceed threshold) estimate. "
            "Use the statistical estimate as your primary anchor."
            "Identify the data frequency and any trends before forecasting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "series_id": {"type": "string", "description": "FRED series ID (e.g. DGS5, CPIAUCSL)"},
                "end_date": {"type": "string", "description": "Set to the forecast due date (YYYY-MM-DD)"},
            },
            "required": ["series_id", "end_date"],
        },
    },
}

_DBNOMICS_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_ts_dbnomics",
        "description": (
            "Fetch a DBnomics time series (last 2 years). "
            f"Returns a CSV of up to {_MAX_ROWS_DISPLAY} (date, value) rows "
            "(15 most recent daily + 15 sampled from older history), "
            "plus a statistical trend analysis with same-period-in-prior-years "
            "comparison and combined P(exceed threshold) estimate. "
            "Use the statistical estimate as your primary anchor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "DBnomics series URL (https://db.nomics.world/provider/dataset/series)"},
                "end_date": {"type": "string", "description": "Set to the forecast due date (YYYY-MM-DD)"},
            },
            "required": ["url", "end_date"],
        },
    },
}

_ANALYZE_TREND_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_trend",
        "description": (
            "Statistically analyze a time series to estimate the probability that "
            "a value will increase by a target date. Fits a linear trend model and "
            "uses the residual variance to compute P(value > comparison_value). "
            "Call this AFTER fetching the time series data (fetch_ts_yfinance, "
            "fetch_ts_fred, or fetch_ts_dbnomics). This gives you a data-driven "
            "probability estimate that you should use as your primary anchor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["yfinance", "fred", "dbnomics"],
                    "description": "Data source (determines analysis window)",
                },
                "comparison_value": {
                    "type": "number",
                    "description": "The threshold value to compare against "
                                   "(typically the value on the forecast_due_date)",
                },
                "resolution_date": {
                    "type": "string",
                    "description": "Target date for the forecast, YYYY-MM-DD",
                },
                "end_date": {
                    "type": "string",
                    "description": "Latest date of available data, YYYY-MM-DD",
                },
            },
            "required": ["source", "comparison_value", "resolution_date", "end_date"],
        },
    },
}

_WIKIPEDIA_TOC_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_wikipedia_toc",
        "description": (
            "Fetch the table of contents of a Wikipedia page as it existed at or before a cutoff date. "
            "Use this first to identify the relevant section, then call fetch_wikipedia_section."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full Wikipedia URL"},
                "end_date": {"type": "string", "description": "Set to the forecast due date (YYYY-MM-DD)"},
            },
            "required": ["url", "end_date"],
        },
    },
}

_WIKIPEDIA_SECTION_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_wikipedia_section",
        "description": (
            "Fetch the text of a specific section of a Wikipedia page as it existed at or before "
            "a cutoff date. Use fetch_wikipedia_toc first to find the right heading."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full Wikipedia URL"},
                "end_date": {"type": "string", "description": "Set to the forecast due date (YYYY-MM-DD)"},
                "section": {"type": "string", "description": "Section heading to fetch"},
            },
            "required": ["url", "end_date", "section"],
        },
    },
}

_POLYMARKET_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_polymarket_info",
        "description": (
            "Fetch Polymarket prediction market data: question, volume, and daily price history "
            "up to the specified cutoff date (which should be set to the forecast due date). "
            "The price is the market's implied probability of YES. "
            "High volume and low volatility suggest a reliable crowd estimate; "
            "low volume or high volatility suggest uncertainty where your own research may add value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Polymarket market URL"},
                "end_date": {"type": "string", "description": "Set to the forecast due date (YYYY-MM-DD)"},
                "days": {"type": "integer", "description": "Days of history (default: 60, max: 90)"},
            },
            "required": ["url", "end_date"],
        },
    },
}

_MANIFOLD_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_manifold_info",
        "description": (
            "Fetch Manifold Markets data: question, volume, bettors, and daily probability history "
            "up to the specified cutoff date (which should be set to the forecast due date). "
            "Probability is reconstructed from bet records. "
            "High volume/many bettors and low volatility suggest a reliable crowd estimate; "
            "low activity or high volatility suggest uncertainty where your own research may add value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "market_id": {"type": "string", "description": "Manifold market ID (alphanumeric)"},
                "end_date": {"type": "string", "description": "Set to the forecast due date (YYYY-MM-DD)"},
                "days": {"type": "integer", "description": "Days of history (default: 60, max: 90)"},
            },
            "required": ["market_id", "end_date"],
        },
    },
}

# Source -> tool schema(s)
_SOURCE_TO_TOOLS: dict[str, list[dict]] = {
    "yfinance":   [_YFINANCE_TOOL],
    "fred":       [_FRED_TOOL],
    "dbnomics":   [_DBNOMICS_TOOL],
    "wikipedia":  [_WIKIPEDIA_TOC_TOOL, _WIKIPEDIA_SECTION_TOOL],
    # Market tools (polymarket, manifold) disabled — the crowd price is
    # sufficient; price history tools add noise without improving BI.
    # See Section 6 (Future Work) for potential improvements.
}


def get_source_tools(source: str) -> list[dict]:
    """Return tool schemas for a given question source, or [] if none."""
    return _SOURCE_TO_TOOLS.get(source, [])


def dispatch_source_tool(name: str, args: dict, cutoff_date: str) -> str:
    """Execute a source-specific tool call. Returns result text.

    All end_date arguments are clamped to cutoff_date to prevent leakage.
    """
    end_date = _clamp_end_date(args.get("end_date", ""), cutoff_date)
    if not end_date:
        return "Error: no end_date provided"
    freeze_dt = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(freeze_dt):
        return f"Invalid end_date: {end_date!r}"

    try:
        if name == "fetch_ts_yfinance":
            df = fetch_yfinance(args.get("ticker", ""), end_date)
            return _df_to_csv(df, end_date)
        elif name == "fetch_ts_fred":
            df = fetch_fred(args.get("series_id", ""), end_date)
            return _df_to_csv(df, end_date)
        elif name == "fetch_ts_dbnomics":
            df = fetch_dbnomics(args.get("url", ""), end_date)
            return _df_to_csv(df, end_date)
        elif name == "analyze_trend":
            src = args.get("source", "")
            comp = float(args.get("comparison_value", 0))
            res_date = args.get("resolution_date", "")
            if src == "yfinance":
                df = fetch_yfinance(args.get("ticker", ""), end_date)
            elif src == "fred":
                df = fetch_fred(args.get("series_id", ""), end_date)
            elif src == "dbnomics":
                df = fetch_dbnomics(args.get("url", ""), end_date)
            else:
                return f"Unknown source for analyze_trend: {src}"
            return _analyze_trend(df, comp, res_date, src)
        elif name == "fetch_wikipedia_toc":
            return _fetch_wikipedia_toc(args.get("url", ""), end_date)
        elif name == "fetch_wikipedia_section":
            return _fetch_wikipedia_section(
                args.get("url", ""), end_date, args.get("section", ""))
        elif name == "fetch_polymarket_info":
            days = min(int(args.get("days", 60)), 90)
            return _fetch_polymarket_info(args.get("url", ""), end_date, days=days)
        elif name == "fetch_manifold_info":
            days = min(int(args.get("days", 60)), 90)
            return _fetch_manifold_info(
                args.get("market_id", ""), end_date, days=days)
        else:
            return f"Unknown source tool: {name!r}"
    except Exception as e:
        return f"Error fetching data: {e}"


# Set of all source tool names (for dispatch routing)
SOURCE_TOOL_NAMES = {
    "fetch_ts_yfinance", "fetch_ts_fred", "fetch_ts_dbnomics",
    "analyze_trend",
    "fetch_wikipedia_toc", "fetch_wikipedia_section",
    "fetch_polymarket_info", "fetch_manifold_info",
}
