#!/usr/bin/env python3
"""Empirical prior probabilities for ForecastBench dataset questions.

For dataset questions (no market price), we use the empirical base rate
as a "crowd estimate" analog.

Two API entry points:

  get_empirical_prior(question)
      Static, hardcoded marginals computed offline from ALL FB questions.
      Convenient and stable, but technically uses the question's own outcome
      and outcomes from later questions, so it is not strictly backtest-valid.

  get_loo_temporal_prior(question, source_pool=None, cutoff_date=...)
      Dynamic, leakage-free. The base rate is computed on the fly from the
      provided pool of questions, EXCLUDING the question itself and FILTERING
      to those whose entire outcome was knowable before cutoff_date (default:
      the question's forecast_due_date). Cached per (source, subtype, cutoff)
      so the per-question lookup is a single subtraction. Falls back to 0.5
      when the pool is empty (e.g. early in a backtest, or for AIBQ2 with no
      historical data).

For market questions, both entry points return None — the market price is
used as the prior instead.
"""

import os
import re
import json
import glob

# ---------------------------------------------------------------------------
# Empirical base rates by (source, question_subtype).
# Computed from ALL ForecastBench questions (~7700 questions, ~17000 resdates).
# These are stable marginal statistics that an online learner would converge
# to within a few dozen questions per type.
# ---------------------------------------------------------------------------

_PRIORS = {
    # ACLED: strong signal from question type
    ("acled", "10x_spike"):  0.002,
    ("acled", "increase"):   0.225,

    # Wikipedia: strong signal from article/question type
    ("wikipedia", "vaccine"):      0.000,
    ("wikipedia", "fide_elo"):     0.007,
    ("wikipedia", "fide_rank"):    0.680,
    ("wikipedia", "swimming"):     0.991,

    # Time-series sources: weak signal, use source-level prior
    ("fred", "all"):      0.422,
    ("yfinance", "all"):  0.576,
    ("dbnomics", "all"):  0.555,
}


def _classify_subtype(question: dict) -> str:
    """Classify a dataset question into its subtype.

    Returns a subtype string that keys into _PRIORS.
    """
    source = question.get("source", "")
    text = question.get("question", "")

    if source == "acled":
        if "ten times" in text:
            return "10x_spike"
        return "increase"

    if source == "wikipedia":
        text_lower = text.lower()
        if "vaccine" in text_lower:
            return "vaccine"
        if "at least 1%" in text or "Elo rating" in text:
            return "fide_elo"
        if "FIDE" in text or "ranking" in text_lower:
            return "fide_rank"
        if "world record" in text_lower or "swimming" in text_lower:
            return "swimming"
        return "vaccine"  # fallback for unknown wikipedia questions

    # fred, yfinance, dbnomics — single type per source
    return "all"


def get_empirical_prior(question: dict) -> float | None:
    """Return the empirical prior probability for a dataset question.

    Returns None for market questions (which use market_value instead).
    """
    source = question.get("source", "")
    _DATASET_SOURCES = {"acled", "wikipedia", "fred", "yfinance", "dbnomics"}
    if source not in _DATASET_SOURCES:
        return None

    subtype = _classify_subtype(question)
    return _PRIORS.get((source, subtype))


# ---------------------------------------------------------------------------
# Dynamic, leakage-free LOO-temporal prior.
# ---------------------------------------------------------------------------

_DATASET_SOURCES = {"acled", "wikipedia", "fred", "yfinance", "dbnomics"}

# Cache: source -> list of (qid, subtype, latest_resolution_date, mean_outcome).
_pool_cache: dict[str, list[tuple]] = {}


def _question_outcome(q: dict) -> float | None:
    """Return the mean of resolved_to (across resolution_dates if multi-res),
    or None if unresolved."""
    o = q.get("resolved_to")
    if isinstance(o, list):
        vals = [float(x) for x in o if x is not None]
        return sum(vals) / len(vals) if vals else None
    if o is None:
        return None
    try:
        return float(o)
    except (TypeError, ValueError):
        return None


def _question_latest_resolution(q: dict) -> str | None:
    """Return the latest resolution date (YYYY-MM-DD) from the question, or
    None if not resolved. Multi-resolution questions report the LATEST date,
    since all must be observed before they're knowable."""
    rds = q.get("resolution_dates")
    if isinstance(rds, list) and rds:
        try:
            return max(str(rd)[:10] for rd in rds if rd)
        except ValueError:
            return None
    rd = q.get("resolution_date")
    if rd:
        return str(rd)[:10]
    return None


def _load_source_pool(source: str, questions_dir: str) -> list[tuple]:
    """Load and cache the (qid, subtype, latest_res_date, outcome) tuples
    for every resolved question in data/questions/{source}/."""
    if source in _pool_cache:
        return _pool_cache[source]
    src_dir = os.path.join(questions_dir, source)
    pool: list[tuple] = []
    if os.path.isdir(src_dir):
        for fn in os.listdir(src_dir):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(src_dir, fn)) as f:
                    q = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            o = _question_outcome(q)
            if o is None:
                continue
            sub = _classify_subtype(q)
            rd = _question_latest_resolution(q)
            qid = str(q.get("id", fn[:-5]))
            pool.append((qid, sub, rd, o))
    _pool_cache[source] = pool
    return pool


# Sentinel for "use the question's forecast_due_date as cutoff".
_USE_FDD = object()


def get_loo_temporal_prior(question: dict,
                           cutoff_date=_USE_FDD,
                           questions_dir: str | None = None
                           ) -> float | None:
    """Backtest-valid empirical prior, computed on the fly.

    Returns the mean outcome of all resolved questions in the same
    (source, subtype) bucket, EXCLUDING the question itself, optionally
    restricted to those whose latest resolution_date is strictly before
    cutoff_date.

    cutoff_date semantics:
      - default (sentinel): use question["forecast_due_date"] — the strict
        backtest-valid choice for FB. A question is included only if it had
        already resolved at forecast time.
      - None: no temporal filter (identity-LOO only). Use this for non-temporal
        benchmarks like AIBQ2, where the questions are contemporary and there
        is no historical data to learn from anyway.
      - str ("YYYY-MM-DD"): use that explicit cutoff.

    Returns None for sources that do not use empirical priors (markets use
    market_value instead). Returns 0.5 when the pool is empty (early questions
    in a backtest, or AIBQ2 with cutoff at fdd).
    """
    source = question.get("source", "")
    if source not in _DATASET_SOURCES and source != "aibq2":
        return None
    if questions_dir is None:
        questions_dir = os.path.join(os.path.expanduser("~/BLF"),
                                     "data", "questions")

    if cutoff_date is _USE_FDD:
        cutoff_raw = question.get("forecast_due_date") or ""
        cutoff = str(cutoff_raw)[:10] or None
    elif cutoff_date is None:
        cutoff = None
    else:
        cutoff = str(cutoff_date)[:10] or None

    excl_id = str(question.get("id", ""))
    target_sub = _classify_subtype(question) if source in _DATASET_SOURCES else "all"

    pool = _load_source_pool(source, questions_dir)
    n, s = 0, 0.0
    for qid, sub, rd, o in pool:
        if qid == excl_id:
            continue
        if sub != target_sub:
            continue
        if cutoff is not None and rd is not None and rd >= cutoff:
            continue  # not yet resolved at forecast time
        n += 1
        s += o
    if n == 0:
        return 0.5  # no historical evidence — fall back to uninformative
    return s / n


def get_prior_explanation(question: dict) -> str:
    """Return a human-readable explanation of the prior."""
    source = question.get("source", "")
    subtype = _classify_subtype(question)
    prior = _PRIORS.get((source, subtype))
    if prior is None:
        return ""

    explanations = {
        ("acled", "10x_spike"): "Empirical base rate for 10x-spike ACLED questions",
        ("acled", "increase"): "Empirical base rate for any-increase ACLED questions",
        ("wikipedia", "vaccine"): "Empirical base rate for vaccine-development questions",
        ("wikipedia", "fide_elo"): "Empirical base rate for FIDE Elo +1% questions",
        ("wikipedia", "fide_rank"): "Empirical base rate for FIDE ranking questions",
        ("wikipedia", "swimming"): "Empirical base rate for swimming world record questions",
        ("fred", "all"): "Empirical base rate for FRED questions",
        ("yfinance", "all"): "Empirical base rate for yfinance questions",
        ("dbnomics", "all"): "Empirical base rate for DBnomics questions",
    }
    return explanations.get((source, subtype), f"Empirical base rate for {source}")
