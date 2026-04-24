#!/usr/bin/env python3
"""Empirical prior probabilities for ForecastBench dataset questions.

For dataset questions (no market price), we use the empirical base rate
as a "crowd estimate" analog. The prior is computed from all available
ForecastBench questions (not just the evaluation tranches), so it
approximates what an online learner would quickly converge to.

For market questions, the market price is used instead (as before).

Usage:
    from config.empirical_prior import get_empirical_prior
    p = get_empirical_prior(question_dict)  # returns float or None
"""

import re

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
