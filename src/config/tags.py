"""tags.py — Unified tagging system for question grouping and policy dispatch.

A "tag" assigns a question to a category within a "label space". Tags can be:
- Virtual (derived from question metadata): Qsource, FBQtype, Qtype, Atype
- LLM-classified (stored on disk): xinghua, ben

Usage:
    from config.tags import get_tag, list_tag_values, TAG_SPACES

    tag = get_tag(question, "Qtype")        # -> "timeseries" | "market" | "wikipedia" | "acled"
    tag = get_tag(question, "FBQtype")       # -> "market" | "dataset"
    tag = get_tag(question, "Qsource")      # -> "polymarket" | "fred" | ...
    tag = get_tag(question, "Atype")         # -> "binary-single" | "binary-multi"
    tag = get_tag(question, "xinghua")       # -> "Politics & Elections" | ...
    tag = get_tag(question, "ben")           # -> "Economics & Business" | ...

Label spaces:
    Qsource   — question source (polymarket, manifold, fred, dbnomics, etc.)
    FBQtype   — ForecastBench question type: "market" or "dataset"
    Qtype     — operational question type for policy dispatch:
                "timeseries" (yfinance, fred, dbnomics),
                "wikipedia", "acled", "market"
    Atype     — answer type: "binary-single" or "binary-multi"
                (multi = multiple resolution dates)
    xinghua   — LLM-classified (15 categories), from data/tags_xinghua/
    ben       — LLM-classified (9 categories), from data/tags_ben/
"""

import json
import os
import re


# ---------------------------------------------------------------------------
# Source classification rules
# ---------------------------------------------------------------------------

_MARKET_SOURCES = {"infer", "manifold", "metaculus", "polymarket"}
_DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}
_TIMESERIES_SOURCES = {"yfinance", "fred", "dbnomics"}

# All known label spaces
TAG_SPACES = ["Qsource", "FBQtype", "Qtype", "Atype", "xinghua", "ben"]


# ---------------------------------------------------------------------------
# Virtual tag functions
# ---------------------------------------------------------------------------

def _tag_qsource(q: dict) -> str:
    """Question source (polymarket, fred, etc.)."""
    return q.get("source", "unknown")


def _tag_fbqtype(q: dict) -> str:
    """ForecastBench question type: market or dataset."""
    src = q.get("source", "")
    if src in _MARKET_SOURCES:
        return "market"
    if src in _DATASET_SOURCES:
        return "dataset"
    # AIBQ2 and other custom sources default to market-like
    return "market"


def _tag_qtype(q: dict) -> str:
    """Operational question type for policy dispatch."""
    src = q.get("source", "")
    if src in _TIMESERIES_SOURCES:
        return "timeseries"
    if src == "wikipedia":
        return "wikipedia"
    if src == "acled":
        return "acled"
    return "market"


def _tag_atype(q: dict) -> str:
    """Answer type: binary-single or binary-multi."""
    rdates = q.get("resolution_dates", [])
    if isinstance(rdates, list) and len(rdates) > 1:
        return "binary-multi"
    return "binary-single"


# ---------------------------------------------------------------------------
# LLM-classified tags (disk lookup)
# ---------------------------------------------------------------------------

def _tag_classified(q: dict, label_space: str) -> str | None:
    """Look up an LLM-classified tag from data/tags_{label_space}/.

    Tries the full ID first, then strips the date suffix (e.g. _2025-12-07)
    since tags may have been classified with non-date-suffixed IDs.
    """
    src = q.get("source", "unknown")
    qid = str(q.get("id", "unknown"))
    safe_id = re.sub(r'[/\\:]', '_', qid)
    tags_dir = os.path.join("data", f"tags_{label_space}", src)
    # Try exact ID first
    path = os.path.join(tags_dir, f"{safe_id}.json")
    if not os.path.exists(path):
        # Try stripping date suffix (e.g. "0x123_2025-12-07" -> "0x123")
        stripped = re.sub(r'_\d{4}-\d{2}-\d{2}$', '', safe_id)
        if stripped != safe_id:
            path = os.path.join(tags_dir, f"{stripped}.json")
            if not os.path.exists(path):
                return None
        else:
            return None
    with open(path) as f:
        t = json.load(f)
    return t.get("category")


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

_VIRTUAL_TAGS = {
    "Qsource": _tag_qsource,
    "FBQtype": _tag_fbqtype,
    "Qtype": _tag_qtype,
    "Atype": _tag_atype,
}


def get_tag(question: dict, label_space: str) -> str | None:
    """Get the tag for a question in the given label space.

    Virtual label spaces (Qsource, FBQtype, Qtype, Atype) are computed
    from question metadata. LLM-classified spaces (xinghua, ben, etc.)
    are looked up from data/tags_{label_space}/.

    Returns None if the tag is not available (e.g., question not classified).
    """
    fn = _VIRTUAL_TAGS.get(label_space)
    if fn:
        return fn(question)
    return _tag_classified(question, label_space)


def list_tag_values(label_space: str) -> list[str] | None:
    """Return the known tag values for a label space, or None if dynamic.

    For virtual spaces with a fixed set of values, returns the list.
    For LLM-classified spaces, returns the category list.
    For Qsource, returns None (depends on available data).
    """
    if label_space == "FBQtype":
        return ["market", "dataset"]
    if label_space == "Qtype":
        return ["timeseries", "wikipedia", "acled", "market"]
    if label_space == "Atype":
        return ["binary-single", "binary-multi"]
    if label_space == "xinghua":
        from data.classify_questions import CATEGORIES_XINGHUA
        return CATEGORIES_XINGHUA
    if label_space == "ben":
        from data.classify_questions import CATEGORIES_BEN
        return CATEGORIES_BEN
    return None  # Qsource is dynamic


def discover_classified_spaces() -> list[str]:
    """Discover available LLM-classified tag spaces from data/tags_* dirs."""
    spaces = []
    if os.path.isdir("data"):
        for d in sorted(os.listdir("data")):
            if d.startswith("tags_") and os.path.isdir(os.path.join("data", d)):
                spaces.append(d.removeprefix("tags_"))
    return spaces


def get_tags_for_exam(exam: dict[str, list[str]], label_space: str) -> dict[tuple[str, str], str]:
    """Get tags for all questions in an exam.

    Returns {(source, qid): tag_value} for questions that have a tag.
    For virtual spaces, loads question JSON to compute the tag.
    For classified spaces, looks up from disk.
    """
    tags = {}
    is_virtual = label_space in _VIRTUAL_TAGS

    for source, ids in exam.items():
        for qid in ids:
            safe_id = re.sub(r'[/\\:]', '_', str(qid))
            if is_virtual:
                # Need to load question to compute virtual tag
                q_path = os.path.join("data", "questions", source, f"{safe_id}.json")
                if os.path.exists(q_path):
                    with open(q_path) as f:
                        q = json.load(f)
                    tag = get_tag(q, label_space)
                else:
                    # Minimal question dict for source-based tags
                    tag = get_tag({"source": source, "id": qid}, label_space)
            else:
                tag = _tag_classified({"source": source, "id": qid}, label_space)

            if tag:
                tags[(source, qid)] = tag

    return tags
