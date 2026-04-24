"""
search_lib.py — shared utilities for search_exa.py and search_claude.py.

Provides:
  _parse_page_age  — parse page_age string into datetime
  _age_flag        — compare page date to cutoff, return display flag
  filter_results   — LLM-based KEEP/DROP filter on search results
  summarize_results — LLM summarizer for filtered results
  _RESULTS_SEPARATOR — separator used to join/split individual search results
"""

import re
from datetime import datetime, timedelta

import dotenv

from agent.llm_client import chat

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

# Separator used to join/split individual search results in raw strings.
_RESULTS_SEPARATOR = "\n----------------------------------------------------\n"

_RELATIVE_RE = re.compile(
    r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", re.IGNORECASE
)
_UNIT_DAYS = {
    "second": 1 / 86400,
    "minute": 1 / 1440,
    "hour":   1 / 24,
    "day":    1,
    "week":   7,
    "month":  30,   # approximate
    "year":   365,
}


def _parse_page_age(page_age_str):
    """Parse a page_age string into a datetime.

    Handles absolute formats ('January 25, 2026', '2026-01-25') and relative
    formats ('1 month ago', '3 weeks ago') — relative ones are resolved against
    datetime.now() (the real current date, not the cutoff date).
    """
    if not page_age_str:
        return None
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(page_age_str.strip(), fmt)
        except ValueError:
            pass
    m = _RELATIVE_RE.search(page_age_str)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        return datetime.now() - timedelta(days=n * _UNIT_DAYS[unit])
    return None


def _age_flag(page_age_str, cutoff_date):
    """Return a display flag comparing page_age to cutoff_date."""
    if not page_age_str:
        return "  [no known date]"
    if not cutoff_date:
        return ""
    dt = _parse_page_age(page_age_str)
    if dt is None:
        return "  [?? unparseable date — check manually]"
    try:
        cutoff_dt = datetime.strptime(cutoff_date, "%Y-%m-%d")
    except ValueError:
        return ""
    if dt > cutoff_dt:
        return f"  [!! AFTER CUTOFF {cutoff_date}]"
    return f"  [ok, before {cutoff_date}]"


# ---------------------------------------------------------------------------
# filter_results / summarize_results
# ---------------------------------------------------------------------------

_AGE_FLAG_RE = re.compile(
    r"  \[(?:ok, before \d{4}-\d{2}-\d{2}"
    r"|no known date"
    r"|\?\? unparseable date — check manually"
    r"|!{2} AFTER CUTOFF \d{4}-\d{2}-\d{2})\]"
)


def _strip_age_flags(text):
    """Remove [ok, before ...], [no known date], etc. flags from result text."""
    return _AGE_FLAG_RE.sub("", text)


def filter_results(raw, cutoff_date, llm=None, debug=False):
    """LLM-based second-pass filter on search results.

    Splits raw by _RESULTS_SEPARATOR, sends all to the LLM in one call,
    and asks it to KEEP or DROP each one based on the content (not metadata flags).
    Age flags (like [ok, before ...]) are stripped before passing to the LLM so it
    judges purely on textual content — a page published before the cutoff may contain
    post-cutoff updates (e.g. liveblogs).

    Conservative AND logic: a result is kept only if the LLM says KEEP. The
    client-side page_age filter in the search engine (brave/exa/google) already
    drops results with page_age after cutoff before this stage.

    Returns (filtered_str, filter_log, input_tokens, output_tokens).
    filter_log is a human-readable string showing KEEP/DROP per result.
    """
    if not raw or not raw.strip():
        return "", "", 0, 0
    if not cutoff_date:
        return raw, "", 0, 0

    cutoff_str = str(cutoff_date)[:10]
    results = [r.strip() for r in raw.split(_RESULTS_SEPARATOR) if r.strip()]
    if not results:
        return "", "", 0, 0

    # Strip age flags so the LLM judges on content, not on potentially misleading
    # metadata (page_age can be the original publish date of an updated liveblog).
    cleaned = [_strip_age_flags(r) for r in results]

    numbered = "\n\n".join(f"[{i}]\n{r}" for i, r in enumerate(cleaned, 1))
    prompt = (
        f"You are a strict information-cutoff filter. The knowledge cutoff date is {cutoff_str}.\n"
        f"For each numbered search result, decide KEEP or DROP.\n\n"
        "Focus on the CONTENT of each result — the highlights, extra snippets, and any "
        "dates mentioned in the text body. Do NOT trust the 'Published on' date alone; "
        "pages are often updated after their original publish date.\n\n"
        "Rules:\n"
        "- If ANY part of the content describes events, outcomes, or updates from AFTER "
        f"{cutoff_str} → DROP the entire result\n"
        "- Look for dates in the text body (e.g. 'January 25, 2026', 'February 6, 2026'). "
        f"If any mentioned date is after {cutoff_str} and the text describes what happened "
        "on that date → DROP\n"
        "- Past tense describing events scheduled after cutoff (e.g. 'the climb was postponed', "
        "'he completed the ascent') → DROP\n"
        "- Liveblog/timeline pages with entries after cutoff → DROP\n"
        "- Content only about events before or on the cutoff date → KEEP\n\n"
        "Reply with one line per result: the number, a colon, KEEP or DROP, then a brief reason.\n"
        "Example:\n1: KEEP - article from Nov 2025 about the announcement\n"
        "2: DROP - contains Jan 25 update describing postponement\n\n"
        f"Results:\n{numbered}"
    )

    text, in_tok, out_tok, _ = chat(
        prompt, model=llm, max_tokens=len(results) * 30 + 100)

    decisions = {}
    reasons = {}
    for line in text.splitlines():
        m = re.match(r"\[?(\d+)\]?:\s*(KEEP|DROP)\b(.*)", line.strip(), re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            decisions[idx] = m.group(2).upper()
            reasons[idx] = m.group(3).strip().lstrip("- ").strip()

    # Build human-readable filter log
    log_lines = [f"Cutoff: {cutoff_str}  |  {len(results)} result(s)\n"]
    n_dropped = 0
    keep_idx = 0  # 0-based index into saved result files
    for i, r in enumerate(results, 1):
        lines = r.splitlines()
        label = lines[0][:120] if lines else f"result {i}"
        # URL is on the second non-empty line
        url_str = ""
        for ln in lines[1:]:
            ln = ln.strip()
            if ln.startswith("http"):
                url_str = f"  {ln}"
                break
        decision = decisions.get(i, "KEEP (default)")
        reason = reasons.get(i, "")
        reason_str = f"  ({reason})" if reason else ""
        if "DROP" not in decision:
            file_ref = f"  → result_{keep_idx}.md"
            keep_idx += 1
        else:
            file_ref = ""
            n_dropped += 1
        log_lines.append(f"[{i}] {decision}{reason_str}{file_ref}{url_str}  {label}")
    filter_log = "\n".join(log_lines)

    if debug:
        print(f"\n[filter_results] LLM decisions ({llm}): {n_dropped} dropped / {len(results)} total")
        for line in log_lines[1:]:
            print(f"  {line}")

    kept = [r for i, r in enumerate(results, 1) if decisions.get(i, "KEEP") != "DROP"]
    filtered = _RESULTS_SEPARATOR.join(kept)
    return filtered, filter_log, in_tok, out_tok


def summarize_prompt_template(question="", summarization_context=""):
    """Return the summarization prompt template (without search results).

    Useful for logging/display — shows the instructions the summarizer LLM receives.
    """
    if summarization_context:
        return (
            "You are an assistant to a superforecaster. Extract the facts and data from "
            "the search results below that are most relevant to predicting the outcome of "
            "this question.\n\n"
            f"Question: {question}\n\n"
            f"Resolution criteria: {summarization_context}\n\n"
            "Instructions:\n"
            "- Extract concrete facts, statistics, dates, and named-source expert opinions.\n"
            "- Note any quantitative data (prices, percentages, counts, trends).\n"
            "- Distinguish hard facts from speculation or editorial opinion.\n"
            "- Omit information that is not relevant to the resolution criteria.\n"
            "- Do NOT add your own analysis or forecast — only extract what the sources say.\n\n"
            "Search results:\n\n<search results omitted>"
        )
    if question:
        return (
            "Summarize the following search results clearly and concisely, "
            f"highlighting the most relevant facts for answering the question: {question}.\n\n"
            "Search results:\n\n<search results omitted>"
        )
    return ""


def summarize_results(filtered, llm=None, question="", summarization_context=""):
    """Summarize filtered search results into a concise context string.

    When summarization_context is provided, uses a question-aware prompt that
    extracts facts relevant to the specific resolution criteria rather than
    producing a generic summary.

    Returns (summary_str, input_tokens, output_tokens).
    Returns ("", 0, 0) if filtered is empty.
    """
    if not filtered or not filtered.strip():
        return "", 0, 0

    if summarization_context:
        prompt = (
            "You are an assistant to a superforecaster. Extract the facts and data from "
            "the search results below that are most relevant to predicting the outcome of "
            "this question.\n\n"
            f"Question: {question}\n\n"
            f"Resolution criteria: {summarization_context}\n\n"
            "Instructions:\n"
            "- Extract concrete facts, statistics, dates, and named-source expert opinions.\n"
            "- Note any quantitative data (prices, percentages, counts, trends).\n"
            "- Distinguish hard facts from speculation or editorial opinion.\n"
            "- Omit information that is not relevant to the resolution criteria.\n"
            "- Do NOT add your own analysis or forecast — only extract what the sources say.\n\n"
            f"Search results:\n\n{filtered}"
        )
    else:
        focus = f" for answering the question: {question}" if question else ""
        prompt = (
            "Summarize the following search results clearly and concisely, "
            f"highlighting the most relevant facts{focus}.\n\n"
            f"Search results:\n\n{filtered}"
        )

    text, in_tok, out_tok, _ = chat(prompt, model=llm, max_tokens=10000)
    return text, in_tok, out_tok


def rewrite_queries(query, n, llm=None):
    """Return [query] + (n-1) paraphrases of query.

    Uses an LLM to generate n-1 alternative phrasings. The original query is
    always first in the returned list. If n <= 1, returns [query].
    """
    if n <= 1:
        return [query]

    prompt = (
        f"Generate {n - 1} paraphrases of the following search query. "
        "Each paraphrase should capture the same information need but use different wording. "
        "Output one query per line, no numbering or bullet points.\n\n"
        f"Query: {query}"
    )
    text, _, _, _ = chat(prompt, model=llm, max_tokens=500)
    paraphrases = [line.strip() for line in text.splitlines() if line.strip()][: n - 1]
    return [query] + paraphrases
