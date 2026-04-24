"""
search_google.py — Vertex AI Search (Discovery Engine) with date filtering and LLM post-processing.

Functions:
  search_google — search the public web via Vertex AI Search, client-side filter,
                  optionally LLM filter+summarize

Shared utilities (filter_results, summarize_results, _age_flag, _parse_page_age)
are in search_lib.py.

Setup:
  1. Create a Vertex AI Search app at console.cloud.google.com/gen-app-builder
     - Choose "Search" type, "Web search" data store, "Search the entire web"
  2. Enable the Discovery Engine API and create an API key
  3. Set environment variables:
       GOOGLE_API_KEY        — API key with Discovery Engine API enabled
       GOOGLE_CLOUD_PROJECT  — Google Cloud project ID
       GOOGLE_ENGINE_ID      — Vertex AI Search engine ID (from step 1)
"""

import os
import re
from datetime import datetime, timedelta

import dotenv
import requests

from .search_lib import _RESULTS_SEPARATOR, _age_flag, filter_results, summarize_results, rewrite_queries

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

_SEARCH_LITE_URL = (
    "https://discoveryengine.googleapis.com/v1"
    "/projects/{project}/locations/global/collections/default_collection"
    "/engines/{engine}/servingConfigs/default_search:searchLite"
)

_B_TAG_RE = re.compile(r"</?b>", re.IGNORECASE)
_DATE_PREFIX_RE = re.compile(r"^([A-Za-z]+ \d{1,2},\s*\d{4})\s*[—\-–]")


def _strip_tags(text):
    """Remove <b></b> highlight tags from a snippet."""
    return _B_TAG_RE.sub("", text)


def _extract_date(derived):
    """Try to extract a publication date string from a derivedStructData dict.

    Vertex AI Search web results do not return a structured date field.
    We fall back to parsing a date prefix from the first snippet
    (e.g. 'Jan 4, 2026 — some text...'), which Google often prepends.
    Returns "" if no date is found.
    """
    for s in derived.get("snippets", []):
        snippet_text = s.get("snippet", "") if isinstance(s, dict) else str(s)
        snippet_text = _strip_tags(snippet_text).strip()
        m = _DATE_PREFIX_RE.match(snippet_text)
        if m:
            return m.group(1)
    return ""


def search_google(query, cutoff_date=None,
                  max_searches=1, max_results_per_search=10,
                  max_chars_highlights=500,
                  search_window_days=None,
                  llm=None, llm_filter=None, question="",
                  summarization_context="", debug=False):
    """Search the public web via Vertex AI Search (Discovery Engine searchLite).

    query: str or list[str]. If str and max_searches > 1, generates max_searches-1
    paraphrases and searches all of them. If list, uses it directly.

    When cutoff_date is set, appends 'before:YYYY-MM-DD' to each query for
    API-level pre-filtering. When search_window_days is also set, appends
    'after:YYYY-MM-DD' (cutoff - search_window_days).
    A client-side _age_flag hard filter then drops any result whose extracted
    snippet date is after cutoff_date.

    max_results_per_search: number of results per query (pageSize), max 10.

    When llm is set, runs filter_results then summarize_results on the raw output.

    Returns (summarized, in_tok, out_tok, nsearch).
    summarized equals raw when llm is None.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    engine  = os.environ.get("GOOGLE_ENGINE_ID", "")
    if not api_key or not project or not engine:
        raise ValueError(
            "GOOGLE_API_KEY, GOOGLE_CLOUD_PROJECT, and GOOGLE_ENGINE_ID must be set "
            "in the environment or .env file.\n"
            "Setup: create a Vertex AI Search app at console.cloud.google.com/gen-app-builder "
            "with 'Search the entire web' enabled."
        )

    url = _SEARCH_LITE_URL.format(project=project, engine=engine)

    # Resolve query list
    if isinstance(query, list):
        queries = query
    elif max_searches > 1:
        llm_rw = llm
        queries = rewrite_queries(query, max_searches, llm=llm_rw)
    else:
        queries = [query]

    cutoff_str = str(cutoff_date)[:10] if cutoff_date else ""
    after_str = ""
    if cutoff_str and search_window_days:
        start_dt = datetime.strptime(cutoff_str, "%Y-%m-%d") - timedelta(days=search_window_days)
        after_str = start_dt.strftime("%Y-%m-%d")
    num = min(max_results_per_search, 10)  # Discovery Engine max pageSize for web

    context_lines = []
    total_results = 0
    n_kept = n_filtered = 0

    for q_idx, q in enumerate(queries, 1):
        # Append before:/after: operators for API-level date pre-filtering
        api_query = q
        if after_str:
            api_query = f"{api_query} after:{after_str}"
        if cutoff_str:
            api_query = f"{api_query} before:{cutoff_str}"

        if debug:
            print(f"\n[search_google] query {q_idx}/{len(queries)}: {api_query!r}")
            print(f"[search_google] pageSize={num}")

        body = {
            "query": api_query,
            "pageSize": num,
            "contentSearchSpec": {
                "snippetSpec": {"returnSnippet": True},
            },
        }
        resp = requests.post(url, params={"key": api_key}, json=body, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("results", [])
        total_results += len(items)

        if debug:
            print(f"[search_google] returned {len(items)} result(s)")

        for i, item in enumerate(items, 1):
            doc     = item.get("document", {})
            derived = doc.get("derivedStructData", {})

            title    = derived.get("title", "") or ""
            url_link = derived.get("link",  "") or ""
            snippets = derived.get("snippets", [])
            snippet  = ""
            if snippets:
                raw_snip = snippets[0].get("snippet", "") if isinstance(snippets[0], dict) else str(snippets[0])
                snippet  = _strip_tags(raw_snip).strip()

            date_str = _extract_date(derived)
            flag = _age_flag(date_str, cutoff_str)
            after_cutoff = cutoff_str and "AFTER CUTOFF" in flag

            if debug:
                status = "  [FILTERED OUT]" if after_cutoff else ""
                print(f"\n  [{i}] {title!r}{status}")
                print(f"      url:  {url_link}")
                print(f"      date: {date_str!r}{flag}")
                if not after_cutoff and snippet:
                    print(f"      snippet: {snippet[:200]!r}")

            if after_cutoff:
                n_filtered += 1
                continue

            n_kept += 1
            display_date = date_str if date_str else "unknown date"
            snippet_trunc = snippet[:max_chars_highlights] if max_chars_highlights else snippet
            parts = [f"Search result {n_kept}: {title}. Published on {display_date}.{flag}", url_link]
            if snippet_trunc:
                parts.append(f"Highlights:\n{snippet_trunc}")
            context_lines.append("\n".join(parts))

    if debug and n_filtered:
        print(f"\n[search_google] kept {n_kept}, filtered out {n_filtered} post-cutoff result(s)")

    raw = _RESULTS_SEPARATOR.join(context_lines)

    if debug:
        print(f"\n{'─' * 40} RAW {'─' * 40}")
        print(raw or "(empty)")

    if not llm:
        return raw, raw, "", 0, 0, 0, 0, 0, 0, total_results

    filtered, filter_log, f_in_tok, f_out_tok = filter_results(raw, cutoff_date, llm=llm_filter or llm, debug=debug)
    if debug:
        print(f"\n{'─' * 40} FILTERED (LLM pass) {'─' * 40}")
        print(filtered or "(empty — all results dropped)")

    summarized, s_in_tok, s_out_tok = summarize_results(filtered, llm=llm, question=question, summarization_context=summarization_context)
    if debug:
        print(f"\n{'─' * 40} SUMMARIZED {'─' * 40}")
        print(summarized or "(empty)")

    in_tok  = f_in_tok  + s_in_tok
    out_tok = f_out_tok + s_out_tok
    return summarized, raw, filter_log, in_tok, out_tok, f_in_tok, f_out_tok, s_in_tok, s_out_tok, total_results
