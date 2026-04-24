"""
search_brave.py — Brave Search API with date filtering and LLM post-processing.

Functions:
  search_brave — search Brave, client-side filter, optionally LLM filter+summarize

Shared utilities (filter_results, summarize_results, _age_flag, _parse_page_age)
are in search_lib.py.

Setup:
  BRAVE_API_KEY — Brave Search API subscription token
                  (get one at api-dashboard.search.brave.com)
"""

import os
import time
from datetime import datetime, timedelta

import dotenv
import requests

from .search_lib import _RESULTS_SEPARATOR, _age_flag, filter_results, summarize_results, rewrite_queries

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


_BRAVE_MAX_WINDOW_DAYS = 25 * 365  # ~25 years; effectively "all time"


def _freshness_param(cutoff_date, search_window_days=None):
    """Build the freshness date-range string for the Brave API.

    Returns 'YYYY-MM-DDtoYYYY-MM-DD' where the end date is cutoff_date and
    the start date is search_window_days before it. When search_window_days
    is None, uses a 25-year window (effectively all time) so that the end-date
    filtering still applies. Returns None if cutoff_date is empty.
    """
    if not cutoff_date:
        return None
    days = search_window_days if search_window_days else _BRAVE_MAX_WINDOW_DAYS
    end = datetime.strptime(str(cutoff_date)[:10], "%Y-%m-%d")
    start = end - timedelta(days=days)
    return f"{start.strftime('%Y-%m-%d')}to{end.strftime('%Y-%m-%d')}"


def search_brave(query, cutoff_date=None,
                 max_searches=1, max_results_per_search=10,
                 max_chars_highlights=5000,
                 search_window_days=None,
                 extra_snippets=True,
                 llm=None, llm_filter=None, question="",
                 summarization_context="",
                 deadline=None, debug=False):
    """Search Brave and format results.

    query: str or list[str]. If str and max_searches > 1, generates max_searches-1
    paraphrases and searches all of them. If list, uses it directly.

    When cutoff_date is set and search_window_days is given, passes a freshness
    date range (cutoff - search_window_days → cutoff) to the Brave API for
    server-side pre-filtering. A client-side _age_flag hard filter then drops
    any result whose age field is after cutoff_date.

    max_results_per_search: number of results per query (count), max 20.

    When llm is set, runs filter_results then summarize_results on the raw output.

    Returns (summarized, in_tok, out_tok, nsearch).
    summarized equals raw when llm is None.
    """
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "BRAVE_API_KEY must be set in the environment or .env file. "
            "Get a key at api-dashboard.search.brave.com."
        )

    # Resolve query list
    if isinstance(query, list):
        queries = query
    elif max_searches > 1:
        llm_rw = llm
        queries = rewrite_queries(query, max_searches, llm=llm_rw)
    else:
        queries = [query]

    cutoff_str = str(cutoff_date)[:10] if cutoff_date else ""
    freshness  = _freshness_param(cutoff_date, search_window_days)
    count      = min(max_results_per_search, 20)  # Brave API max

    context_lines = []
    total_results = 0
    n_kept = n_filtered = 0

    for q_idx, q in enumerate(queries, 1):
        if deadline and time.time() > deadline:
            break
        # Brave API rejects queries over ~400 chars; truncate at last word boundary
        if len(q) > 400:
            q = q[:400].rsplit(" ", 1)[0]
        if debug:
            print(f"\n[search_brave] query {q_idx}/{len(queries)}: {q!r}")
            if freshness:
                print(f"[search_brave] freshness={freshness!r}  count={count}")
            else:
                print(f"[search_brave] count={count}")

        params = {"q": q, "count": count}
        if freshness:
            params["freshness"] = freshness

        resp = requests.get(
            _BRAVE_SEARCH_URL,
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        items = data.get("web", {}).get("results", [])
        total_results += len(items)

        if debug:
            print(f"[search_brave] returned {len(items)} result(s)")

        for i, item in enumerate(items, 1):
            title    = item.get("title",       "") or ""
            url      = item.get("url",         "") or ""
            desc     = item.get("description", "") or ""
            # Brave returns age as a relative or absolute string (e.g. "2 months ago",
            # "January 4, 2026"). _age_flag / _parse_page_age handle both formats.
            age      = item.get("age",         "") or item.get("page_age", "") or ""

            flag = _age_flag(age, cutoff_str)
            after_cutoff = cutoff_str and "AFTER CUTOFF" in flag

            if debug:
                status = "  [FILTERED OUT]" if after_cutoff else ""
                print(f"\n  [{i}] {title!r}{status}")
                print(f"      url:  {url}")
                print(f"      age:  {age!r}{flag}")
                if not after_cutoff and desc:
                    print(f"      description: {desc[:200]!r}")

            if after_cutoff:
                n_filtered += 1
                continue

            n_kept += 1
            display_date = age if age else "unknown date"
            desc_trunc = desc[:max_chars_highlights] if max_chars_highlights else desc
            parts = [f"Search result {n_kept}: {title}. Published on {display_date}.{flag}", url]
            if desc_trunc:
                parts.append(f"Highlights:\n{desc_trunc}")

            if extra_snippets:
                extra = item.get("extra_snippets", []) or []
                if extra:
                    extra_text = "\n".join(s for s in extra if s)
                    if extra_text:
                        parts.append(f"Extra snippets:\n{extra_text}")

            context_lines.append("\n".join(parts))

    if debug and n_filtered:
        print(f"\n[search_brave] kept {n_kept}, filtered out {n_filtered} post-cutoff result(s)")

    raw = _RESULTS_SEPARATOR.join(context_lines)

    if debug:
        print(f"\n{'─' * 40} RAW {'─' * 40}")
        print(raw or "(empty)")

    if not llm:
        return raw, raw, "", 0, 0, 0, 0, 0, 0, total_results

    if deadline and time.time() > deadline:
        return raw, raw, "", 0, 0, 0, 0, 0, 0, total_results

    filtered, filter_log, f_in_tok, f_out_tok = filter_results(raw, cutoff_date, llm=llm_filter or llm, debug=debug)
    if debug:
        print(f"\n{'─' * 40} FILTERED (LLM pass) {'─' * 40}")
        print(filtered or "(empty — all results dropped)")

    if deadline and time.time() > deadline:
        return filtered or raw, raw, filter_log, f_in_tok, f_out_tok, f_in_tok, f_out_tok, 0, 0, total_results

    summarized, s_in_tok, s_out_tok = summarize_results(filtered, llm=llm, question=question, summarization_context=summarization_context)
    if debug:
        print(f"\n{'─' * 40} SUMMARIZED {'─' * 40}")
        print(summarized or "(empty)")

    in_tok  = f_in_tok  + s_in_tok
    out_tok = f_out_tok + s_out_tok
    return summarized, raw, filter_log, in_tok, out_tok, f_in_tok, f_out_tok, s_in_tok, s_out_tok, total_results
