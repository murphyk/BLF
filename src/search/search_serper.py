"""
search_serper.py — Serper.dev (Google Search) API with date filtering, optional
page scraping, and LLM post-processing.

Functions:
  search_serper — search Google via Serper, optionally scrape page content,
                  client-side filter, optionally LLM filter+summarize

Shared utilities (filter_results, summarize_results, _age_flag, _parse_page_age)
are in search_lib.py.

Setup:
  SERPER_API_KEY — Serper.dev API key (get one at serper.dev/api-keys)
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import dotenv
import requests

from .search_lib import _RESULTS_SEPARATOR, _age_flag, filter_results, summarize_results, rewrite_queries

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

_SERPER_SEARCH_URL = "https://google.serper.dev/search"
_SERPER_SCRAPE_URL = "https://scrape.serper.dev"


def _tbs_param(cutoff_date, search_window_days=None):
    """Build the tbs date-range string for the Serper API.

    Returns 'cdr:1,cd_min:M/D/YYYY,cd_max:M/D/YYYY' where cd_max is cutoff_date
    and cd_min is search_window_days before it. When search_window_days is None,
    uses a 25-year window (effectively all time) so that the end-date filtering
    still applies. Returns None if cutoff_date is empty.
    """
    if not cutoff_date:
        return None
    _MAX_WINDOW_DAYS = 25 * 365
    days = search_window_days if search_window_days else _MAX_WINDOW_DAYS
    end = datetime.strptime(str(cutoff_date)[:10], "%Y-%m-%d")
    start = end - timedelta(days=days)
    return f"cdr:1,cd_min:{start.month}/{start.day}/{start.year},cd_max:{end.month}/{end.day}/{end.year}"


def _scrape_url(api_key, url, timeout=10):
    """Scrape a single URL via Serper's scrape endpoint. Returns markdown text or ''."""
    try:
        resp = requests.post(
            _SERPER_SCRAPE_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"url": url},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # Prefer markdown, fall back to text
        return data.get("markdown", "") or data.get("text", "") or ""
    except Exception:
        return ""


def _scrape_urls(api_key, urls, max_chars, debug=False):
    """Scrape multiple URLs in parallel. Returns {url: truncated_text}."""
    results = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_scrape_url, api_key, url): url for url in urls}
        for fut in as_completed(futures):
            url = futures[fut]
            text = fut.result()
            if text and max_chars:
                text = text[:max_chars]
            results[url] = text
            if debug:
                print(f"[search_serper] scraped {url[:60]}... → {len(text)} chars")
    return results


def search_serper(query, cutoff_date=None,
                  max_searches=1, max_results_per_search=10,
                  max_chars_highlights=5000,
                  search_window_days=None,
                  scrape=True,
                  llm=None, llm_filter=None, question="",
                  summarization_context="",
                  deadline=None, debug=False):
    """Search Google via Serper.dev and format results.

    query: str or list[str]. If str and max_searches > 1, generates max_searches-1
    paraphrases and searches all of them. If list, uses it directly.

    When cutoff_date is set, passes a tbs date range to the Serper API for
    server-side pre-filtering. A client-side _age_flag hard filter then drops
    any result whose date field is after cutoff_date.

    max_results_per_search: number of results per query (num), max 10.

    scrape: if True, fetches full page content for each kept result via the
    Serper scrape endpoint (scrape.serper.dev), truncated to max_chars_highlights.
    Costs 1 additional Serper credit per page scraped.

    When llm is set, runs filter_results then summarize_results on the raw output.

    Returns 10-tuple: (summarized, raw, filter_log, in_tok, out_tok,
                       f_in_tok, f_out_tok, s_in_tok, s_out_tok, total_results).
    """
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "SERPER_API_KEY must be set in the environment or .env file. "
            "Get a key at serper.dev/api-keys."
        )

    # Resolve query list
    if isinstance(query, list):
        queries = query
    elif max_searches > 1:
        queries = rewrite_queries(query, max_searches, llm=llm)
    else:
        queries = [query]

    cutoff_str = str(cutoff_date)[:10] if cutoff_date else ""
    tbs = _tbs_param(cutoff_date, search_window_days)
    num = min(max_results_per_search, 10)  # Serper returns max 10 per page

    # First pass: collect kept results (url, title, snippet, date, flag)
    kept_items = []
    total_results = 0
    n_filtered = 0

    for q_idx, q in enumerate(queries, 1):
        if deadline and time.time() > deadline:
            break
        if debug:
            print(f"\n[search_serper] query {q_idx}/{len(queries)}: {q!r}")
            if tbs:
                print(f"[search_serper] tbs={tbs!r}  num={num}")
            else:
                print(f"[search_serper] num={num}")

        body = {"q": q, "num": num}
        if tbs:
            body["tbs"] = tbs

        resp = requests.post(
            _SERPER_SEARCH_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json=body,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        items = data.get("organic", [])
        total_results += len(items)

        if debug:
            print(f"[search_serper] returned {len(items)} result(s)")

        for i, item in enumerate(items, 1):
            title   = item.get("title",   "") or ""
            url     = item.get("link",    "") or ""
            snippet = item.get("snippet", "") or ""
            date    = item.get("date",    "") or ""

            flag = _age_flag(date, cutoff_str)
            after_cutoff = cutoff_str and "AFTER CUTOFF" in flag

            if debug:
                status = "  [FILTERED OUT]" if after_cutoff else ""
                print(f"\n  [{i}] {title!r}{status}")
                print(f"      url:  {url}")
                print(f"      date: {date!r}{flag}")
                if not after_cutoff and snippet:
                    print(f"      snippet: {snippet[:200]!r}")

            if after_cutoff:
                n_filtered += 1
                continue

            kept_items.append({
                "title": title, "url": url, "snippet": snippet,
                "date": date, "flag": flag,
            })

    if debug and n_filtered:
        print(f"\n[search_serper] kept {len(kept_items)}, filtered out {n_filtered} post-cutoff result(s)")

    # Second pass: scrape kept URLs for full page content
    scraped = {}
    if scrape and kept_items:
        urls_to_scrape = list(dict.fromkeys(it["url"] for it in kept_items if it["url"]))
        if debug:
            print(f"\n[search_serper] scraping {len(urls_to_scrape)} page(s)...")
        scraped = _scrape_urls(api_key, urls_to_scrape, max_chars_highlights, debug=debug)

    # Build context lines
    context_lines = []
    for idx, it in enumerate(kept_items, 1):
        display_date = it["date"] if it["date"] else "unknown date"
        parts = [f"Search result {idx}: {it['title']}. Published on {display_date}.{it['flag']}", it["url"]]

        page_text = scraped.get(it["url"], "")
        if page_text:
            parts.append(f"Page content:\n{page_text}")
        elif it["snippet"]:
            snippet_trunc = it["snippet"][:max_chars_highlights] if max_chars_highlights else it["snippet"]
            parts.append(f"Highlights:\n{snippet_trunc}")

        context_lines.append("\n".join(parts))

    raw = _RESULTS_SEPARATOR.join(context_lines)

    if debug:
        print(f"\n{'─' * 40} RAW {'─' * 40}")
        print(raw[:3000] if len(raw) > 3000 else (raw or "(empty)"))
        if len(raw) > 3000:
            print(f"... ({len(raw)} chars total, truncated for display)")

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
