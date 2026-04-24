"""
search_perplexity.py — Perplexity Sonar API search with date filtering.

Sonar returns AI-generated summaries grounded in web search results,
with citations and source URLs. No separate LLM filter/summarize step needed.

Setup:
  PERPLEXITY_API_KEY — Perplexity API key (get one at perplexity.ai/settings/api)

Models:
  sonar           — fast, ~$0.005/query (Llama 3.3 70B)
  sonar-pro       — deeper retrieval, ~$0.01-0.02/query
  sonar-reasoning-pro — chain-of-thought reasoning + search
"""

import os
import time

import dotenv
import requests

from .search_lib import _RESULTS_SEPARATOR

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))


def search_perplexity(query, cutoff_date=None,
                      max_searches=1, max_results_per_search=10,
                      max_chars_highlights=5000,
                      search_window_days=None,
                      model="sonar",
                      llm=None, llm_filter=None, question="",
                      deadline=None, debug=False):
    """Search via Perplexity Sonar API.

    query: str or list[str]. If list, each query is searched separately.

    Sonar returns a grounded AI summary with citations. The summary is used
    directly as the "summarized" output — no separate LLM filter/summarize needed.
    The raw output contains the individual search result metadata (title, URL, date).

    Returns 10-tuple: (summarized, raw, filter_log, in_tok, out_tok,
                       f_in_tok, f_out_tok, s_in_tok, s_out_tok, nsearch).
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key:
        raise ValueError(
            "PERPLEXITY_API_KEY must be set in the environment or .env file. "
            "Get a key at perplexity.ai/settings/api."
        )

    # Resolve query list
    if isinstance(query, list):
        queries = query
    else:
        queries = [query]

    cutoff_str = str(cutoff_date)[:10] if cutoff_date else ""

    # Build date filters (Perplexity uses m/d/yyyy format)
    search_params = {}
    if cutoff_str:
        try:
            parts = cutoff_str.split("-")
            before_date = f"{int(parts[1])}/{int(parts[2])}/{parts[0]}"
            search_params["search_before_date_filter"] = before_date
        except (IndexError, ValueError):
            pass
        if search_window_days:
            from datetime import datetime, timedelta
            end_dt = datetime.strptime(cutoff_str, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=search_window_days)
            after_date = f"{start_dt.month}/{start_dt.day}/{start_dt.year}"
            search_params["search_after_date_filter"] = after_date

    summaries = []
    raw_blocks = []
    total_in_tok = 0
    total_out_tok = 0
    total_results = 0

    for q_idx, q in enumerate(queries, 1):
        if deadline and time.time() > deadline:
            break
        if debug:
            print(f"\n[search_perplexity] query {q_idx}/{len(queries)}: {q!r}")
            if search_params:
                print(f"[search_perplexity] date filters: {search_params}")

        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": q}],
                **search_params,
            }
            resp = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            if debug:
                print(f"[search_perplexity] error: {e}")
            continue

        # Extract response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        # Token usage
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        total_in_tok += in_tok
        total_out_tok += out_tok

        # Citations and search results
        citations = data.get("citations", [])
        search_results = data.get("search_results", [])
        total_results += len(search_results) or len(citations)

        if debug:
            print(f"[search_perplexity] {len(citations)} citations, "
                  f"{len(search_results)} search_results, "
                  f"tokens: {in_tok}+{out_tok}")

        if content:
            summaries.append(content)

        # Build raw output from search results metadata
        for i, sr in enumerate(search_results, 1):
            title = sr.get("title", "") or ""
            url = sr.get("url", "") or ""
            pub_date = sr.get("date", "") or sr.get("publish_date", "") or ""
            display_date = str(pub_date)[:10] if pub_date else "unknown date"
            raw_blocks.append(
                f"Source [{i}]: {title}. Published: {display_date}.\n{url}"
            )

        # If no search_results, use citations as raw
        if not search_results and citations:
            for i, url in enumerate(citations, 1):
                raw_blocks.append(f"Source [{i}]: {url}")

    # Combine
    summarized = "\n\n".join(summaries)
    raw = _RESULTS_SEPARATOR.join(raw_blocks)

    if debug:
        print(f"\n{'─' * 40} SUMMARIZED {'─' * 40}")
        print(summarized[:500] if summarized else "(empty)")

    # Sonar does its own search+summarize, so all tokens are "summarize" tokens
    # (no separate filter stage)
    return (summarized, raw, "", total_in_tok, total_out_tok,
            0, 0, total_in_tok, total_out_tok, total_results)
