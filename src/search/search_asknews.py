"""
search_asknews.py — AskNews API search with date filtering.

Functions:
  search_asknews — search AskNews, return pre-structured article summaries

AskNews returns pre-summarized, LLM-ready article summaries, so no additional
LLM filter/summarize step is needed (though one can be applied optionally).

Setup (pick one):
  ASKNEWS_API_KEY       — AskNews API key (simpler, from API Keys tab)
  or:
  ASKNEWS_CLIENT_ID     — AskNews OAuth2 client ID
  ASKNEWS_CLIENT_SECRET — AskNews OAuth2 client secret
                          (get credentials at my.asknews.app > Settings > API Credentials)

  pip install asknews
"""

import os
import time
from datetime import datetime, timedelta

import dotenv

from .search_lib import _RESULTS_SEPARATOR, filter_results, summarize_results, rewrite_queries

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))


def search_asknews(query, cutoff_date=None,
                   max_searches=1, max_results_per_search=10,
                   max_chars_highlights=5000,
                   search_window_days=None,
                   llm=None, llm_filter=None, question="",
                   summarization_context="",
                   deadline=None, debug=False):
    """Search AskNews and format results.

    query: str or list[str]. If str and max_searches > 1, generates paraphrases.

    AskNews returns pre-summarized articles with title, summary, source, and
    publication date. The `historical=True` flag searches the full archive
    (back to 2023); otherwise only the last 48 hours are searched.

    Returns 10-tuple: (summarized, raw, filter_log, in_tok, out_tok,
                       f_in_tok, f_out_tok, s_in_tok, s_out_tok, nsearch).
    """
    from asknews_sdk import AskNewsSDK

    api_key = os.environ.get("ASKNEWS_API_KEY", "")
    client_id = os.environ.get("ASKNEWS_CLIENT_ID", "")
    client_secret = os.environ.get("ASKNEWS_CLIENT_SECRET", "")

    if api_key:
        ask = AskNewsSDK(api_key=api_key, scopes=["news"])
    elif client_id and client_secret:
        ask = AskNewsSDK(client_id=client_id, client_secret=client_secret,
                         scopes=["news"])
    else:
        raise ValueError(
            "Set ASKNEWS_API_KEY (or ASKNEWS_CLIENT_ID + ASKNEWS_CLIENT_SECRET) "
            "in the environment or .env file. Get credentials at my.asknews.app."
        )

    # Resolve query list
    if isinstance(query, list):
        queries = query
    elif max_searches > 1:
        queries = rewrite_queries(query, max_searches, llm=llm)
    else:
        queries = [query]

    cutoff_str = str(cutoff_date)[:10] if cutoff_date else ""

    # Build time range: AskNews requires both start_timestamp and end_timestamp
    # together (end_timestamp alone returns nothing). Max window is 160 days.
    _MAX_WINDOW_DAYS = 150  # stay under the 160-day API limit
    start_ts = None
    end_ts = None
    if cutoff_str:
        end_dt = datetime.strptime(cutoff_str, "%Y-%m-%d")
        end_ts = int(end_dt.timestamp())
        window = min(search_window_days or _MAX_WINDOW_DAYS, _MAX_WINDOW_DAYS)
        start_dt = end_dt - timedelta(days=window)
        start_ts = int(start_dt.timestamp())

    context_lines = []
    total_results = 0
    seen_urls = set()

    for q_idx, q in enumerate(queries, 1):
        if deadline and time.time() > deadline:
            break
        if debug:
            print(f"\n[search_asknews] query {q_idx}/{len(queries)}: {q!r}")

        try:
            kwargs = dict(
                query=q,
                n_articles=max_results_per_search,
                return_type="dicts",
                method="kw",
                historical=True,
            )
            if start_ts and end_ts:
                kwargs["start_timestamp"] = start_ts
                kwargs["end_timestamp"] = end_ts

            response = ask.news.search_news(**kwargs)
        except Exception as e:
            if debug:
                print(f"[search_asknews] error: {e}")
            continue

        articles = response.as_dicts or []
        total_results += len(articles)

        if debug:
            print(f"[search_asknews] returned {len(articles)} article(s)")

        for i, article in enumerate(articles, 1):
            url = str(getattr(article, "article_url", "") or "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = getattr(article, "title", "") or ""
            summary = getattr(article, "summary", "") or ""
            pub_date = getattr(article, "pub_date", "") or ""
            source = getattr(article, "source_id", "") or ""

            # Client-side cutoff filter
            if cutoff_str and pub_date:
                try:
                    pub_str = str(pub_date)[:10]
                    if pub_str > cutoff_str:
                        if debug:
                            print(f"  [{i}] FILTERED (after cutoff): {title!r}")
                        continue
                except (TypeError, ValueError):
                    pass

            if debug:
                print(f"  [{i}] {title!r} ({source}, {pub_date})")

            display_date = str(pub_date)[:10] if pub_date else "unknown date"
            summary_trunc = summary[:max_chars_highlights] if max_chars_highlights else summary
            parts = [
                f"Search result {len(context_lines)+1}: {title}. "
                f"Published on {display_date}. Source: {source}.",
            ]
            if url:
                parts.append(url)
            if summary_trunc:
                parts.append(f"Summary:\n{summary_trunc}")

            context_lines.append("\n".join(parts))

    raw = _RESULTS_SEPARATOR.join(context_lines)

    if debug:
        print(f"\n{'─' * 40} RAW {'─' * 40}")
        print(raw or "(empty)")

    if not llm:
        return raw, raw, "", 0, 0, 0, 0, 0, 0, total_results

    if deadline and time.time() > deadline:
        return raw, raw, "", 0, 0, 0, 0, 0, 0, total_results

    # Optional LLM filter + summarize (AskNews results are already good quality,
    # but filtering can still help with cutoff enforcement)
    filtered, filter_log, f_in_tok, f_out_tok = filter_results(
        raw, cutoff_date, llm=llm_filter or llm, debug=debug)

    if deadline and time.time() > deadline:
        return filtered or raw, raw, filter_log, f_in_tok, f_out_tok, f_in_tok, f_out_tok, 0, 0, total_results

    summarized, s_in_tok, s_out_tok = summarize_results(
        filtered, llm=llm, question=question, summarization_context=summarization_context)

    in_tok = f_in_tok + s_in_tok
    out_tok = f_out_tok + s_out_tok
    return summarized, raw, filter_log, in_tok, out_tok, f_in_tok, f_out_tok, s_in_tok, s_out_tok, total_results
