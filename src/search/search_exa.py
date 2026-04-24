"""
search_exa.py — Exa.ai search with date filtering and LLM post-processing.

Functions:
  search_exa — search Exa, client-side filter, optionally LLM filter+summarize

Shared utilities (filter_results, summarize_results, _age_flag, _parse_page_age)
are in search_lib.py.
"""

import os
from datetime import datetime, timedelta, timezone

import dotenv

from .search_lib import _RESULTS_SEPARATOR, _age_flag, filter_results, summarize_results, rewrite_queries

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))


def search_exa(query, cutoff_date=None,
               max_searches=1, max_results_per_search=10,
               max_chars_highlights=500,
               search_window_days=None,
               llm=None, llm_filter=None, question="",
               summarization_context="", debug=False):
    """Call Exa and format results.

    query: str or list[str]. If str and max_searches > 1, generates max_searches-1
    paraphrases and searches all of them. If list, uses it directly.

    Highlights are always requested. max_chars_highlights is passed to Exa as
    highlights.maxCharacters (API-level); omitted from the request if None.
    max_chars_highlights is passed to Exa as highlights.maxCharacters.

    When cutoff_date is set, passes end_published_date to Exa's API as a
    first-pass filter. When search_window_days is also set, passes
    start_published_date = cutoff - search_window_days.
    A second client-side filter always drops any result whose published_date
    is after cutoff_date (using _age_flag), since Exa's API filter is not
    fully reliable.

    When llm is set, runs filter_results then summarize_results on the raw output.

    Returns (summarized, in_tok, out_tok, nsearch).
    summarized equals raw when llm is None.
    """
    from exa_py import Exa

    # Resolve query list
    if isinstance(query, list):
        queries = query
    elif max_searches > 1:
        llm_rw = llm
        queries = rewrite_queries(query, max_searches, llm=llm_rw)
    else:
        queries = [query]

    exa_client = Exa(api_key=os.environ["EXA_API_KEY"])

    end_published_date = None
    start_published_date = None
    if cutoff_date:
        cutoff_dt = (
            datetime.fromisoformat(str(cutoff_date)[:10])
            .replace(tzinfo=timezone.utc)
        )
        end_published_date = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if search_window_days:
            start_dt = cutoff_dt - timedelta(days=search_window_days)
            start_published_date = start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    highlights = {"maxCharacters": max_chars_highlights} if max_chars_highlights else {}
    exa_base_kwargs = {
        "num_results": max_results_per_search,
        "contents": {"highlights": highlights},
    }
    if end_published_date:
        exa_base_kwargs["end_published_date"] = end_published_date
    if start_published_date:
        exa_base_kwargs["start_published_date"] = start_published_date

    cutoff_str = str(cutoff_date)[:10] if cutoff_date else ""
    context_lines = []
    total_results = 0
    n_kept = n_filtered = 0

    for q_idx, q in enumerate(queries, 1):
        if debug:
            print(f"\n[search_exa] query {q_idx}/{len(queries)}: {q!r}")
            cutoff_info = f"  end_published_date={end_published_date}" if end_published_date else ""
            print(f"[search_exa] max_results_per_search={max_results_per_search}{cutoff_info}")

        results = exa_client.search(q, **exa_base_kwargs)
        result_list = results.results
        total_results += len(result_list)

        if debug:
            print(f"[search_exa] returned {len(result_list)} result(s)")

        for i, r in enumerate(result_list, 1):
            title     = getattr(r, "title",         "") or ""
            url       = getattr(r, "url",            "") or ""
            published = getattr(r, "published_date", "") or ""
            date_str  = published[:10] if published else ""

            flag = _age_flag(date_str, cutoff_str)
            after_cutoff = cutoff_str and "AFTER CUTOFF" in flag

            if debug:
                status = "  [FILTERED OUT]" if after_cutoff else ""
                print(f"\n  [{i}] {title!r}{status}")
                print(f"      url:            {url}")
                print(f"      published_date: {date_str!r}{flag}")
                if not after_cutoff:
                    hl_list = getattr(r, "highlights", []) or []
                    if hl_list:
                        print(f"      highlights ({len(hl_list)}):")
                        for hl in hl_list:
                            print(f"        {hl!r}")

            if after_cutoff:
                n_filtered += 1
                continue

            n_kept += 1
            display_date = date_str if date_str else "unknown date"
            parts = [f"Search result {n_kept}: {title}. Published on {display_date}.{flag}", url]

            hl_list = getattr(r, "highlights", []) or []
            if hl_list:
                parts.append(f"Highlights:\n{chr(10).join(hl_list)}")

            context_lines.append("\n".join(parts))

    if debug and n_filtered:
        print(f"\n[search_exa] kept {n_kept}, filtered out {n_filtered} post-cutoff result(s)")

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
