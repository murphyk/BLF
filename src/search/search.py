"""
search.py — unified web search dispatcher with knowledge-cutoff filtering.

Library:
  SearchResult — dataclass: summarized text, token counts, search count
  do_search    — unified dispatcher → SearchResult

CLI (lightweight smoke test):
  python3 search.py [QUERY] [--search-engine ENGINE] [--cutoff-date YYYY-MM-DD]

  Defaults to a single "nvidia stock price" query with cutoff 2026-01-04.
  For full debug/inspection tooling, use search_debug.py instead.
"""

import argparse
import re
import time
from dataclasses import dataclass

import dotenv

from .search_exa import search_exa
from .search_google import search_google
from .search_brave import search_brave
from .search_serper import search_serper
from .search_asknews import search_asknews
from .search_perplexity import search_perplexity
from .search_claude import search_claude, search_claude_v1, search_claude_v2

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    summarized:  str = ""  # filtered/summarized context injected into the reasoning prompt
    raw:         str = ""  # unfiltered/raw results before LLM filtering
    filter_log:  str = ""  # human-readable KEEP/DROP decisions from LLM date filtering
    in_tok:     int = 0    # total tokens consumed during search + summarization
    out_tok:    int = 0
    filter_in_tok:  int = 0   # tokens for LLM filter stage only
    filter_out_tok: int = 0
    summarize_in_tok:  int = 0   # tokens for LLM summarize stage only
    summarize_out_tok: int = 0
    nsearch:    int = 0    # number of search calls / results fetched
    n_cutoff:   int = 0


# ---------------------------------------------------------------------------
# do_search — unified dispatcher
# ---------------------------------------------------------------------------

_SEARCH_LLM = "openrouter/google/gemini-3-flash-preview"


def do_search(search_prompt, search_engine, *,
              cutoff_date="",
              max_tokens=4000, max_searches=5, max_results_per_search=10,
              max_chars_per_doc=5000,
              search_window_days=None,
              extra_snippets=True,
              serper_scrape=True,
              question="",
              summarization_context="",
              deadline=None,
              llm_summarize=None,
              debug=False):
    """Always returns SearchResult with both raw and summarized context.

    search_engine:         'asknews', 'brave', 'exa', 'google', 'claude', 'claude_v2', 'perplexity', 'sonar', 'sonar-pro', 'serper', or None/'none'.
    cutoff_date:           YYYY-MM-DD string; when set, each engine filters to pre-cutoff results.
    max_searches:          number of distinct search queries / tool calls.
    max_results_per_search: number of results per query.
    max_chars_per_doc:     character limit per document/extract (default 5000).
    search_window_days:    how many days before cutoff_date to include in the search window.
    extra_snippets:        Brave only: include extra_snippets in results (default True).
    serper_scrape:         Serper only: scrape full page content for kept results (default True).
    summarization_context:   When non-empty, summarize_results uses a question-aware prompt
                           that extracts facts relevant to these criteria.
    llm_summarize:         LLM for summarization. None=default (flash), "none"=skip summarization,
                           or a model name string.
    debug:                 verbose per-block/result output passed to the search function.

    Filter LLM is always gemini-3-flash. Summarization LLM is controlled by llm_summarize.
    """
    # Per-search deadline: 30s cap, but don't exceed the overall deadline.
    _PER_SEARCH_TIMEOUT = 30
    if deadline:
        search_deadline = min(time.time() + _PER_SEARCH_TIMEOUT, deadline)
    else:
        search_deadline = time.time() + _PER_SEARCH_TIMEOUT

    llm_search = _SEARCH_LLM
    # Resolve summarization LLM: None=default, "none"=skip, else use as model name
    if llm_summarize is None:
        llm_sum = _SEARCH_LLM
    elif str(llm_summarize).lower() == "none":
        llm_sum = None  # will be handled by each engine to skip summarization
    else:
        llm_sum = llm_summarize

    def _sr(tup):
        """Unpack the 10-value tuple from search functions into a SearchResult."""
        (summarized, raw, filter_log, in_tok, out_tok,
         f_in, f_out, s_in, s_out, nsearch) = tup
        return SearchResult(summarized=summarized, raw=raw, filter_log=filter_log,
                            in_tok=in_tok, out_tok=out_tok,
                            filter_in_tok=f_in, filter_out_tok=f_out,
                            summarize_in_tok=s_in, summarize_out_tok=s_out,
                            nsearch=nsearch)

    # When llm_sum is None (skip summarization), run engine with llm=None to get
    # raw results only, then filter in do_search and return filtered as summary.
    _skip_summarize = llm_sum is None
    _engine_llm = None if _skip_summarize else llm_sum

    if not search_engine or str(search_engine).lower() == "none":
        return SearchResult()

    sr = None
    if search_engine == "claude":
        sr = _sr(search_claude(
            search_prompt, llm=llm_search, max_tokens=max_tokens, max_searches=max_searches,
            cutoff_date=cutoff_date, llm_summarize=_engine_llm, question=question, debug=debug,
        ))
    elif search_engine == "claude_v1":
        sr = _sr(search_claude_v1(
            search_prompt, llm=llm_search, max_tokens=max_tokens, max_searches=max_searches,
            cutoff_date=cutoff_date, debug=debug,
        ))
    elif search_engine == "claude_v2":
        sr = _sr(search_claude_v2(
            search_prompt, llm=llm_search, max_tokens=max_tokens, max_searches=max_searches,
            cutoff_date=cutoff_date, max_chars_per_doc=max_chars_per_doc,
            llm_summarize=_engine_llm, question=question,
            summarization_context=summarization_context, debug=debug,
        ))
    elif search_engine == "exa":
        sr = _sr(search_exa(
            search_prompt, cutoff_date=cutoff_date,
            max_searches=max_searches, max_results_per_search=max_results_per_search,
            max_chars_highlights=max_chars_per_doc,
            search_window_days=search_window_days,
            llm=_engine_llm, llm_filter=llm_search, question=question,
            summarization_context=summarization_context, debug=debug,
        ))
    elif search_engine == "google":
        sr = _sr(search_google(
            search_prompt, cutoff_date=cutoff_date,
            max_searches=max_searches, max_results_per_search=max_results_per_search,
            max_chars_highlights=max_chars_per_doc,
            search_window_days=search_window_days,
            llm=_engine_llm, llm_filter=llm_search, question=question,
            summarization_context=summarization_context, debug=debug,
        ))
    elif search_engine == "brave":
        sr = _sr(search_brave(
            search_prompt, cutoff_date=cutoff_date,
            max_searches=max_searches, max_results_per_search=max_results_per_search,
            max_chars_highlights=max_chars_per_doc,
            search_window_days=search_window_days,
            extra_snippets=extra_snippets,
            llm=_engine_llm, llm_filter=llm_search, question=question,
            summarization_context=summarization_context,
            deadline=search_deadline, debug=debug,
        ))
    elif search_engine == "asknews":
        sr = _sr(search_asknews(
            search_prompt, cutoff_date=cutoff_date,
            max_searches=max_searches, max_results_per_search=max_results_per_search,
            max_chars_highlights=max_chars_per_doc,
            search_window_days=search_window_days,
            llm=_engine_llm, llm_filter=llm_search, question=question,
            summarization_context=summarization_context,
            deadline=search_deadline, debug=debug,
        ))
    elif search_engine in ("perplexity", "sonar", "sonar-pro"):
        _pplx_model = "sonar-pro" if search_engine == "sonar-pro" else "sonar"
        sr = _sr(search_perplexity(
            search_prompt, cutoff_date=cutoff_date,
            max_searches=max_searches, max_results_per_search=max_results_per_search,
            max_chars_highlights=max_chars_per_doc,
            search_window_days=search_window_days,
            model=_pplx_model,
            question=question,
            deadline=search_deadline, debug=debug,
        ))
    elif search_engine == "serper":
        sr = _sr(search_serper(
            search_prompt, cutoff_date=cutoff_date,
            max_searches=max_searches, max_results_per_search=max_results_per_search,
            max_chars_highlights=max_chars_per_doc,
            search_window_days=search_window_days,
            scrape=serper_scrape,
            llm=_engine_llm, llm_filter=llm_search, question=question,
            summarization_context=summarization_context,
            deadline=search_deadline, debug=debug,
        ))
    else:
        raise ValueError(f"Unknown search_engine {search_engine!r}")

    # When skipping summarization: engine returned raw results (llm=None).
    # Apply LLM filter, then use filtered text as the "summarized" output.
    if _skip_summarize and sr.raw:
        from .search_lib import filter_results
        filtered, filter_log, f_in, f_out = filter_results(
            sr.raw, cutoff_date, llm=llm_search, debug=debug)
        sr = SearchResult(
            summarized=filtered or sr.raw,
            raw=sr.raw,
            filter_log=filter_log,
            in_tok=f_in, out_tok=f_out,
            filter_in_tok=f_in, filter_out_tok=f_out,
            summarize_in_tok=0, summarize_out_tok=0,
            nsearch=sr.nsearch,
        )

    return sr


# ---------------------------------------------------------------------------
# Lightweight CLI (smoke test)
# ---------------------------------------------------------------------------

_DEFAULT_LLM = "openrouter/google/gemini-3-flash-preview"


def main():
    parser = argparse.ArgumentParser(
        description="Quick search smoke test. For full debug tooling, use search_debug.py."
    )
    parser.add_argument("query", nargs="?", default="nvidia stock price",
                        help="Search query (default: 'nvidia stock price')")
    parser.add_argument("--search-engine", default="brave",
                        choices=["asknews", "brave", "claude", "claude_v1", "claude_v2", "exa", "google",
                                 "perplexity", "sonar", "sonar-pro", "serper"])
    parser.add_argument("--cutoff-date", default="2026-01-04", metavar="YYYY-MM-DD")
    parser.add_argument("--model", default=_DEFAULT_LLM,
                        help=f"LLM for filter + summarize (default: {_DEFAULT_LLM})")
    parser.add_argument("--max-searches", type=int, default=1)
    parser.add_argument("--max-results-per-search", type=int, default=10)
    args = parser.parse_args()

    print(f"Query:  {args.query!r}")
    print(f"Engine: {args.search_engine}  Cutoff: {args.cutoff_date}  LLM: {args.model}")
    print()

    sr = do_search(
        search_prompt=args.query,
        search_engine=args.search_engine,
        cutoff_date=args.cutoff_date,
        max_searches=args.max_searches,
        max_results_per_search=args.max_results_per_search,
    )

    # Count kept/total from filter log
    n_total = n_dropped = 0
    if sr.filter_log:
        n_total = len(re.findall(r"^\[\d+\] ", sr.filter_log, re.MULTILINE))
        n_dropped = len(re.findall(r"^\[\d+\] DROP", sr.filter_log, re.MULTILINE))
    n_kept = n_total - n_dropped

    print(sr.summarized or "(empty)")
    print(f"\n[nsearch={sr.nsearch}"
          f"  filter: kept {n_kept}/{n_total}"
          f"  tokens: filter={sr.filter_in_tok}+{sr.filter_out_tok}"
          f" summarize={sr.summarize_in_tok}+{sr.summarize_out_tok}]")


if __name__ == "__main__":
    main()
