"""search — Web search dispatcher and engine implementations."""

from .search import do_search, SearchResult, _SEARCH_LLM
from .search_lib import summarize_results, filter_results, _RESULTS_SEPARATOR
