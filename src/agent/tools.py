"""tools.py — Tool definitions and dispatch for the agentic forecaster."""

import os
import re as _re

from agent.belief_state import BeliefState
from config.config import AgentConfig
from search import do_search, _SEARCH_LLM, summarize_results, _RESULTS_SEPARATOR, filter_results


# ---------------------------------------------------------------------------
# Source-aware URL blacklist (prevent outcome leakage from prediction markets)
# ---------------------------------------------------------------------------

# Domains whose question pages could leak resolved outcomes.
# Keyed by source name; each entry is a list of domain substrings to block.
# Applied to both search queries (strip site: directives) and results (drop URLs).
_SOURCE_BLOCKED_DOMAINS = {
    # Market sources — block their own sites to prevent outcome leakage
    "polymarket":  ["polymarket.com"],
    "manifold":    ["manifold.markets"],
    "metaculus":   ["metaculus.com"],
    "infer":       ["randforecastinginitiative.org"],
    "kalshi":      ["kalshi.com"],
    "predictit":   ["predictit.org"],
    # Dataset sources — block live data sites (use tools instead, which enforce date filtering)
    "dbnomics":    ["db.nomics.world"],
    "fred":        ["fred.stlouisfed.org"],
    "yfinance":    ["finance.yahoo.com"],
    "wikipedia":   ["en.wikipedia.org", "wikipedia.org"],
}

# Always block these regardless of source (prediction market aggregators)
_ALWAYS_BLOCKED_DOMAINS = [
    "metaculus.com/questions/",
    "manifold.markets/",
    "polymarket.com/",
    "predictit.org/",
    "kalshi.com/",
]


def _get_blocked_domains(source: str, question: dict | None = None) -> list[str]:
    """Return list of blocked domain substrings for a given source.

    Includes:
    - Always-blocked prediction market domains
    - Source-specific blocked domains
    - URLs from the question's resolution_criteria (could resolve the question)
    - The question's own URL
    """
    domains = list(_ALWAYS_BLOCKED_DOMAINS)
    for d in _SOURCE_BLOCKED_DOMAINS.get(source, []):
        if d not in domains:
            domains.append(d)
    # Block URLs found in resolution criteria and the question URL
    if question:
        import re as _re
        q_url = question.get("url", "")
        if q_url:
            # Extract domain from question URL
            m = _re.match(r'https?://(?:www\.)?([^/]+)', q_url)
            if m and m.group(1) not in " ".join(domains):
                domains.append(m.group(1))
        rc = question.get("resolution_criteria", "")
        if rc:
            for url_match in _re.findall(r'https?://(?:www\.)?([^/\s)]+)', rc):
                if url_match not in " ".join(domains):
                    domains.append(url_match)
    return domains


def _is_leaky_url(url: str) -> bool:
    """Check if a URL points to a prediction market page that could leak outcomes."""
    return any(d in url.lower() for d in _ALWAYS_BLOCKED_DOMAINS)


def _sanitize_query(query: str, blocked_domains: list[str]) -> str:
    """Strip site: directives that target blocked domains.

    Preserves -site: exclusions (those are helpful, not harmful).
    Other operators (between:, OR, quoted phrases, etc.) are left untouched.
    """
    for domain in blocked_domains:
        # Remove site:domain.com but NOT -site:domain.com (exclusion)
        query = _re.sub(
            rf'(?<!-)site:\s*"?{_re.escape(domain)}[^\s"]*"?\s*',
            '', query, flags=_re.IGNORECASE)
    return query.strip()


def _filter_blocked_urls(results: list[str], blocked_domains: list[str]) -> list[str]:
    """Drop results whose URL matches any blocked domain."""
    kept = []
    for r in results:
        # URL is typically on the second non-empty line
        lines = r.strip().splitlines()
        url = ""
        for ln in lines[:5]:
            ln = ln.strip()
            if ln.startswith("http"):
                url = ln
                break
        if url and any(d in url.lower() for d in blocked_domains):
            continue
        kept.append(r)
    return kept


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

_BELIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "p": {"type": "number", "description": "Updated probability (0.02-0.98)"},
        "base_rate_anchor": {"type": "string", "description": "Base rate estimate and rationale"},
        "evidence_for": {
            "type": "array", "items": {"type": "string"},
            "description": "Key evidence pushing probability UP. "
                           "ACCUMULATE across steps — add new items, don't rewrite. "
                           "Cite sources as (search_X_result_Y)."
        },
        "evidence_against": {
            "type": "array", "items": {"type": "string"},
            "description": "Key evidence pushing probability DOWN. "
                           "ACCUMULATE across steps — add new items, don't rewrite. "
                           "Cite sources as (search_X_result_Y)."
        },
        "key_uncertainties": {
            "type": "array", "items": {"type": "string"},
            "description": "What information would most change your estimate?"
        },
        "confidence": {
            "type": "string", "enum": ["low", "medium", "high"],
            "description": "How confident are you in your current estimate?"
        },
        "update_reasoning": {
            "type": "string",
            "description": "Explain HOW and WHY the new evidence changed your probability. "
                           "E.g. 'The petition requires 177,000 signatures in 120 days, "
                           "making the timeline very tight — p decreased from 0.35 to 0.25.'"
        },
    },
    "required": ["p", "confidence", "update_reasoning"],
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for information relevant to the question. "
            "Returns snippets from top results. Use targeted, specific queries. "
            "Results containing information after the forecast due date are "
            "automatically filtered out. "
            "After seeing snippets, you can call summarize_results to read "
            "the most promising results in detail."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Be specific and targeted."
                },
                "updated_belief": _BELIEF_SCHEMA,
            },
            "required": ["query", "updated_belief"],
        },
    },
}

SUMMARIZE_RESULTS_TOOL = {
    "type": "function",
    "function": {
        "name": "summarize_results",
        "description": (
            "Read and summarize selected results from a previous web search. "
            "This gives you detailed information from the full page content. "
            "Specify which search (by index) and which result numbers to "
            "summarize. Select the most relevant results based on snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_index": {
                    "type": "integer",
                    "description": "Which search to summarize (0-indexed)"
                },
                "result_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Which results to summarize (0-indexed)."
                },
                "updated_belief": _BELIEF_SCHEMA,
            },
            "required": ["search_index", "result_indices", "updated_belief"],
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": (
            "Submit your final probability estimate. This ends the forecasting loop. "
            "Only call this when you have gathered enough evidence and are ready "
            "to commit to a probability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "probability": {
                    "type": "number",
                    "description": "Final probability (0.02-0.98)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief summary of your reasoning"
                },
                "updated_belief": _BELIEF_SCHEMA,
            },
            "required": ["probability", "reasoning", "updated_belief"],
        },
    },
}

# Alternative submit tool for multi-resolution-date questions (dataset sources).
# Accepts a list of probabilities, one per resolution date.
SUBMIT_MULTI_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": (
            "Submit your final probability estimates — one per resolution date. "
            "This ends the forecasting loop. The probabilities list must have "
            "exactly one entry per resolution date, in the same order as listed "
            "in the question. Uncertainty should INCREASE with forecast horizon: "
            "probabilities for distant dates should be closer to 0.5."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "probabilities": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of probabilities (0.02-0.98), one per resolution date"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief summary of your reasoning"
                },
                "updated_belief": _BELIEF_SCHEMA,
            },
            "required": ["probabilities", "reasoning", "updated_belief"],
        },
    },
}

LOOKUP_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_url",
        "description": (
            "Fetch and summarize the content of a specific URL. "
            "Use this FIRST to look up any URLs mentioned in the question, "
            "background, or resolution criteria before doing web searches. "
            "Also useful for following links found in search results. "
            "Pages with content after the forecast due date are rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and summarize."
                },
                "updated_belief": _BELIEF_SCHEMA,
            },
            "required": ["url", "updated_belief"],
        },
    },
}


def _strip_belief_from_tools(tools: list) -> list:
    """Remove updated_belief parameter from all tool schemas (for nobelief mode)."""
    import copy
    stripped = []
    for tool in tools:
        tool = copy.deepcopy(tool)
        params = tool.get("function", {}).get("parameters", {})
        props = params.get("properties", {})
        if "updated_belief" in props:
            del props["updated_belief"]
            req = params.get("required", [])
            if "updated_belief" in req:
                params["required"] = [r for r in req if r != "updated_belief"]
        stripped.append(tool)
    return stripped


def get_tool_schemas(config: AgentConfig, source: str = "",
                     question: dict | None = None) -> list:
    """Return the list of tool schemas available to the agent.

    Always includes submit (multi-resolution variant if question has multiple
    resolution dates). Includes web_search, lookup_url, summarize_results
    unless search_engine="none". Includes source-specific tools if
    config.use_tools is True and the source has tools available.
    """
    # Use multi-resolution submit if question has >1 resolution date
    rdates = (question or {}).get("resolution_dates", [])
    if len(rdates) > 1:
        tools = [SUBMIT_MULTI_TOOL]
    else:
        tools = [SUBMIT_TOOL]
    if config.search_engine != "none":
        tools.extend([WEB_SEARCH_TOOL, LOOKUP_URL_TOOL, SUMMARIZE_RESULTS_TOOL])
    if config.use_tools and source:
        from agent.source_tools import get_source_tools
        tools.extend(get_source_tools(source))
    if config.nobelief:
        tools = _strip_belief_from_tools(tools)
    return tools


# ---------------------------------------------------------------------------
# Belief state update (shared across all tools)
# ---------------------------------------------------------------------------

def parse_belief_update(args: dict, state: BeliefState) -> BeliefState:
    """Parse the updated_belief from tool call args into a new BeliefState."""
    ub = args.get("updated_belief", {})
    if not ub:
        return state

    return BeliefState(
        p=max(0.02, min(0.98, ub.get("p", state.p))),
        base_rate_anchor=ub.get("base_rate_anchor", state.base_rate_anchor),
        evidence_for=ub.get("evidence_for", state.evidence_for),
        evidence_against=ub.get("evidence_against", state.evidence_against),
        key_uncertainties=ub.get("key_uncertainties", state.key_uncertainties),
        confidence=ub.get("confidence", state.confidence),
        update_reasoning=ub.get("update_reasoning", ""),
        searches_tried=state.searches_tried,  # managed by tools, not by LLM
        step=state.step,
    )


# ---------------------------------------------------------------------------
# Tool context: shared state passed to each tool function
# ---------------------------------------------------------------------------

class ToolContext:
    """Shared context passed to each tool implementation."""
    __slots__ = ("config", "search_cache", "output_dir", "question_stem",
                 "question", "cutoff_date", "deadline", "search_dir")

    def __init__(self, config, search_cache, output_dir, question_stem,
                 question, cutoff_date, deadline):
        self.config = config
        self.search_cache = search_cache
        self.output_dir = output_dir
        self.question_stem = question_stem
        self.question = question
        self.cutoff_date = cutoff_date
        self.deadline = deadline
        self.search_dir = os.path.join(output_dir, "searches", question_stem)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_web_search(args: dict, state: BeliefState, ctx: ToolContext) -> tuple[str, BeliefState, dict]:
    """Search the web and return snippets."""
    meta = {"tool": "web_search"}
    query = args.get("query", "")
    search_idx = len(ctx.search_cache)

    # Sanitize query: strip site: directives targeting blocked domains
    source = ctx.question.get("source", "")
    blocked = _get_blocked_domains(source, ctx.question if ctx.config.backtesting else None)

    # Also block the question's own URL when backtesting — but only if it's
    # a prediction market page (not a news article or subject URL like AIBQ2)
    question_url = ctx.question.get("url", "")
    if ctx.config.backtesting and question_url and _is_leaky_url(question_url):
        blocked = list(blocked)
        url_key = _re.sub(r'^https?://(www\.)?', '', question_url).rstrip('/')
        if url_key and url_key not in blocked:
            blocked.append(url_key)
        query = query.replace(question_url, "").strip()

    clean_query = _sanitize_query(query, blocked)
    if clean_query != query:
        meta["query_sanitized"] = True
        meta["query_original"] = query
    query = clean_query

    state.searches_tried = list(state.searches_tried) + [query]

    sr = do_search(
        search_prompt=query,
        search_engine=ctx.config.search_engine,
        cutoff_date=ctx.cutoff_date,
        max_searches=1,
        max_results_per_search=ctx.config.max_results_per_search,
        extra_snippets=True,
        llm_summarize="none",
        question=ctx.question.get("question", ""),
        deadline=ctx.deadline,
    )

    # Use LLM-filtered results (post-cutoff content dropped)
    filtered_text = sr.summarized or sr.raw
    results = []
    if filtered_text:
        results = [r.strip() for r in filtered_text.split(_RESULTS_SEPARATOR) if r.strip()]

    # Drop results from blocked domains (prediction market pages)
    n_before = len(results)
    results = _filter_blocked_urls(results, blocked)
    if len(results) < n_before:
        meta["n_blocked"] = n_before - len(results)

    # Save kept results to files
    os.makedirs(ctx.search_dir, exist_ok=True)
    for j, text in enumerate(results):
        with open(os.path.join(ctx.search_dir, f"search_{search_idx}_result_{j}.md"), "w") as f:
            f.write(text)
    if sr.filter_log:
        with open(os.path.join(ctx.search_dir, f"search_{search_idx}_filter_log.txt"), "w") as f:
            f.write(sr.filter_log)
    # Save rejected results for post-hoc analysis
    if sr.raw and sr.filter_log:
        all_raw = [r.strip() for r in sr.raw.split(_RESULTS_SEPARATOR) if r.strip()]
        kept_set = set(r.strip()[:200] for r in results)  # fingerprint by prefix
        rejected_dir = os.path.join(ctx.search_dir, "rejected")
        rej_idx = 0
        for j, raw_text in enumerate(all_raw):
            if raw_text.strip()[:200] not in kept_set:
                os.makedirs(rejected_dir, exist_ok=True)
                with open(os.path.join(rejected_dir,
                           f"search_{search_idx}_result_{j}.md"), "w") as f:
                    f.write(raw_text)
                rej_idx += 1

    ctx.search_cache[search_idx] = results

    # Build snippet response (truncated per result)
    snippets = []
    for j, r in enumerate(results):
        snippet = r[:500] + ("..." if len(r) > 500 else "")
        snippets.append(f"[Result {j}]\n{snippet}")

    if snippets:
        result_text = (
            f"Search #{search_idx} for '{query}' returned {len(results)} results.\n"
            f"Filtered results (snippets below). Use summarize_results({search_idx}, [indices]) "
            "to read full content of promising results.\n\n"
            + "\n\n".join(snippets)
        )
    else:
        result_text = f"Search #{search_idx} for '{query}' returned no results."

    meta.update(search_index=search_idx, query=query, n_results=len(results),
                tokens_in=sr.in_tok, tokens_out=sr.out_tok)
    return result_text, state, meta


def _tool_summarize_results(args: dict, state: BeliefState, ctx: ToolContext) -> tuple[str, BeliefState, dict]:
    """Read and summarize selected search results."""
    meta = {"tool": "summarize_results"}
    search_idx = args.get("search_index", 0)
    result_indices = args.get("result_indices", [])

    if search_idx not in ctx.search_cache:
        return (f"Error: search #{search_idx} not found. "
                f"Available: {list(ctx.search_cache.keys())}"), state, meta

    results = ctx.search_cache[search_idx]
    selected = []
    for j in result_indices:
        if 0 <= j < len(results):
            selected.append(results[j])
        else:
            selected.append(f"(result {j} not found)")

    combined = _RESULTS_SEPARATOR.join(selected)
    summary, in_tok, out_tok = summarize_results(
        combined,
        llm=_SEARCH_LLM,
        question=ctx.question.get("question", ""),
        summarization_context=ctx.question.get("resolution_criteria", ""),
    )
    if not summary:
        summary = "(No content to summarize)"

    # Save summary file
    os.makedirs(ctx.search_dir, exist_ok=True)
    summary_file = f"search_{search_idx}_summary.md"
    with open(os.path.join(ctx.search_dir, summary_file), "w") as f:
        f.write(f"Summary of search #{search_idx}, results {result_indices}\n\n{summary}")

    result_text = (
        f"Summary of search #{search_idx}, results {result_indices}. "
        f"Cite as (search_{search_idx}_summary).\n\n{summary}"
    )
    meta.update(search_index=search_idx, result_indices=result_indices,
                tokens_in=in_tok, tokens_out=out_tok)
    return result_text, state, meta


def _tool_lookup_url(args: dict, state: BeliefState, ctx: ToolContext) -> tuple[str, BeliefState, dict]:
    """Fetch and summarize a URL."""
    import requests
    meta = {"tool": "lookup_url"}
    url = args.get("url", "")
    if not url:
        return "Error: no URL provided", state, meta

    # Block URLs from leaky/data-source domains when backtesting
    if ctx.config.backtesting:
        source = ctx.question.get("source", "")
        blocked = _get_blocked_domains(source, ctx.question)
        url_lower = url.lower()
        if any(d in url_lower for d in blocked):
            meta["url"] = url
            meta["blocked"] = True
            return ("This URL is blocked during backtesting to prevent outcome leakage. "
                    "For data sources, use the source-specific tool (e.g. fetch_ts_dbnomics) "
                    "which enforces date filtering."), state, meta

    meta["url"] = url
    os.makedirs(ctx.search_dir, exist_ok=True)
    lookup_idx = sum(1 for f in os.listdir(ctx.search_dir)
                     if f.startswith("lookup_")) if os.path.isdir(ctx.search_dir) else 0

    _BROWSER_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    def _extract_text(html_text):
        """Extract readable text from HTML, return (title, content)."""
        title = ""
        try:
            from readability import Document
            doc = Document(html_text)
            title = doc.title()
            content = _re.sub(r'<[^>]+>', ' ', doc.summary())
            content = _re.sub(r'\s+', ' ', content).strip()
        except ImportError:
            m = _re.search(r'<title[^>]*>([^<]+)</title>', html_text, _re.IGNORECASE)
            title = m.group(1).strip() if m else ""
            content = _re.sub(r'<script[^>]*>.*?</script>', '', html_text, flags=_re.DOTALL)
            content = _re.sub(r'<style[^>]*>.*?</style>', '', content, flags=_re.DOTALL)
            content = _re.sub(r'<[^>]+>', ' ', content)
            content = _re.sub(r'\s+', ' ', content).strip()
        return title, content

    _MIN_CONTENT_LEN = 200  # if less, try Google cache (likely JS-rendered page)

    try:
        resp = requests.get(url, timeout=15, headers=_BROWSER_HEADERS)

        # On 403 or JS-rendered pages with little content, try Google cache
        need_cache = resp.status_code == 403
        if resp.ok and not need_cache:
            title, content = _extract_text(resp.text)
            if len(content) < _MIN_CONTENT_LEN:
                need_cache = True

        if need_cache:
            cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{url}"
            cache_resp = requests.get(cache_url, timeout=15, headers=_BROWSER_HEADERS)
            if cache_resp.ok:
                title, content = _extract_text(cache_resp.text)
                meta["fetched_via"] = "google_cache"
            elif not resp.ok:
                resp.raise_for_status()  # original request failed and cache failed too

        if not resp.ok and not need_cache:
            resp.raise_for_status()

        max_chars = 5000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        # Save raw fetch to file
        filename = f"lookup_{lookup_idx}.md"
        with open(os.path.join(ctx.search_dir, filename), "w") as f:
            f.write(f"URL: {url}\nTitle: {title}\n\n{content}")

        # Date-leakage filter: reject pages with post-cutoff content
        if ctx.config.backtesting and ctx.cutoff_date and len(content) >= _MIN_CONTENT_LEN:
            page_text = f"URL: {url}\nTitle: {title}\n\n{content}"
            filtered, filter_log, filt_in, filt_out = filter_results(
                page_text, ctx.cutoff_date, llm=_SEARCH_LLM)
            if not filtered.strip():
                # Save filter log
                log_file = f"lookup_{lookup_idx}_filter_log.txt"
                with open(os.path.join(ctx.search_dir, log_file), "w") as f:
                    f.write(filter_log)
                meta.update(lookup_index=lookup_idx, title=title,
                            tokens_in=filt_in, tokens_out=filt_out,
                            date_filtered=True)
                return (f"Page '{title or url}' was filtered out because it contains "
                        f"information from after the cutoff date ({ctx.cutoff_date}). "
                        f"Try a different source."), state, meta

        # If content is too short, skip summarization and inform the agent
        if len(content) < _MIN_CONTENT_LEN:
            result_text = (
                f"Fetched: {title or url}\n"
                f"Page returned very little text ({len(content)} chars) — likely "
                f"requires JavaScript or is paywalled. Use web_search instead to "
                f"find information about this topic."
            )
            meta.update(lookup_index=lookup_idx, title=title, tokens_in=0, tokens_out=0,
                        warning="insufficient_content")
            return result_text, state, meta

        # Summarize
        summary, in_tok, out_tok = summarize_results(
            content,
            llm=_SEARCH_LLM,
            question=ctx.question.get("question", ""),
            summarization_context=ctx.question.get("resolution_criteria", ""),
        )

        result_text = (
            f"Fetched: {title or url}\n"
            f"Saved as {filename}. Cite as (lookup_{lookup_idx}).\n\n"
            f"{summary or content[:2000]}"
        )
        meta.update(lookup_index=lookup_idx, title=title,
                    tokens_in=in_tok, tokens_out=out_tok)

    except requests.RequestException as e:
        result_text = f"Failed to fetch {url}: {e}"
        meta["error"] = str(e)

    return result_text, state, meta


def _tool_submit(args: dict, state: BeliefState, ctx: ToolContext) -> tuple[str, BeliefState, dict]:
    """Record the final probability (single or multi-resolution)."""
    reasoning = args.get("reasoning", "")

    # Multi-resolution: list of probabilities
    probs = args.get("probabilities")
    if probs and isinstance(probs, list):
        probs = [max(0.02, min(0.98, float(p))) for p in probs]
        state.p = probs[0]  # use first for belief state
        meta = {"tool": "submit", "final_ps": probs, "final_p": probs[0],
                "reasoning": reasoning}
        ps_str = ", ".join(f"{p:.4f}" for p in probs)
        return f"Submitted probabilities: [{ps_str}]", state, meta

    # Single probability (market questions)
    p = max(0.02, min(0.98, args.get("probability", state.p)))
    state.p = p
    meta = {"tool": "submit", "final_p": p, "reasoning": reasoning}
    return f"Submitted probability: {p:.4f}", state, meta


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TOOL_DISPATCH = {
    "web_search": _tool_web_search,
    "summarize_results": _tool_summarize_results,
    "lookup_url": _tool_lookup_url,
    "submit": _tool_submit,
}


def dispatch_tool(name: str, args: dict, state: BeliefState, config: AgentConfig,
                  search_cache: dict, output_dir: str, question_stem: str,
                  question: dict, cutoff_date: str,
                  deadline: float | None = None) -> tuple[str, BeliefState, dict]:
    """Dispatch a tool call. Returns (result_text, new_state, metadata).

    1. Parses updated_belief from args to get the LLM's new belief state.
    2. Dispatches to the tool-specific function.
    3. Returns the tool output, updated state, and metadata.
    """
    new_state = parse_belief_update(args, state)
    ctx = ToolContext(config, search_cache, output_dir, question_stem,
                      question, cutoff_date, deadline)

    tool_fn = _TOOL_DISPATCH.get(name)
    if tool_fn:
        return tool_fn(args, new_state, ctx)

    # Try source-specific tools
    from agent.source_tools import (SOURCE_TOOL_NAMES, NUMERIC_TS_SOURCES,
                              dispatch_source_tool, _analyze_trend)
    if name in SOURCE_TOOL_NAMES:
        result = dispatch_source_tool(name, args, cutoff_date)
        # Auto-append trend analysis for fetch_ts_* calls on numeric sources
        _TS_TOOL_TO_SOURCE = {
            "fetch_ts_yfinance": "yfinance",
            "fetch_ts_fred": "fred",
            "fetch_ts_dbnomics": "dbnomics",
        }
        ts_source = _TS_TOOL_TO_SOURCE.get(name)
        if ts_source and "No data" not in result:
            # Extract comparison value and resolution date from the question.
            # The comparison value is the LAST value in the fetched time series
            # (i.e. the value on the forecast_due_date), NOT the stale value
            # from the resolution criteria text (which may be from weeks earlier).
            q = question
            import re as _re

            # Parse the actual last value from the CSV result
            comp_val = None
            csv_lines = result.strip().split("\n")
            if len(csv_lines) >= 2:
                last_line = csv_lines[-1]  # last data row
                parts = last_line.split(",")
                if len(parts) >= 2:
                    try:
                        comp_val = float(parts[-1])
                    except ValueError:
                        pass

            # Fallback: try resolution criteria or freeze_datetime_value
            if comp_val is None:
                rc = q.get("resolution_criteria", "")
                comp_match = _re.search(r"most recent known value.*?was\s+([\d.,+-]+[0-9])", rc)
                if comp_match:
                    try:
                        comp_val = float(comp_match.group(1).replace(",", ""))
                    except ValueError:
                        pass
            if comp_val is None:
                fdv = q.get("freeze_datetime_value") or q.get("market_value")
                if fdv is not None:
                    try:
                        comp_val = float(fdv)
                    except (ValueError, TypeError):
                        pass
            res_dates = q.get("resolution_dates", [])
            res_date = res_dates[0] if res_dates else ""
            if comp_val is not None and res_date:
                # Re-fetch the data using the question's own identifiers
                from agent.data_tools import fetch_yfinance, fetch_fred, fetch_dbnomics
                df = None
                try:
                    q_url = q.get("url", "")
                    q_id = q.get("id", "")
                    if ts_source == "yfinance":
                        # Extract ticker from question id (e.g. "AEE_2026-03-15" -> "AEE")
                        ticker = q_id.split("_")[0] if "_" in q_id else q_id
                        df = fetch_yfinance(ticker, cutoff_date)
                    elif ts_source == "fred":
                        series_id = q_id.split("_")[0] if "_" in q_id else q_id
                        df = fetch_fred(series_id, cutoff_date)
                    elif ts_source == "dbnomics":
                        df = fetch_dbnomics(q_url, cutoff_date)
                except Exception:
                    pass
                if df is not None and not df.empty:
                    if ts_source == "dbnomics" and len(res_dates) > 0:
                        # Use harmonic model for seasonal data, all dates
                        from agent.source_tools import _harmonic_forecast
                        harmonic = _harmonic_forecast(df, comp_val, res_dates)
                        lines = [
                            "\n\n=== Seasonal Forecast (empirical exceedance) ===",
                            f"Threshold (value on forecast date): {comp_val:.4g}",
                            f"Method: historical exceedance rate (same time of year ±10 days)",
                            "",
                        ]
                        for rd, n_hist, p in harmonic:
                            lines.append(
                                f"  {rd}: P(value > {comp_val:.2f}) = {p:.2f} "
                                f"(based on {n_hist} historical observations)")
                        if len(harmonic) == 1:
                            lines.append(
                                f"\n>>> Use P = {harmonic[0][2]:.2f} "
                                f"as your primary anchor. <<<")
                        else:
                            ps = [f"{p:.2f}" for _, _, p in harmonic]
                            lines.append(
                                f"\n>>> Submit probabilities [{', '.join(ps)}] "
                                f"(one per resolution date) as your primary anchor. <<<")
                        result += "\n".join(lines)
                    else:
                        analysis = _analyze_trend(
                            df, comp_val, res_date, ts_source)
                        result += "\n\n" + analysis
                        # Append FRED-specific enhanced analysis (if enabled)
                        if ts_source == "fred" and config.fred_enhanced:
                            try:
                                from misc.fred_models import fred_enhanced_analysis
                                q_id = q.get("id", "")
                                series_id = q_id.split("_")[0] if "_" in q_id else q_id
                                fdd = q.get("forecast_due_date", cutoff_date)
                                fred_extra = fred_enhanced_analysis(
                                    df, comp_val, res_date, series_id, fdd)
                                if fred_extra:
                                    result += "\n" + fred_extra
                            except Exception:
                                pass
        # Save source tool result to disk for trace visualization
        tool_step = new_state.step
        result_fname = f"tool_{name}_{tool_step}.txt"
        os.makedirs(ctx.search_dir, exist_ok=True)
        with open(os.path.join(ctx.search_dir, result_fname), "w") as f:
            f.write(result)
        return result, new_state, {"tool": name, "result_file": result_fname}

    return f"Unknown tool: {name}", new_state, {"tool": name}
