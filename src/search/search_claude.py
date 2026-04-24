"""
search_claude.py — Claude web_search_20260209 search with date filtering.

Functions:
  search_claude    — shim: calls search_claude_v2 by default.
  search_claude_v1 — v1: call Claude with web_search tool; Claude synthesizes
                     results internally (code-based date filtering optional).
                     Returns (summarized, in_tok, out_tok, nsearch).
  search_claude_v2 — v2: 3-stage pipeline matching search_exa structure:
                     Stage 1 retrieves structured RESULT blocks via web_search,
                     Stage 2 LLM-filters by date, Stage 3 LLM-summarizes.
                     Returns (summarized, in_tok, out_tok, nsearch).

Debug block printers (used when debug=True):
  print_text_block, print_server_tool_use, print_web_search_result,
  print_code_execution_tool_use, print_code_execution_result,
  print_unknown_block, _print_block
"""

import json
import re
import textwrap

import anthropic
import dotenv

from .search_lib import _age_flag, _RESULTS_SEPARATOR, filter_results, summarize_results, rewrite_queries

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

# Approximate number of results returned per web_search tool call.
# Used to convert max_results → max_searches for the Anthropic API.
CLAUDE_NUM_RESULTS_PER_SEARCH = 10


# ---------------------------------------------------------------------------
# Internal prompt builders
# ---------------------------------------------------------------------------

def _cutoff_system_prompt(cutoff_date, max_searches):
    """Build the system prompt for cutoff-aware searches."""
    return (
        f"Today's date is {cutoff_date}. "
        f"You have at most {max_searches} search calls available. "
        "The web_search tool and `await web_search()` inside code execution share the same quota — "
        "once the limit is reached, any further search calls will fail with an error.\n\n"
        "RECOMMENDED APPROACH: Do all your searches inside a SINGLE code execution block "
        "using `await web_search()`. Collect results in a list, filter by page_age, "
        "and summarize — all in one block. "
        "Variables do NOT persist between separate code execution blocks.\n\n"
        "EXAMPLE PATTERN:\n"
        "    import json\n"
        "    from datetime import datetime\n"
        "    cutoff = datetime(YYYY, MM, DD)\n"
        "    findings = []\n"
        "    for query in ['query 1', 'query 2', ...]:\n"
        "        results = json.loads(await web_search({'query': query}))\n"
        "        for r in results:\n"
        "            try:\n"
        "                if datetime.strptime(r['page_age'], '%B %d, %Y') <= cutoff:\n"
        "                    findings.append(r)\n"
        "            except ValueError:\n"
        "                pass  # skip unparseable dates\n"
        "    for f in findings:\n"
        "        print(f['title'], f['url'])\n"
        "        print(f['content'][:500])\n\n"
        f"Only include information from results published on or before {cutoff_date}. "
        "Do NOT reference post-cutoff results."
    )


def _claude_search_prompt(query, cutoff_date):
    """Build the user-facing prompt for search_claude from a raw query + cutoff date.

    search_exa handles date filtering via Exa's API parameters; this function does
    the equivalent for Claude by instructing it to target pre-cutoff sources and
    use code-based date filtering.
    """
    if cutoff_date:
        return (
            f"Find information relevant to answering the following question:\n{query}\n\n"
            f"Only use sources available before the cutoff date of {cutoff_date}. "
            "Your goal is to simulate the results of running a web search as it was back on "
            f"{cutoff_date}, so generate search queries that target pre-cutoff sources. "
        )
    return f"Find information relevant to answering the following question:\n{query}"




# ---------------------------------------------------------------------------
# search_claude_v1
# ---------------------------------------------------------------------------

def search_claude_v1(query, llm="claude-sonnet-4-6", max_tokens=4000,
                     max_searches=5, cutoff_date="", debug=False):
    """Call Claude with web_search_20260209 to find information relevant to query.

    Handles date filtering internally: when cutoff_date is provided, the user
    prompt instructs Claude to target pre-cutoff sources and a system prompt
    enforces single-block code filtering. Claude is allowed max_searches tool calls
    and uses its system prompt to rewrite the query into multiple variants internally.

    query: str or list[str]. If list, a warning is printed and the first element is used
    (v1 handles query rewriting internally via the system prompt).

    When debug=True, streams and prints every response block as it arrives,
    then prints usage stats. The same API call is used for both debug and
    non-debug paths, so the returned text is always consistent.

    Returns (summarized, input_tokens, output_tokens, nsearch).
    summarized is the joined text from all text blocks in the response.
    """
    if isinstance(query, list):
        print(f"[search_claude_v1] WARNING: list of queries passed to v1; using first query only")
        query = query[0]
    search_prompt = _claude_search_prompt(query, cutoff_date)

    if debug:
        print(f"\n[search_claude_v1] prompt:\n{textwrap.indent(search_prompt, '  ')}")

    client = anthropic.Anthropic()
    kwargs = dict(
        model=llm,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": search_prompt}],
    )
    if cutoff_date:
        kwargs["system"] = _cutoff_system_prompt(cutoff_date, max_searches)
    if max_searches > 0:
        kwargs["tools"] = [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches},
        ]

    printed = set()
    with client.messages.stream(**kwargs) as stream:
        if debug:
            for event in stream:
                etype = getattr(event, "type", "")
                idx   = getattr(event, "index", None)
                if etype == "content_block_start":
                    cb    = getattr(event, "content_block", None)
                    btype = getattr(cb, "type", "")
                    if btype == "server_tool_use":
                        name = getattr(cb, "name", "?")
                        print(f"\n  → [{name}] running...", flush=True)
                    elif btype == "web_search_tool_result":
                        print_web_search_result(idx, cb, cutoff_date=cutoff_date)
                        printed.add(idx)
                elif etype == "content_block_stop" and idx not in printed:
                    try:
                        block = stream.current_message_snapshot.content[idx]
                    except (IndexError, AttributeError):
                        continue
                    _print_block(idx, block, cutoff_date=cutoff_date)
                    printed.add(idx)
        message = stream.get_final_message()

    u = message.usage
    if debug:
        print(f"\n{'=' * 80}")
        print(f"Stop reason: {message.stop_reason}")
        print(f"Usage: input_tokens={u.input_tokens}  output_tokens={u.output_tokens}")
        print(f"Total content blocks: {len(message.content)}")
        print("=" * 80)

    nsearch = 0
    text_parts = []
    for block in message.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "server_tool_use" and getattr(block, "name", None) == "web_search":
            nsearch += 1

    summarized = "\n\n".join(text_parts)
    # v1 does everything in a single Claude call — no separate filter/summarize stages
    return summarized, summarized, "", u.input_tokens, u.output_tokens, 0, 0, 0, 0, nsearch


# ---------------------------------------------------------------------------
# search_claude_v2 — structured 3-stage pipeline
# ---------------------------------------------------------------------------
# search_claude — shim (defined after search_claude_v2; see bottom of file)

def _cutoff_system_prompt_v2(max_chars_per_doc=1000, cutoff_date=""):
    """Build the system prompt for v2 structured-output searches."""
    cutoff_line = ""
    if cutoff_date:
        cutoff_str = str(cutoff_date)[:10]
        cutoff_line = (
            f"\n\nWhen formulating your web search queries, include date-specific terms "
            f"(e.g., the year or month) to bias results toward the time period before {cutoff_str}. "
            "Do not mention the cutoff date or date filtering in your output."
        )
    return (
        "Output EVERY result in the EXACT format below. "
        "Output ONLY structured result blocks — no introductory text, notes, commentary, "
        "or analysis before, between, or after the results. "
        "Use plain text headers only (no markdown, no asterisks).\n\n"
        "RESULT N\n"
        "Title: <title>\n"
        "Date: <publication date, e.g. \"March 15, 2025\"; or \"unknown\">\n"
        "URL: <url>\n"
        f"Extract: <verbatim extract of most relevant content, ≤{max_chars_per_doc} chars>\n\n"
        "Replace N with the result number. Number results sequentially (RESULT 1, RESULT 2, ...). "
        + cutoff_line
    )


def _claude_search_prompt_v2(query, cutoff_date):
    """Build the user prompt for search_claude_v2 from a single query.

    cutoff_date is intentionally not mentioned here — date filtering is handled
    entirely by stages 2 and 3 (filter_results, summarize_results). Mentioning
    the cutoff date causes Claude to add meta-commentary about which results are
    pre- vs post-cutoff instead of outputting plain structured blocks.
    """
    return (
        f"Search the web for this query:\n{query}\n\n"
        "Output each result in the structured format as instructed."
    )


def _parse_v2_results(text, cutoff_date, debug=False):
    """Parse RESULT N blocks from search_claude_v2 stage-1 output.

    Extracts Title/Date/URL/Extract from each block, applies _age_flag for
    hard cutoff filtering, and reformats into _RESULTS_SEPARATOR-joined string
    compatible with filter_results.

    Returns (formatted_raw, n_kept, n_hard_filtered).
    """
    cutoff_str = str(cutoff_date)[:10] if cutoff_date else ""

    # Handle common Claude formatting variations:
    #   "RESULT 1", "RESULT 1:", "RESULT 1.", "**RESULT 1**", "**RESULT 1**:", "Result 1:"
    # \r? handles Windows line endings; \*{0,2} handles optional markdown bold.
    blocks = re.split(
        r"\n?\*{0,2}RESULT\s+\d+\*{0,2}[:\.]?\s*\r?\n",
        text.strip(),
        flags=re.IGNORECASE,
    )
    blocks = [b.strip() for b in blocks if b.strip()]
    if debug:
        print(f"\n[parse_v2_results] regex found {len(blocks)} raw block(s)")

    def get_line_field(block, name):
        m = re.search(rf"^{re.escape(name)}:\s*(.+)$", block, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else ""

    def get_extract(block):
        # Remove single-line fields so Extract: captures everything that follows
        remaining = block
        for field in ("Title", "Date", "URL"):
            remaining = re.sub(
                rf"^{field}:.*$\n?", "", remaining, count=1,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        m = re.search(r"^Extract:\s*(.*)", remaining, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        return m.group(1).strip() if m else ""

    context_lines = []
    n_kept = n_hard_filtered = 0
    for block in blocks:
        title   = get_line_field(block, "Title")
        date    = get_line_field(block, "Date")
        url     = get_line_field(block, "URL")
        extract = get_extract(block)

        if not title and not url:
            continue  # skip preamble / malformed blocks

        flag = _age_flag(date, cutoff_str)
        after_cutoff = cutoff_str and "AFTER CUTOFF" in flag

        if debug:
            status = "  [HARD-FILTERED]" if after_cutoff else ""
            print(f"\n  [parse] {title!r}{status}  date={date!r}{flag}")

        if after_cutoff:
            n_hard_filtered += 1
            continue

        n_kept += 1
        display_date = date if date and date.lower() != "unknown" else "unknown date"
        parts = [f"Search result {n_kept}: {title}. Published on {display_date}.{flag}", url]
        if extract:
            parts.append(f"Highlights:\n{extract}")
        context_lines.append("\n".join(parts))

    return _RESULTS_SEPARATOR.join(context_lines), n_kept, n_hard_filtered


def search_claude_v2(query, llm="claude-sonnet-4-6", max_tokens=4000,
                     max_searches=5, cutoff_date="",
                     max_chars_per_doc=1000,
                     llm_summarize=None, llm_query=None, question="",
                     summarization_context="", debug=False):
    """3-stage Claude web search: structured retrieval → date filter → summarize.

    query: str or list[str]. If str and max_searches > 1, generates max_searches-1
    paraphrases using llm_query (defaults to claude-haiku) and loops over them,
    making one API call per query with max_uses=1 each. If list, uses it directly.
    Results from all queries are concatenated before stages 2 and 3.

    Stage 1: For each query, Claude searches with web_search (max_uses=1) and
             outputs structured RESULT blocks (Title, Date, URL, Extract).
             Code execution is disabled; Claude does no date filtering itself.
    Stage 2: filter_results (LLM) applies KEEP/DROP per result based on date.
    Stage 3: summarize_results (LLM) produces the final context string.

    This pipeline mirrors search_exa: stage 1 is the retrieval step (analogous
    to the Exa API call), stages 2 and 3 are identical to search_exa's LLM passes.

    max_chars_per_doc: character limit for each Extract field (instructed to Claude).
    llm_summarize: model for stages 2 and 3 (defaults to llm).
    llm_query: model for query rewriting (defaults to claude-haiku-4-5-20251001).
    Returns (summarized, in_tok, out_tok, nsearch).
    """
    llm_sum = llm_summarize or llm

    # Resolve query list
    if isinstance(query, list):
        queries = query
    elif max_searches > 1:
        llm_rw = llm_query
        queries = rewrite_queries(query, max_searches, llm=llm_rw)
    else:
        queries = [query]

    if debug:
        print(f"\n{'─' * 40}  {'─' * 40}")
        print(f"\n[search_claude_v2] {len(queries)} quer{'y' if len(queries)==1 else 'ies'}")
        for qi, q in enumerate(queries, 1):
            print(f"  query {qi}: {q!r}")

    client = anthropic.Anthropic()
    system_prompt = _cutoff_system_prompt_v2(max_chars_per_doc, cutoff_date=cutoff_date)

    # Stage 1: one API call per query, concatenate raw text
    all_stage1_parts = []
    total_nsearch = 0
    in_tok_s1 = out_tok_s1 = 0

    for q_idx, q in enumerate(queries, 1):
        search_prompt = _claude_search_prompt_v2(q, cutoff_date)

        if debug:
            print(f"\n{'─' * 40}  {'─' * 40}")
            print(f"\n[search_claude_v2] query {q_idx}/{len(queries)}: {q!r}")
            print(f"  prompt: {textwrap.indent(search_prompt, '  ')}")

        kwargs = dict(
            model=llm,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": search_prompt}],
            tools=[
                {"type": "web_search_20260209", "name": "web_search",
                 "max_uses": 1, "allowed_callers": ["direct"]},
            ],
        )

        printed = set()
        with client.messages.stream(**kwargs) as stream:
            if debug:
                for event in stream:
                    etype = getattr(event, "type", "")
                    idx   = getattr(event, "index", None)
                    if etype == "content_block_start":
                        cb    = getattr(event, "content_block", None)
                        btype = getattr(cb, "type", "")
                        if btype == "server_tool_use":
                            name = getattr(cb, "name", "?")
                            print(f"\n  → [{name}] running...", flush=True)
                        elif btype == "web_search_tool_result":
                            print_web_search_result(idx, cb, cutoff_date=cutoff_date)
                            printed.add(idx)
                    elif etype == "content_block_stop" and idx not in printed:
                        try:
                            block = stream.current_message_snapshot.content[idx]
                        except (IndexError, AttributeError):
                            continue
                        _print_block(idx, block, cutoff_date=cutoff_date)
                        printed.add(idx)
            message = stream.get_final_message()

        u = message.usage
        in_tok_s1  += u.input_tokens
        out_tok_s1 += u.output_tokens

        text_parts = []
        for block in message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "server_tool_use" and getattr(block, "name", None) == "web_search":
                total_nsearch += 1

        q_text = "\n".join(text_parts)
        if debug:
            print(f"\n[search_claude_v2] query {q_idx} stage 1 complete")
            print(f"\n{'─' * 40} STAGE 1 RAW (query {q_idx}) {'─' * 40}")
            print(q_text or "(empty)")
        if q_text.strip():
            all_stage1_parts.append(q_text)

    stage1_text = "\n".join(all_stage1_parts)

    if debug:
        print(f"\n[search_claude_v2] Stage 1 complete: nsearch={total_nsearch}")

    # Parse structured blocks; hard-filter past-cutoff results
    raw_formatted, n_kept, n_hard_filtered = _parse_v2_results(
        stage1_text, cutoff_date, debug=debug)
    if debug:
        msg = f"parsed {n_kept} result(s)"
        if n_hard_filtered:
            msg += f", hard-filtered {n_hard_filtered}"
        print(f"\n[search_claude_v2] {msg}")

    if not raw_formatted:
        return "", "", "", in_tok_s1, out_tok_s1, 0, 0, 0, 0, total_nsearch

    # Stage 2: LLM-based date filter
    # do_search passes the filter model as llm, summarize model as llm_summarize
    filtered, filter_log, f_in_tok, f_out_tok = filter_results(
        raw_formatted, cutoff_date, llm=llm, debug=debug)
    if debug:
        print(f"\n{'─' * 40} FILTERED (LLM pass) {'─' * 40}")
        print(filtered or "(empty — all results dropped)")

    # Stage 3: summarize
    summarized, s_in_tok, s_out_tok = summarize_results(filtered, llm=llm_sum, question=question, summarization_context=summarization_context)
    if debug:
        print(f"\n{'─' * 40} SUMMARIZED {'─' * 40}")
        print(summarized or "(empty)")

    in_tok  = in_tok_s1  + f_in_tok  + s_in_tok
    out_tok = out_tok_s1 + f_out_tok + s_out_tok
    return summarized, raw_formatted, filter_log, in_tok, out_tok, f_in_tok, f_out_tok, s_in_tok, s_out_tok, total_nsearch


# ---------------------------------------------------------------------------
# search_claude — shim that defaults to search_claude_v2
# ---------------------------------------------------------------------------

def search_claude(query, llm="claude-sonnet-4-6", max_tokens=4000,
                  max_searches=5, cutoff_date="",
                  max_chars_per_doc=1000,
                  llm_summarize=None, llm_query=None, question="", debug=False):
    """Default Claude search: delegates to search_claude_v2.

    Use search_claude_v1 explicitly for the legacy code-execution pipeline.
    """
    return search_claude_v2(
        query, llm=llm, max_tokens=max_tokens, max_searches=max_searches,
        cutoff_date=cutoff_date, max_chars_per_doc=max_chars_per_doc,
        llm_summarize=llm_summarize, llm_query=llm_query, question=question, debug=debug,
    )


# ---------------------------------------------------------------------------
# Debug block printers
# ---------------------------------------------------------------------------

def _wrap(text, indent=4, width=100):
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)


def _is_encrypted_blob(v):
    """Heuristic: long string with no whitespace → likely base64/encrypted."""
    return isinstance(v, str) and len(v) > 80 and " " not in v[:80]


def print_text_block(idx, block):
    text      = getattr(block, "text", "") or ""
    citations = getattr(block, "citations", []) or []
    print(f"\n[{idx}] TEXT BLOCK")
    print(_wrap(text[:2000] + ("..." if len(text) > 2000 else "")))
    if citations:
        print(f"    -- {len(citations)} citation(s):")
        for c in citations:
            ctype      = getattr(c, "type", "?")
            cited_text = getattr(c, "cited_text", "") or ""
            url        = getattr(c, "url", "") or ""
            title      = getattr(c, "title", "") or ""
            page_age   = getattr(c, "page_age", "") or ""
            print(f"       [{ctype}] {title!r}  url={url}  page_age={page_age!r}")
            if cited_text:
                print(f"       cited_text: {cited_text[:200]!r}")


def print_server_tool_use(idx, block):
    name    = getattr(block, "name", "?")
    inp     = getattr(block, "input", {}) or {}
    tool_id = getattr(block, "id", "?")
    print(f"\n[{idx}] SERVER_TOOL_USE  name={name!r}  id={tool_id}")
    if isinstance(inp, dict):
        for k, v in inp.items():
            print(f"    {k}: {v!r}")
    else:
        print(f"    input: {inp!r}")


def print_web_search_result(idx, block, cutoff_date=None):
    tool_use_id = getattr(block, "tool_use_id", "?")
    content     = getattr(block, "content", []) or []
    print(f"\n[{idx}] WEB_SEARCH_TOOL_RESULT  tool_use_id={tool_use_id}")
    for j, r in enumerate(content):
        rtype    = getattr(r, "type", "?")
        url      = getattr(r, "url", "") or ""
        title    = getattr(r, "title", "") or ""
        page_age = getattr(r, "page_age", "") or ""
        enc      = getattr(r, "encrypted_content", None)
        flag     = _age_flag(page_age, cutoff_date)
        print(f"    result {j}: [{rtype}] {title!r}")
        print(f"      url:      {url}")
        print(f"      page_age: {page_age!r}{flag}")
        if enc is not None:
            print(f"      encrypted_content: <{len(enc)} chars, not client-decryptable>")
        print("\n")


def print_code_execution_tool_use(idx, block):
    name    = getattr(block, "name", "?")
    inp     = getattr(block, "input", {}) or {}
    tool_id = getattr(block, "id", "?")
    print(f"\n[{idx}] CODE_EXECUTION_TOOL_USE  name={name!r}  id={tool_id}")
    code = inp.get("code", "") if isinstance(inp, dict) else ""
    if code:
        print("    code:")
        for line in code.splitlines()[:30]:
            print(f"      {line}")


def print_code_execution_result(idx, block):
    tool_use_id = getattr(block, "tool_use_id", "?")
    btype       = getattr(block, "type", "?")
    print(f"\n[{idx}] CODE_EXECUTION_TOOL_RESULT  [{btype}]  tool_use_id={tool_use_id}")
    try:
        d = block.model_dump()
    except AttributeError:
        try:
            d = vars(block)
        except Exception:
            print("    (unable to inspect block)")
            return
    for k, v in d.items():
        if k in ("type", "tool_use_id"):
            continue
        if "encrypted" in k:
            print(f"    {k}: <{len(v) if isinstance(v, str) else '?'} chars, not shown>")
            continue
        if _is_encrypted_blob(v):
            print(f"    {k}: <{len(v)} chars, not client-decryptable>")
        elif isinstance(v, dict):
            safe = {
                dk: (f"<{len(dv) if isinstance(dv, str) else '?'} chars, not shown>"
                     if "encrypted" in dk or _is_encrypted_blob(dv) else dv)
                for dk, dv in v.items()
            }
            print(f"    {k}: {safe!r}")
        elif isinstance(v, list):
            if not v:
                print(f"    {k}: []")
            else:
                for i, item in enumerate(v):
                    if _is_encrypted_blob(item):
                        print(f"    {k}[{i}]: <{len(item)} chars, not client-decryptable>")
                    else:
                        print(f"    {k}[{i}]: {item!r}")
        elif v is not None and v != "":
            print(f"    {k}: {v!r}")


def print_unknown_block(idx, block):
    btype = getattr(block, "type", "?")
    print(f"\n[{idx}] UNKNOWN BLOCK  type={btype!r}")
    try:
        d = vars(block)
        print(f"    {json.dumps(d, default=str, indent=4)[:500]}")
    except Exception:
        print(f"    {block!r}")


def _print_block(idx, block, cutoff_date=None):
    btype = getattr(block, "type", None)
    if btype == "text":
        print_text_block(idx, block)
    elif btype == "server_tool_use":
        name = getattr(block, "name", "")
        if name == "code_execution":
            print_code_execution_tool_use(idx, block)
        else:
            print_server_tool_use(idx, block)
    elif btype == "web_search_tool_result":
        print_web_search_result(idx, block, cutoff_date=cutoff_date)
    elif btype and "code_execution" in btype:
        print_code_execution_result(idx, block)
    else:
        print_unknown_block(idx, block)
