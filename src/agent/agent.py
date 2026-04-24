"""agent.py — Core agentic forecasting loop."""

import json
import re
import time
import litellm

from agent.belief_state import BeliefState, compact_belief
from config.config import AgentConfig
from agent.prompts import get_system_prompt, format_question_prompt
from agent.tools import get_tool_schemas, dispatch_tool, parse_belief_update

# Suppress litellm noise
litellm.suppress_debug_info = True


def _question_stem(question: dict) -> str:
    """Build a filesystem-safe stem from the question id."""
    qid = question.get("id", "unknown")
    return re.sub(r'[/\\:]', '_', str(qid))


def _get_cutoff_date(question: dict) -> str:
    """Get the knowledge cutoff date from the question's forecast_due_date."""
    fdd = question.get("forecast_due_date", "")
    return fdd[:10] if fdd else ""


def run_agent(question: dict, config: AgentConfig, output_dir: str,
              verbose: bool = False) -> dict:
    """Run the agentic forecasting loop on a single question.

    Returns a dict with: p, reasoning, belief_history, tool_log, tokens, etc.
    """
    if config.clairvoyant:
        from datetime import date
        cutoff_date = str(date.today())
    else:
        cutoff_date = _get_cutoff_date(question)
    qid = question.get("id", "unknown")
    stem = _question_stem(question)
    source = question.get("source", "")
    prefix = f"  [{config.name} on {source}/{qid}]" if verbose else ""

    state = BeliefState(p=0.5)
    search_cache = {}  # search_idx -> list of raw result strings
    tool_log = []
    belief_history = [state.to_dict()]
    total_in_tok = 0
    total_out_tok = 0
    t0 = time.time()

    deadline = t0 + config.question_timeout

    # Build system prompt (clairvoyant uses live mode — no restrictions)
    live_mode = config.clairvoyant or not config.backtesting
    if config.halawi_prompt:
        from agent.prompts import HALAWI_SYSTEM, format_halawi_prompt
        system = HALAWI_SYSTEM
        user_prompt = format_halawi_prompt(question, cutoff_date,
                                           show_crowd=config.show_crowd,
                                           show_prior=config.show_prior)
    else:
        system = get_system_prompt(config.max_steps, live=live_mode,
                                   source=source, nobelief=config.nobelief,
                                   use_tools=config.use_tools,
                                   use_search=(config.search_engine != "none"))

        # Build initial user prompt
        backtesting_prompt = config.backtesting and not config.clairvoyant
        user_prompt = format_question_prompt(question, cutoff_date,
                                             show_crowd=config.show_crowd,
                                             show_prior=config.show_prior,
                                             use_tools=config.use_tools,
                                             backtesting=backtesting_prompt,
                                             nobelief=config.nobelief)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    submitted = False
    _force_submit = (config.max_steps == 1)  # single-step mode: force submit immediately

    for step in range(config.max_steps):
        state.step = step + 1

        if time.time() > deadline:
            tool_log.append({"step": step + 1, "type": "timeout"})
            break

        # Get available tools (includes source-specific tools if use_tools=True)
        tools = get_tool_schemas(config, source=source, question=question)
        # On the last step, only offer submit so the agent must submit
        if _force_submit:
            tools = [t for t in tools if t["function"]["name"] == "submit"]

        # Call LLM (with retry on transient errors)
        try:
            kwargs = dict(
                model=config.llm,
                messages=messages,
                tools=tools,
                max_tokens=config.max_tokens,
                timeout=min(120, max(10, deadline - time.time())),
                num_retries=2,
            )
            # Force the model to call submit on the last step
            if _force_submit:
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": "submit"},
                }
            if config.reasoning_effort:
                # Only pass reasoning_effort for providers that support it
                _RE_PROVIDERS = ("google/", "anthropic/")
                if any(p in config.llm for p in _RE_PROVIDERS):
                    kwargs["reasoning_effort"] = config.reasoning_effort

            response = litellm.completion(**kwargs)
        except Exception as e:
            tool_log.append({"step": step + 1, "type": "error", "error": str(e)})
            # Extract short error description (skip huge tracebacks)
            err_name = type(e).__name__
            err_msg = str(e).split('\n')[0][:120]
            print(f"{prefix} step {step + 1}: LLM error ({err_name}: {err_msg})")
            break

        choice = response.choices[0]
        usage = response.usage
        total_in_tok += usage.prompt_tokens
        total_out_tok += usage.completion_tokens

        text = choice.message.content or ""
        thinking = getattr(choice.message, "reasoning_content", None) or ""
        tool_calls = choice.message.tool_calls

        # No tool call = done
        if not tool_calls:
            if verbose:
                print(f"{prefix} step {step + 1}: no tool call, text={text[:80]}")
            tool_log.append({"step": step + 1, "type": "no_tool_call", "text": text})
            break

        # Process first tool call
        tc = tool_calls[0]
        fn_name = tc.function.name
        try:
            fn_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            fn_args = {}

        # Reject hallucinated tool calls (model called a tool not in the schema)
        available_names = {t["function"]["name"] for t in tools}
        if fn_name not in available_names:
            if verbose:
                print(f"{prefix} step {step + 1}: rejected hallucinated tool {fn_name}, "
                      f"available: {available_names}")
            messages.append(choice.message)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"Error: '{fn_name}' is not available. "
                           f"Available tools: {sorted(available_names)}. "
                           f"Please call submit with your probability estimate.",
            })
            tool_log.append({"step": step + 1, "type": "rejected_tool",
                            "tool": fn_name, "available": sorted(available_names)})
            continue

        if verbose:
            args_str = json.dumps({k: v for k, v in fn_args.items() if k != 'updated_belief'}, default=str)[:80]
            print(f"{prefix} step {step + 1}: {fn_name}({args_str})")

        # Dispatch
        try:
            result_text, new_state, meta = dispatch_tool(
                fn_name, fn_args, state, config,
                search_cache, output_dir, stem,
                question, cutoff_date,
                deadline=deadline,
            )
        except Exception as e:
            result_text = f"Tool error: {e}"
            new_state = state
            meta = {"tool": fn_name, "error": str(e)}
            err_name = type(e).__name__
            err_msg = str(e).split('\n')[0][:120]
            print(f"{prefix} step {step + 1}: tool error ({err_name}: {err_msg})")

        state = new_state
        if not config.nobelief:
            # Auto-compact belief state when evidence lists get long
            state = compact_belief(state, config)
        belief_history.append(state.to_dict())

        # Store tool call args (excluding updated_belief which is bulky)
        args_clean = {k: v for k, v in fn_args.items() if k != "updated_belief"}
        log_entry = {
            "step": step + 1,
            "type": "tool_call",
            "args": args_clean,
            **meta,
            "belief_p": state.p,
        }
        if text:
            log_entry["reasoning_text"] = text
        if thinking:
            log_entry["thinking"] = thinking
        tool_log.append(log_entry)

        # Append assistant message (with tool_calls) to conversation
        messages.append(choice.message)
        # Append tool result
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result_text,
        })

        if fn_name == "submit":
            submitted = True
            break

        # Post-tool-call context injection
        remaining = config.max_steps - (step + 1)
        if not config.nobelief:
            messages.append({
                "role": "user",
                "content": f"[Belief state updated]\n{state.to_prompt_str(config.max_steps)}",
            })
        else:
            messages.append({
                "role": "user",
                "content": f"[{remaining} steps remaining.]",
            })

        # Force submit on the next (last) step
        if remaining == 1:
            _force_submit = True

    # Extract final forecast(s) from submit tool log
    final_ps = None  # multi-resolution list, or None
    final_p = max(0.05, min(0.95, state.p))
    reasoning = ""
    for entry in reversed(tool_log):
        if entry.get("tool") == "submit":
            if "final_ps" in entry:
                final_ps = entry["final_ps"]
                final_p = final_ps[0]
            elif "final_p" in entry:
                final_p = entry["final_p"]
            reasoning = entry.get("reasoning", "")
            break
    if not reasoning:
        reasoning = (
            f"Agent {'submitted' if submitted else 'timed out / ran out of steps'} "
            f"with p={final_p:.3f}. "
            f"Evidence for: {state.evidence_for}. "
            f"Evidence against: {state.evidence_against}."
        )

    # For multi-resolution: if model didn't submit a list but question has
    # multiple dates, duplicate the single forecast
    rdates = question.get("resolution_dates", [])
    if len(rdates) > 1 and final_ps is None:
        final_ps = [final_p] * len(rdates)

    result = {
        "id": qid,
        "source": question.get("source", "metaculus"),
        "question": question.get("question", ""),
        "background": question.get("background", ""),
        "resolution_criteria": question.get("resolution_criteria", ""),
        "forecast_due_date": question.get("forecast_due_date", ""),
        "market_value": question.get("market_value", ""),
        "market_date": question.get("market_date", ""),
        "market_value_explanation": question.get("market_value_explanation", ""),
        "forecast": final_p,
        "reasoning": reasoning,
        "resolution_date": question.get("resolution_date", ""),
        "resolution_dates": rdates,
        "resolved_to": question.get("resolved_to"),
        "system_prompt": system,
        "question_prompt": user_prompt,
        "belief_history": belief_history,
        "tool_log": tool_log,
        "n_steps": state.step,
        "submitted": submitted,
        "tokens_in": total_in_tok,
        "tokens_out": total_out_tok,
        "elapsed_seconds": round(time.time() - t0, 1),
        "config": {
            **config.to_dict(),
        },
    }
    if final_ps is not None:
        result["forecasts"] = final_ps

    return result


# ---------------------------------------------------------------------------
# Batch (non-agentic) mode
# ---------------------------------------------------------------------------

def run_batch_agent(question: dict, config: AgentConfig, output_dir: str,
                    verbose: bool = False) -> dict:
    """Non-agentic batch search: generate N queries, search all at once,
    summarize results, then reason and submit.

    This emulates a non-agentic pipeline:
    1. LLM generates N diverse search queries
    2. All queries are executed in parallel
    3. Results are summarized
    4. LLM reasons over summaries and produces forecast

    config.batch_queries controls N (default 5).
    Always returns a result dict (falls back to p=0.5 on any error).
    """
    t0 = time.time()
    try:
        return _run_batch_agent_inner(question, config, output_dir, verbose)
    except Exception as e:
        qid = question.get("id", "unknown")
        source = question.get("source", "")
        if verbose:
            print(f"  [{config.name} on {source}/{qid}] batch agent failed: {e}")
        _U = type('U', (), {'prompt_tokens': 0, 'completion_tokens': 0})()
        return _build_batch_result(question, config, 0.5,
                                    f"Batch agent failed: {e}",
                                    [], t0, _U, _U, 0, 0, 0, 0, [],
                                    submitted=False)


def _run_batch_agent_inner(question: dict, config: AgentConfig, output_dir: str,
                           verbose: bool = False) -> dict:
    """Inner implementation of run_batch_agent."""
    from search import do_search, summarize_results, _SEARCH_LLM, _RESULTS_SEPARATOR

    if config.clairvoyant:
        from datetime import date
        cutoff_date = str(date.today())
    else:
        cutoff_date = _get_cutoff_date(question)
    qid = question.get("id", "unknown")
    stem = _question_stem(question)
    source = question.get("source", "")
    prefix = f"  [{config.name} on {source}/{qid}]" if verbose else ""
    N = config.batch_queries or 5
    t0 = time.time()
    deadline = t0 + config.question_timeout

    # Step 1: Ask LLM to generate N diverse search queries
    backtesting_prompt = config.backtesting and not config.clairvoyant
    user_prompt = format_question_prompt(question, cutoff_date,
                                         show_crowd=config.show_crowd,
                                         use_tools=False,
                                         backtesting=backtesting_prompt)

    query_gen_prompt = (
        f"You are a forecasting researcher. Given the question below, "
        f"generate exactly {N} diverse web search queries that would help "
        f"you forecast the answer. The queries should cover different aspects "
        f"of the question: background context, recent developments, expert "
        f"opinions, relevant data, and counterarguments.\n\n"
        f"Return ONLY a JSON array of {N} query strings, nothing else.\n\n"
        f"Question:\n{user_prompt}"
    )

    try:
        kwargs = dict(
            model=config.llm,
            messages=[{"role": "user", "content": query_gen_prompt}],
            max_tokens=1000,
            timeout=60,
        )
        if config.reasoning_effort:
            _RE_PROVIDERS = ("google/", "anthropic/")
            if any(p in config.llm for p in _RE_PROVIDERS):
                kwargs["reasoning_effort"] = config.reasoning_effort
        resp = litellm.completion(**kwargs)
        query_text = resp.choices[0].message.content or "[]"
        usage1 = resp.usage
    except Exception as e:
        if verbose:
            print(f"{prefix} batch query generation failed: {e}")
        # Fallback: use the question text itself
        query_text = json.dumps([question.get("question", "")])
        usage1 = type('U', (), {'prompt_tokens': 0, 'completion_tokens': 0})()

    # Parse queries
    try:
        # Strip markdown fences if present
        clean = re.sub(r'```json\s*', '', query_text)
        clean = re.sub(r'```\s*', '', clean)
        queries = json.loads(clean)
        if not isinstance(queries, list):
            queries = [str(queries)]
    except json.JSONDecodeError:
        queries = [question.get("question", "")]

    queries = queries[:N]
    if verbose:
        print(f"{prefix} batch: generated {len(queries)} queries")

    # Step 2: Execute all searches
    import os
    search_dir = os.path.join(output_dir, stem, "searches")
    os.makedirs(search_dir, exist_ok=True)

    all_results = []
    total_search_in = 0
    total_search_out = 0
    for qi, query in enumerate(queries):
        sr = do_search(
            search_prompt=query,
            search_engine=config.search_engine,
            cutoff_date=cutoff_date,
            max_searches=1,
            max_results_per_search=config.max_results_per_search,
            extra_snippets=True,
            llm_summarize="none",
            question=question.get("question", ""),
            deadline=deadline,
        )
        filtered = sr.summarized or sr.raw
        if filtered:
            results = [r.strip() for r in filtered.split(_RESULTS_SEPARATOR) if r.strip()]
            for j, text in enumerate(results):
                fname = f"search_0_result_{len(all_results)}.md"
                with open(os.path.join(search_dir, fname), "w") as f:
                    f.write(text)
                all_results.append(text)
        total_search_in += sr.in_tok
        total_search_out += sr.out_tok
        if verbose:
            n_res = len(results) if filtered else 0
            print(f"{prefix} batch query {qi}: '{query[:60]}' -> {n_res} results")

    # Step 2b: Call source-specific tool if available (e.g. fetch_ts_fred)
    # Extract tool args from question metadata, then dispatch.
    # Skipped gracefully if args can't be determined or tool fails.
    tool_result_text = ""
    tool_in = tool_out = 0
    if config.use_tools and source:
        try:
            from agent.source_tools import get_source_tools
            source_tools = get_source_tools(source)
            if source_tools:
                tool_name = source_tools[0]["function"]["name"]
                props = source_tools[0]["function"]["parameters"].get("properties", {})
                bg = question.get("background", "") + " " + question.get("question", "")
                rc = question.get("resolution_criteria", "")

                tool_args = {}
                if "series_id" in props:
                    for pattern in [r'series/(\w+)', r'ticker[:\s]+(\w+)',
                                    r'symbol[:\s]+(\w+)',
                                    r"Will (\w+)'s market",
                                    r"Will (\w+) (?:stock|share|price)"]:
                        m = re.search(pattern, bg, re.IGNORECASE)
                        if m:
                            tool_args["series_id"] = m.group(1)
                            break
                    # Fallback: use the question ID prefix (e.g. "MU" from "MU_2025-10-26")
                    if "series_id" not in tool_args:
                        qid_base = question.get("id", "").split("_")[0]
                        if qid_base and qid_base.isalnum():
                            tool_args["series_id"] = qid_base
                if "url" in props:
                    for text in [bg, rc]:
                        m = re.search(r'https?://\S+', text)
                        if m:
                            tool_args["url"] = m.group(0).rstrip('.,;)')
                            break

                if tool_args:
                    state_tmp = BeliefState(p=0.5)
                    result, _, meta = dispatch_tool(
                        tool_name, tool_args, state_tmp, config,
                        {}, output_dir, stem, question, cutoff_date,
                        deadline=deadline)
                    tool_result_text = f"\n\n## Source Data ({tool_name})\n{result}"
                    tool_in = meta.get("tokens_in", 0)
                    tool_out = meta.get("tokens_out", 0)
                    if verbose:
                        print(f"{prefix} batch tool: {tool_name}({tool_args}) "
                              f"-> {len(result)} chars")
        except Exception as e:
            if verbose:
                print(f"{prefix} batch tool failed: {e}")

    # Step 3: Summarize all results
    combined_text = _RESULTS_SEPARATOR.join(all_results)
    if combined_text.strip():
        summary, sum_in, sum_out = summarize_results(
            combined_text,
            llm=_SEARCH_LLM,
            question=question.get("question", ""),
            summarization_context=question.get("resolution_criteria", ""),
        )
    else:
        summary = "No search results found."
        sum_in = sum_out = 0

    # Save summary
    with open(os.path.join(search_dir, "batch_summary.md"), "w") as f:
        f.write(f"Queries: {json.dumps(queries, indent=2)}\n\n"
                f"Results: {len(all_results)}\n\n{summary}")

    # Step 4: Reason and submit
    live_mode = config.clairvoyant or not config.backtesting
    system = get_system_prompt(1, live=live_mode, source=source,
                               use_tools=False, use_search=False)

    reason_prompt = (
        f"{user_prompt}\n\n"
        f"## Search Results Summary\n"
        f"You searched for {len(queries)} queries and found {len(all_results)} results. "
        f"Here is the summarized evidence:\n\n{summary}"
        f"{tool_result_text}\n\n"
        f"Based on this evidence, provide your forecast by calling the submit tool."
    )

    tools = get_tool_schemas(config, source=source, question=question)
    # Only keep the submit tool
    tools = [t for t in tools if t["function"]["name"] == "submit"]

    try:
        kwargs = dict(
            model=config.llm,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": reason_prompt},
            ],
            tools=tools,
            max_tokens=config.max_tokens,
            timeout=120,
        )
        if config.reasoning_effort:
            _RE_PROVIDERS = ("google/", "anthropic/")
            if any(p in config.llm for p in _RE_PROVIDERS):
                kwargs["reasoning_effort"] = config.reasoning_effort
        resp = litellm.completion(**kwargs)
        usage2 = resp.usage
    except Exception as e:
        if verbose:
            print(f"{prefix} batch reasoning failed: {e}")
        usage2 = type('U', (), {'prompt_tokens': 0, 'completion_tokens': 0})()
        # Return default
        return _build_batch_result(question, config, 0.5, "Batch reasoning failed",
                                    [], t0, usage1, usage2, total_search_in,
                                    total_search_out, sum_in, sum_out, queries)

    choice = resp.choices[0]
    text = choice.message.content or ""
    tool_calls = choice.message.tool_calls

    # Extract forecast from submit tool call
    final_p = 0.5
    reasoning = text
    submitted = False
    if tool_calls:
        tc = tool_calls[0]
        try:
            fn_args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            fn_args = {}
        if tc.function.name == "submit":
            p = fn_args.get("probability")
            if isinstance(p, list):
                final_p = max(0.05, min(0.95, p[0]))
            elif p is not None:
                final_p = max(0.05, min(0.95, float(p)))
            reasoning = fn_args.get("reasoning", text)
            submitted = True

    if verbose:
        print(f"{prefix} batch: submitted p={final_p:.3f}")

    return _build_batch_result(question, config, final_p, reasoning,
                                queries, t0, usage1, usage2, total_search_in,
                                total_search_out, sum_in, sum_out, queries,
                                submitted=submitted)


def _build_batch_result(question, config, final_p, reasoning, queries,
                         t0, usage1, usage2, search_in, search_out,
                         sum_in, sum_out, query_list, submitted=False):
    qid = question.get("id", "unknown")
    rdates = question.get("resolution_dates", [])
    final_ps = [final_p] * len(rdates) if len(rdates) > 1 else None

    total_in = (getattr(usage1, 'prompt_tokens', 0)
                + getattr(usage2, 'prompt_tokens', 0)
                + search_in + sum_in)
    total_out = (getattr(usage1, 'completion_tokens', 0)
                 + getattr(usage2, 'completion_tokens', 0)
                 + search_out + sum_out)

    result = {
        "id": qid,
        "source": question.get("source", ""),
        "question": question.get("question", ""),
        "forecast_due_date": question.get("forecast_due_date", ""),
        "forecast": final_p,
        "reasoning": reasoning,
        "resolution_date": question.get("resolution_date", ""),
        "resolution_dates": rdates,
        "resolved_to": question.get("resolved_to"),
        "batch_queries": query_list,
        "n_steps": 2,
        "submitted": submitted,
        "mode": "batch",
        "tokens_in": total_in,
        "tokens_out": total_out,
        "elapsed_seconds": round(time.time() - t0, 1),
        "config": config.to_dict(),
    }
    if final_ps is not None:
        result["forecasts"] = final_ps
    return result
