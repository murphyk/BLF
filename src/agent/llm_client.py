"""
llm_client.py — thin wrapper around litellm for provider-agnostic LLM calls.

All LLM interactions in the codebase go through this module.
Supports any provider that litellm supports (Anthropic, OpenAI, OpenRouter, etc.)
via the model string prefix (e.g. "anthropic/claude-sonnet-4-6", "openrouter/deepseek/deepseek-v3.2").

Tool schemas use the OpenAI format:
    {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
"""

import json
import threading
import time
import warnings
from collections import defaultdict
import litellm

# Suppress litellm's noisy logging and Pydantic serialization warnings
litellm.suppress_debug_info = True
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Per-call defaults.
# num_retries=3: retry transient errors (Connection reset by peer = ECONNRESET,
# 429s, 502s) at the litellm layer with exponential backoff. Without this,
# even short-lived connection blips kill tool calls (e.g., summarize-results)
# and burn agent-loop steps. Dispatch_tool catches those errors so the
# trial doesn't die, but every retry-eligible failure that we don't retry
# is a wasted step. timeout is set dynamically via _get_timeout().
_CALL_DEFAULTS = dict(num_retries=3)

# ---------------------------------------------------------------------------
# Per-provider concurrency semaphores
# Governs individual litellm.completion calls, not entire agent runs.
# This maximises throughput: slots are freed while agents wait for tool results.
# ---------------------------------------------------------------------------

_DEFAULT_PROVIDER_LIMIT = 10
_PROVIDER_LIMITS = {
    "openrouter": 50,
    # Gemini's per-minute RPM cap on OpenRouter is tight; with multiple
    # predict.py processes running in parallel, 15/process saturates.
    # 8/process keeps the 2-parallel case at 16 concurrent which empirically
    # avoids the "Rate limit exceeded: limit_rpm/google" errors.
    "gemini": 4,
    "anthropic": 10,
    "openai": 20,
    # Moonshot direct API (api.moonshot.ai). Required to leverage their
    # automatic context caching, which is *not* exposed via OpenRouter.
    # MOONSHOT_API_KEY env var must be set; conservative limit until
    # we measure their per-key RPM.
    "moonshot": 8,
}

_provider_sems: dict[str, threading.Semaphore] = defaultdict(
    lambda: threading.Semaphore(_DEFAULT_PROVIDER_LIMIT)
)
for _prov, _lim in _PROVIDER_LIMITS.items():
    _provider_sems[_prov] = threading.Semaphore(_lim)


def _get_sem(model: str) -> threading.Semaphore:
    """Return the semaphore for the model's actual rate-limited backend.

    Models routed through OpenRouter (e.g. 'openrouter/google/gemini-3-flash')
    are throttled by the BACKEND provider's per-minute limit
    (Google in this case), not by OpenRouter itself. Key on the second
    path component when present so we don't blow past Google's limit
    when running multiple Flash/Pro jobs in parallel.
    """
    parts = model.split("/")
    if not parts:
        return _provider_sems[model]
    provider = parts[0]
    if provider == "openrouter" and len(parts) >= 2:
        backend = parts[1].lower()
        # Map FB-style backend names to our semaphore keys.
        backend_key = {"google": "gemini"}.get(backend, backend)
        if backend_key in _PROVIDER_LIMITS:
            return _provider_sems[backend_key]
    return _provider_sems[provider]

def _get_timeout(model, reasoning_effort=None):
    """Return timeout in seconds, scaled for model and reasoning effort.

    Anthropic models are slower due to extended thinking overhead;
    high reasoning with large max_tokens can take 2-3 minutes.
    """
    is_anthropic = "anthropic" in model
    if reasoning_effort == "high":
        return 240 if is_anthropic else 180
    if is_anthropic:
        return 150
    return 90

# Models that don't accept reasoning_effort (they think by default or don't support it).
# For these, we strip the param before calling litellm to avoid UnsupportedParamsError.
_NO_REASONING_EFFORT = {
    "openrouter/qwen/qwen3-max-thinking",   # always thinks; litellm rejects the param
    "openrouter/moonshotai/kimi-k2-thinking",  # always thinks; litellm rejects the param
    "openrouter/moonshotai/kimi-k2.5",          # has reasoning but not via reasoning_effort
    "moonshot/kimi-k2.5",                        # direct API, same hybrid model
    "moonshot/kimi-k2.6",                        # direct API
    "openrouter/x-ai/grok-4-fast",           # uses reasoning.enabled, not reasoning_effort
    "openrouter/x-ai/grok-4.1-fast",
    "openrouter/x-ai/grok-4.20-beta-20260309",
    "openrouter/x-ai/grok-4",
    "openrouter/x-ai/grok-3",
    "openrouter/x-ai/grok-3-beta",
}


def _reasoning_tokens(usage):
    """Extract reasoning/thinking tokens from litellm usage object."""
    details = getattr(usage, "completion_tokens_details", None)
    if details:
        return getattr(details, "reasoning_tokens", 0) or 0
    return 0


def cached_tokens(usage) -> int:
    """Extract cache-hit token count from a litellm usage object.

    OpenAI-compat (Gemini, GPT, Kimi):
        usage.prompt_tokens_details.cached_tokens
    Anthropic native (also normalised by litellm):
        usage.cache_read_input_tokens
    Returns 0 when no cache hit info is reported.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        n = getattr(details, "cached_tokens", 0) or 0
        if n:
            return int(n)
    return int(getattr(usage, "cache_read_input_tokens", 0) or 0)


def cache_creation_tokens(usage) -> int:
    """Anthropic-only: tokens written into cache on this call (priced 1.25x).
    Returns 0 if not reported (Gemini/OpenAI don't expose this)."""
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        n = getattr(details, "cache_creation_tokens", 0) or 0
        if n:
            return int(n)
    return int(getattr(usage, "cache_creation_input_tokens", 0) or 0)


def _is_anthropic(model: str) -> bool:
    """Anthropic models — direct or via OpenRouter."""
    m = model.lower()
    return m.startswith("anthropic/") or "/anthropic/" in m


def _is_moonshot_direct(model: str) -> bool:
    """Moonshot direct API (api.moonshot.ai), used to leverage their
    automatic context caching that OpenRouter does not expose."""
    return model.lower().startswith("moonshot/")


def _moonshot_kwargs(model: str) -> dict:
    """Extra kwargs to inject for Moonshot direct API calls.

    litellm's default Moonshot base URL is the China endpoint
    (api.moonshot.cn). For the international endpoint, set
    MOONSHOT_API_BASE=https://api.moonshot.ai/v1 in your .env.
    """
    import os
    if not _is_moonshot_direct(model):
        return {}
    out = {}
    base = os.environ.get("MOONSHOT_API_BASE")
    if base:
        out["api_base"] = base
    if "MOONSHOT_API_KEY" in os.environ:
        out["api_key"] = os.environ["MOONSHOT_API_KEY"]
    return out


def apply_anthropic_cache(messages, tools, model: str, enabled: bool = True):
    """Wrap system + tool schemas + initial user prompt with cache_control
    blocks so Anthropic caches the stable prefix across an agent loop.

    Anthropic supports up to 4 cache breakpoints. We use 3 stable ones:
      1. system message
      2. last tool schema (caches all preceding tool definitions)
      3. initial user prompt
    Non-Anthropic models or enabled=False: no-op (returns inputs unchanged).
    """
    if not enabled or not _is_anthropic(model):
        return messages, tools

    EPHEMERAL = {"type": "ephemeral"}

    def _wrap_text(content):
        if isinstance(content, str):
            return [{"type": "text", "text": content,
                     "cache_control": EPHEMERAL}]
        if isinstance(content, list):
            out = list(content)
            for i in range(len(out) - 1, -1, -1):
                if isinstance(out[i], dict) and out[i].get("type") == "text":
                    out[i] = {**out[i], "cache_control": EPHEMERAL}
                    break
            return out
        return content

    new_messages = []
    seen_user = False
    for m in messages:
        role = m.get("role")
        if role == "system":
            new_messages.append({**m, "content": _wrap_text(m.get("content", ""))})
        elif role == "user" and not seen_user:
            new_messages.append({**m, "content": _wrap_text(m.get("content", ""))})
            seen_user = True
        else:
            new_messages.append(m)

    new_tools = tools
    if tools:
        new_tools = list(tools[:-1]) + [{**tools[-1],
                                         "cache_control": EPHEMERAL}]

    return new_messages, new_tools


def chat(prompt, model, max_tokens=4000,
         system="", tools=None, reasoning_effort=None,
         cache_anthropic=True):
    """Simple LLM call (no tool-use loop).

    Returns (text, input_tokens, output_tokens, reasoning_tokens).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    messages, tools = apply_anthropic_cache(messages, tools, model,
                                            enabled=cache_anthropic)

    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages,
                  timeout=_get_timeout(model, reasoning_effort),
                  **_moonshot_kwargs(model), **_CALL_DEFAULTS)
    if tools:
        kwargs["tools"] = tools
    if reasoning_effort and reasoning_effort != "none" and model not in _NO_REASONING_EFFORT:
        kwargs["reasoning_effort"] = reasoning_effort

    with _get_sem(model):
        response = litellm.completion(**kwargs)
    choice = response.choices[0]
    text = choice.message.content or ""
    usage = response.usage
    return text, usage.prompt_tokens, usage.completion_tokens, _reasoning_tokens(usage)


def chat_with_tools(prompt, model, max_tokens=4000,
                    system="", tools=None, dispatch_fn=None, max_rounds=10,
                    reasoning_effort=None, prefix_messages=None,
                    deadline=None, cache_anthropic=True):
    """Agentic tool-use loop.

    Calls the LLM, dispatches any tool calls via dispatch_fn(name, args) -> str,
    feeds results back, and repeats until the model stops calling tools.

    prefix_messages: optional list of messages to inject after the user prompt
        (e.g. forced tool call + result pairs). The LLM sees these as if it had
        made those calls itself.

    Returns (text, input_tokens, output_tokens, tool_calls_log,
             reasoning_tokens, cached_in_tokens).
    tool_calls_log is a list of {"name", "input", "result"} dicts.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if prefix_messages:
        messages.extend(prefix_messages)

    messages, tools = apply_anthropic_cache(messages, tools, model,
                                            enabled=cache_anthropic)

    total_in = total_out = total_reasoning = total_cached = 0
    text = ""
    tool_calls_log = []

    timed_out = False
    for _ in range(max_rounds):
        if deadline and time.time() > deadline:
            timed_out = True
            break
        kwargs = dict(model=model, max_tokens=max_tokens, messages=messages,
                      timeout=_get_timeout(model, reasoning_effort),
                      **_moonshot_kwargs(model), **_CALL_DEFAULTS)
        if tools:
            kwargs["tools"] = tools
        if reasoning_effort and reasoning_effort != "none" and model not in _NO_REASONING_EFFORT:
            kwargs["reasoning_effort"] = reasoning_effort

        with _get_sem(model):
            response = litellm.completion(**kwargs)
        choice = response.choices[0]
        usage = response.usage
        total_in += usage.prompt_tokens
        total_out += usage.completion_tokens
        total_reasoning += _reasoning_tokens(usage)
        total_cached += cached_tokens(usage)

        text = choice.message.content or ""
        thinking = getattr(choice.message, "reasoning_content", None) or ""
        tool_calls = choice.message.tool_calls

        if not tool_calls or choice.finish_reason == "stop":
            break

        # Capture intermediate thinking (extended thinking / chain-of-thought)
        if thinking:
            tool_calls_log.append({"type": "thinking", "text": thinking})
        # Capture intermediate visible reasoning before tool calls
        if text:
            tool_calls_log.append({"type": "reasoning", "text": text})

        # Append assistant message (with tool_calls) to conversation
        messages.append(choice.message)

        # Dispatch each tool call and append results
        for tc in tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            result = dispatch_fn(fn_name, fn_args) if dispatch_fn else ""
            tool_calls_log.append({"type": "tool_call", "name": fn_name, "input": fn_args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # Check deadline after tool dispatch (tool calls can be slow)
        if deadline and time.time() > deadline:
            timed_out = True
            break

    # If timed out mid-conversation, do a final cleanup call (no tools)
    # to get a well-formed probability= answer from partial work.
    if timed_out and messages:
        tool_calls_log.append({"type": "timeout", "text": "Deadline reached — requesting final answer."})
        messages.append({"role": "user",
                         "content": "Time is up. Based on all the information above, "
                                    "output your final answer now as probability=X.XX "
                                    "(a single number between 0 and 1). No explanation needed."})
        try:
            cleanup_kwargs = dict(model=model, max_tokens=200, messages=messages,
                                  timeout=30,
                                  **_moonshot_kwargs(model), **_CALL_DEFAULTS)
            with _get_sem(model):
                cleanup_resp = litellm.completion(**cleanup_kwargs)
            cleanup_text = cleanup_resp.choices[0].message.content or ""
            total_in += cleanup_resp.usage.prompt_tokens
            total_out += cleanup_resp.usage.completion_tokens
            total_cached += cached_tokens(cleanup_resp.usage)
            if cleanup_text.strip():
                text = cleanup_text
        except Exception:
            pass  # cleanup failed — will fall back to 0.5

    return text, total_in, total_out, tool_calls_log, total_reasoning, total_cached
