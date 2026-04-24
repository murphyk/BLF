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
# num_retries=0: no litellm-level retries — we retry at higher levels.
# timeout is set dynamically via _get_timeout() based on model/reasoning config.
_CALL_DEFAULTS = dict(num_retries=0)

# ---------------------------------------------------------------------------
# Per-provider concurrency semaphores
# Governs individual litellm.completion calls, not entire agent runs.
# This maximises throughput: slots are freed while agents wait for tool results.
# ---------------------------------------------------------------------------

_DEFAULT_PROVIDER_LIMIT = 10
_PROVIDER_LIMITS = {
    "openrouter": 50,
    "gemini": 15,
    "anthropic": 10,
    "openai": 20,
}

_provider_sems: dict[str, threading.Semaphore] = defaultdict(
    lambda: threading.Semaphore(_DEFAULT_PROVIDER_LIMIT)
)
for _prov, _lim in _PROVIDER_LIMITS.items():
    _provider_sems[_prov] = threading.Semaphore(_lim)


def _get_sem(model: str) -> threading.Semaphore:
    provider = model.split("/")[0] if "/" in model else model
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


def chat(prompt, model, max_tokens=4000,
         system="", tools=None, reasoning_effort=None):
    """Simple LLM call (no tool-use loop).

    Returns (text, input_tokens, output_tokens, reasoning_tokens).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages,
                  timeout=_get_timeout(model, reasoning_effort), **_CALL_DEFAULTS)
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
                    deadline=None):
    """Agentic tool-use loop.

    Calls the LLM, dispatches any tool calls via dispatch_fn(name, args) -> str,
    feeds results back, and repeats until the model stops calling tools.

    prefix_messages: optional list of messages to inject after the user prompt
        (e.g. forced tool call + result pairs). The LLM sees these as if it had
        made those calls itself.

    Returns (text, input_tokens, output_tokens, tool_calls_log, reasoning_tokens).
    tool_calls_log is a list of {"name", "input", "result"} dicts.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if prefix_messages:
        messages.extend(prefix_messages)

    total_in = total_out = total_reasoning = 0
    text = ""
    tool_calls_log = []

    timed_out = False
    for _ in range(max_rounds):
        if deadline and time.time() > deadline:
            timed_out = True
            break
        kwargs = dict(model=model, max_tokens=max_tokens, messages=messages,
                      timeout=_get_timeout(model, reasoning_effort), **_CALL_DEFAULTS)
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
                                  timeout=30, **_CALL_DEFAULTS)
            with _get_sem(model):
                cleanup_resp = litellm.completion(**cleanup_kwargs)
            cleanup_text = cleanup_resp.choices[0].message.content or ""
            total_in += cleanup_resp.usage.prompt_tokens
            total_out += cleanup_resp.usage.completion_tokens
            if cleanup_text.strip():
                text = cleanup_text
        except Exception:
            pass  # cleanup failed — will fall back to 0.5

    return text, total_in, total_out, tool_calls_log, total_reasoning
