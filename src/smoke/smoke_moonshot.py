"""smoke_moonshot.py — direct-API + thinking + KV cache smoke test.

Verifies three things for the moonshot/kimi-k2.5 alias:

1. Round-trip works against api.moonshot.ai (uses MOONSHOT_API_KEY
   from .env; litellm defaults the base URL to https://api.moonshot.ai/v1).
2. Thinking mode is on — checks that the response carries a non-empty
   `reasoning_content` block. K2.5 is hybrid; we explicitly request
   thinking via extra_body={"thinking": {"type": "enabled"}}.
3. Automatic KV caching kicks in — issues two calls with the same long
   prefix and inspects usage.cached_tokens on the second call.

Run:
    python -m src.smoke.smoke_moonshot
"""

import os
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

import litellm
from src.agent.llm_client import cached_tokens, _reasoning_tokens

MODEL = "moonshot/kimi-k2.5"


def _has_key():
    if not os.environ.get("MOONSHOT_API_KEY"):
        raise RuntimeError("MOONSHOT_API_KEY not set in environment / .env")


def _call(messages, *, thinking=True, max_tokens=400):
    kwargs = dict(model=MODEL, messages=messages, max_tokens=max_tokens)
    if thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    return litellm.completion(**kwargs)


def test_basic_call():
    print("\n=== TEST 1: basic round trip ===")
    resp = _call([{"role": "user", "content": "Say 'hello, moonshot' and nothing else."}],
                 thinking=False, max_tokens=200)
    msg = resp.choices[0].message
    text = msg.content or ""
    rc = getattr(msg, "reasoning_content", None) or ""
    rt = _reasoning_tokens(resp.usage)
    print(f"  text: {text!r}")
    print(f"  reasoning_content: {rc[:80]!r}{'...' if len(rc)>80 else ''}")
    print(f"  usage: prompt={resp.usage.prompt_tokens} "
          f"completion={resp.usage.completion_tokens} reasoning={rt}")
    assert text.strip() or rc.strip(), "no text and no reasoning_content"
    print("  PASS")


def test_thinking_enabled():
    print("\n=== TEST 2: thinking mode ===")
    resp = _call(
        [{"role": "user",
          "content": "What is 17 * 23? Show your reasoning briefly."}],
        thinking=True, max_tokens=600)
    msg = resp.choices[0].message
    text = msg.content or ""
    rc = getattr(msg, "reasoning_content", None) or ""
    rt = _reasoning_tokens(resp.usage)
    print(f"  content      : {text[:120]}{'...' if len(text)>120 else ''}")
    print(f"  reasoning_content (first 200 chars): {rc[:200]}{'...' if len(rc)>200 else ''}")
    print(f"  reasoning_tokens (usage): {rt}")
    if rc or rt > 0:
        print("  PASS — thinking observed")
    else:
        print("  WARN — no reasoning_content / reasoning_tokens; thinking may be off")


def test_caching():
    print("\n=== TEST 3: automatic KV caching ===")
    big_text = (
        "You are a careful assistant. Below is reference material; "
        "study it carefully before responding.\n\n"
        + ("Sentence about widgets and gadgets, with mildly interesting "
           "but ultimately filler content. " * 200)
    )
    msgs1 = [
        {"role": "system", "content": big_text},
        {"role": "user", "content": "Reply with the single word: ONE."},
    ]
    msgs2 = [
        {"role": "system", "content": big_text},
        {"role": "user", "content": "Reply with the single word: TWO."},
    ]
    r1 = _call(msgs1, thinking=False, max_tokens=10)
    r2 = _call(msgs2, thinking=False, max_tokens=10)
    c1 = cached_tokens(r1.usage)
    c2 = cached_tokens(r2.usage)
    print(f"  call 1: prompt={r1.usage.prompt_tokens}  cached={c1}")
    print(f"  call 2: prompt={r2.usage.prompt_tokens}  cached={c2}")
    if c2 > 0:
        print(f"  PASS — second call hit cache ({c2} tokens cached)")
    else:
        print("  WARN — no cached_tokens reported on second call")


def main():
    _has_key()
    print(f"Smoke testing {MODEL} via {os.environ.get('MOONSHOT_API_BASE', 'litellm default')}")
    test_basic_call()
    test_thinking_enabled()
    test_caching()


if __name__ == "__main__":
    main()
