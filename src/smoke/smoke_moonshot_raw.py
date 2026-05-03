"""smoke_moonshot_raw.py — raw HTTP call to api.moonshot.ai (bypass litellm).

Purpose: see exactly what Moonshot returns in its native response, so we
can decide whether reasoning_tokens is missing because litellm drops it
or because Moonshot doesn't return it.

Run:
    python -m src.smoke.smoke_moonshot_raw
"""

import json
import os

import dotenv
import httpx

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

API_BASE = os.environ.get("MOONSHOT_API_BASE", "https://api.moonshot.ai/v1")
API_KEY = os.environ["MOONSHOT_API_KEY"]


def main():
    body = {
        "model": "kimi-k2.5",
        "messages": [{"role": "user",
                      "content": "What is 17 * 23? Show your reasoning briefly."}],
        "max_tokens": 600,
    }
    r = httpx.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=90.0,
    )
    r.raise_for_status()
    data = r.json()
    print("=== top-level keys:", list(data.keys()))
    print("=== usage:")
    print(json.dumps(data.get("usage", {}), indent=2))
    msg = data["choices"][0]["message"]
    print("=== message keys:", list(msg.keys()))
    rc = msg.get("reasoning_content")
    content = msg.get("content")
    print(f"=== reasoning_content (len={len(rc) if rc else 0}): "
          f"{rc[:200] if rc else '<none>'}")
    print(f"=== content (len={len(content) if content else 0}): "
          f"{content[:200] if content else '<none>'}")


if __name__ == "__main__":
    main()
