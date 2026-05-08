"""Slim litellm wrapper used by CFB LLM agents.

Kept inside src/cfb/ (not src/agent/) so the CFB benchmark stays portable for
a future repo split. Loads .env on import so env vars are available.
"""

from __future__ import annotations
import os
import re
import threading
import warnings
from collections import defaultdict

import dotenv
import litellm

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

litellm.suppress_debug_info = True
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

_PROVIDER_LIMITS = {
    "openrouter": 50,
    "gemini": 4,
    "anthropic": 10,
    "openai": 20,
}
_provider_sems: dict[str, threading.Semaphore] = defaultdict(
    lambda: threading.Semaphore(10))
for _p, _l in _PROVIDER_LIMITS.items():
    _provider_sems[_p] = threading.Semaphore(_l)


def _sem(model: str) -> threading.Semaphore:
    parts = model.split("/")
    if not parts:
        return _provider_sems[model]
    if parts[0] == "openrouter" and len(parts) >= 2:
        backend = parts[1].lower()
        backend = {"google": "gemini"}.get(backend, backend)
        if backend in _PROVIDER_LIMITS:
            return _provider_sems[backend]
    return _provider_sems[parts[0]]


def chat(prompt: str, model: str, system: str = "",
         max_tokens: int = 512, temperature: float = 0.0,
         timeout: int = 60) -> str:
    """Single-turn LLM call. Returns text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    with _sem(model):
        resp = litellm.completion(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
            timeout=timeout, num_retries=3,
        )
    return resp.choices[0].message.content or ""


_PROB_RE = re.compile(
    r"(?:probability|p|answer|prob)\s*[:=]\s*([01](?:\.\d+)?|\.\d+)",
    re.IGNORECASE,
)
_BARE_RE = re.compile(r"\b([01](?:\.\d+)?|\.\d+)\b")


def parse_probability(text: str, default: float = 0.5) -> float:
    """Extract a probability in [0,1] from model output. Tries `probability=X`
    style first, then any bare decimal in [0,1]. Falls back to `default`."""
    if not text:
        return default
    m = _PROB_RE.search(text)
    if m:
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            pass
    # Last-resort: scan for any in-range decimal
    for m in _BARE_RE.finditer(text):
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return default
