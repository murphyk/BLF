"""config_display.py — Backward-compatible re-exports from config.py.

All config logic is now in config.py. This module provides the old API
so existing imports continue to work without changes.
"""

from config.config import (
    model_short_name,
    load_results_config,
    pprint as _pprint,
    AgentConfig,
    _LLM_FULL_TO_SHORT,
    _THK_FULL_TO_SHORT,
    _SEARCH_FULL_TO_SHORT,
)

# Re-export _THINKING_SHORT and _SEARCH_SHORT under old names
_THINKING_SHORT = {
    "none": "none", "low": "low", "medium": "med", "high": "high", None: "default",
}
_SEARCH_SHORT = {
    "brave": "brave", "serper": "serper", "perplexity": "pplx",
    "asknews": "asknews", "none": "none",
}


def config_struct(config: dict) -> dict:
    """Extract key fields from a config into a structured display dict."""
    llm = config.get("llm", "")
    reasoning = config.get("reasoning_effort")
    search = config.get("search_engine", "brave")
    crowd = config.get("show_crowd", 0)
    tools = config.get("use_tools", True)
    ntrials = config.get("ntrials", 1)
    name = config.get("name", "")

    return {
        "model": model_short_name(llm),
        "think": _THINKING_SHORT.get(reasoning, str(reasoning) if reasoning else "default"),
        "search": _SEARCH_SHORT.get(search, search),
        "crowd": int(bool(crowd)),
        "tools": int(bool(tools)),
        "ntrials": ntrials,
        "name": name,
    }


def canonical_name(config: dict) -> str:
    """Generate a canonical name: {model}-{think}-{search}-c{crowd}-t{tools}[-n{ntrials}]."""
    s = config_struct(config)
    parts = [s["model"], s["think"], s["search"], f"c{s['crowd']}", f"t{s['tools']}"]
    if s["ntrials"] > 1:
        parts.append(f"n{s['ntrials']}")
    return "-".join(parts)
