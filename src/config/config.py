"""config.py — Agent configuration: defaults, delta parsing, display.

A config is specified as a delta string relative to a default template:

    "flash/thk:high/crowd:1"   →  change llm to flash, thinking to high, crowd to 1
    "pro/thk:med/search:none"  →  change llm to pro, thinking to medium, no search

The "/" separator is used for user input. For filenames and results paths,
pprint_path(cfg) returns a "-"-separated version safe for the filesystem.

For backward compat, legacy config names (e.g. "flash-high-brave-crowd1-tools1")
are resolved from experiments/configs/ or experiments/forecasts_raw/.
"""

import json
import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Short name ↔ full value mappings
# ---------------------------------------------------------------------------

_LLM_SHORT_TO_FULL = {
    "flash": "openrouter/google/gemini-3-flash-preview",
    "pro": "openrouter/google/gemini-3.1-pro-preview",
    "sonnet": "openrouter/anthropic/claude-sonnet-4-6",
    "opus": "openrouter/anthropic/claude-opus-4-6",
    "haiku": "openrouter/anthropic/claude-haiku-4-5-20251001",
    "gpt5": "openrouter/openai/gpt-5.4",
    "grok4": "openrouter/x-ai/grok-4",
    "grok4f": "openrouter/x-ai/grok-4.1-fast",
    "deepseek-r2": "openrouter/deepseek/deepseek-r2",
    "deepseek-v3": "openrouter/deepseek/deepseek-v3.2",
    "kimi-k2": "openrouter/moonshotai/kimi-k2",
    "kimi-k2t": "openrouter/moonshotai/kimi-k2-thinking",
    "kimi-k2.5": "openrouter/moonshotai/kimi-k2.5",
    "qwen3": "openrouter/qwen/qwen3-max-thinking",
}
_LLM_FULL_TO_SHORT = {v: k for k, v in _LLM_SHORT_TO_FULL.items()}

_THK_SHORT_TO_FULL = {"default": None, "none": "none", "low": "low", "med": "medium", "high": "high"}
_THK_FULL_TO_SHORT = {"none": "none", "low": "low", "medium": "med", "high": "high", None: "default"}

_SEARCH_SHORT_TO_FULL = {"brave": "brave", "serper": "serper", "pplx": "perplexity", "none": "none"}
_SEARCH_FULL_TO_SHORT = {v: k for k, v in _SEARCH_SHORT_TO_FULL.items()}
_SEARCH_FULL_TO_SHORT["brave"] = "brave"


# ---------------------------------------------------------------------------
# AgentConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    name: str = ""
    llm: str = "openrouter/google/gemini-3-flash-preview"
    max_tokens: int = 16000
    reasoning_effort: str | None = None
    search_engine: str = "brave"
    max_results_per_search: int = 10
    show_crowd: int = 0          # 1: include market price for market questions
    show_prior: int = 0          # 1: include empirical prior for dataset questions
    backtesting: bool = True
    use_tools: bool = True
    clairvoyant: bool = False
    max_steps: int = 10
    compact_threshold: int = 2000
    question_timeout: int = 240
    agg_method: str = "logit-mean"  # "logit-mean" (paper §C.9 eq.8, α=1; default) or "plain-mean" (arithmetic mean of probabilities)
    batch_queries: int = 0  # >0: non-agentic batch mode (N parallel queries, then submit)
    nobelief: bool = False   # True: disable structured belief state (text accumulation mode)
    fred_enhanced: bool = False  # True: append per-series classification to FRED tool output
    prior_only: bool = False     # True: submit empirical prior (dataset) or market price (market), no LLM
    halawi_prompt: bool = False  # True: use Halawi et al. (2024) zero-shot prompt instead of ours

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if not cfg.name:
            cfg.name = os.path.splitext(os.path.basename(path))[0]
        return cfg

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def copy(self, **overrides):
        d = self.to_dict()
        d.update(overrides)
        return AgentConfig(**d)


# ---------------------------------------------------------------------------
# Default config (template for delta parsing)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = AgentConfig(
    llm="openrouter/google/gemini-3-flash-preview",
    max_tokens=16000,
    reasoning_effort="high",
    search_engine="brave",
    max_results_per_search=10,
    show_crowd=1,
    show_prior=0,
    backtesting=True,
    use_tools=True,
    max_steps=10,
    compact_threshold=2000,
    question_timeout=600,
)


# ---------------------------------------------------------------------------
# Delta string parsing
# ---------------------------------------------------------------------------

_DELTA_KEYS = {
    "llm":    ("llm",              lambda v: _LLM_SHORT_TO_FULL.get(v, v)),
    "thk":    ("reasoning_effort", lambda v: _THK_SHORT_TO_FULL.get(v, v)),
    "search": ("search_engine",    lambda v: _SEARCH_SHORT_TO_FULL.get(v, v)),
    "crowd":  ("show_crowd",       lambda v: int(v)),
    "prior":  ("show_prior",       lambda v: int(v)),
    "tools":  ("use_tools",        lambda v: bool(int(v))),
    "steps":  ("max_steps",        lambda v: int(v)),
    "batch_queries": ("batch_queries", lambda v: int(v)),
    "nobelief": ("nobelief",          lambda v: bool(int(v))),
    "clairvoyant": ("clairvoyant",  lambda v: bool(int(v))),
    "fred_enhanced": ("fred_enhanced", lambda v: bool(int(v))),
    "prior_only": ("prior_only",       lambda v: bool(int(v))),
    "halawi": ("halawi_prompt",        lambda v: bool(int(v))),
    "timeout":("question_timeout", lambda v: int(v)),
    "tokens": ("max_tokens",       lambda v: int(v)),
}


def parse_config(delta_str: str, default: AgentConfig | None = None) -> AgentConfig:
    """Parse a delta string into a full AgentConfig.

    Format: "flash/thk:high/crowd:1/tools:0"
    - First bare token (no colon) = llm shortname
    - key:value pairs override specific fields
    - If it's a .json path or legacy config name, falls back to file loading
    """
    if default is None:
        default = DEFAULT_CONFIG

    if delta_str.endswith(".json"):
        return AgentConfig.from_json(delta_str)

    legacy_path = os.path.join("experiments", "configs", f"{delta_str}.json")
    if os.path.exists(legacy_path):
        return AgentConfig.from_json(legacy_path)

    cfg = default.copy()
    parts = delta_str.split("/")

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, val = part.split(":", 1)
            if key in _DELTA_KEYS:
                field_name, parser = _DELTA_KEYS[key]
                setattr(cfg, field_name, parser(val))
            else:
                raise ValueError(f"Unknown delta key: {key!r}. "
                                 f"Valid: {sorted(_DELTA_KEYS)}")
        else:
            cfg.llm = _LLM_SHORT_TO_FULL.get(part, part)

    cfg.name = pprint_path(cfg)
    return cfg


def resolve_config(name_or_delta: str) -> AgentConfig:
    """Resolve a config name, delta string, or file path to AgentConfig.

    Tries: legacy JSON → results config → delta string parse.
    """
    legacy_path = os.path.join("experiments", "configs", f"{name_or_delta}.json")
    if os.path.exists(legacy_path):
        return AgentConfig.from_json(legacy_path)

    results_path = os.path.join("experiments", "forecasts_raw", name_or_delta, "config.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            d = json.load(f)
        cfg = AgentConfig(**{k: v for k, v in d.items() if k in AgentConfig.__dataclass_fields__})
        if not cfg.name:
            cfg.name = name_or_delta
        return cfg

    return parse_config(name_or_delta)


# ---------------------------------------------------------------------------
# Display / pretty-print
# ---------------------------------------------------------------------------

def pprint(cfg: AgentConfig) -> str:
    """Short display string: "flash-high-brave-c1-t1" (for plot labels)."""
    parts = [
        _LLM_FULL_TO_SHORT.get(cfg.llm, cfg.llm.split("/")[-1]),
        _THK_FULL_TO_SHORT.get(cfg.reasoning_effort, str(cfg.reasoning_effort or "none")),
        _SEARCH_FULL_TO_SHORT.get(cfg.search_engine, cfg.search_engine),
        f"c{int(cfg.show_crowd)}",
        *([ f"p{int(cfg.show_prior)}" ] if cfg.show_prior else []),
        f"t{int(cfg.use_tools)}",
    ]
    if cfg.batch_queries > 0:
        parts.append(f"batch{cfg.batch_queries}")
    if cfg.nobelief:
        parts.append("nobelief")
    if cfg.clairvoyant:
        parts.append("clairvoyant")
    if cfg.fred_enhanced:
        parts.append("fredv2")
    if cfg.prior_only:
        parts.append("prior")
    if cfg.halawi_prompt:
        parts.append("halawi")
    return "-".join(parts)


def pprint_path(cfg: AgentConfig) -> str:
    """Filesystem-safe display string (same format as pprint, used for directory names)."""
    return pprint(cfg)


def pprint_multi(*cfgs: AgentConfig) -> tuple[str, list[str]]:
    """Factor configs into (common_str, [varying_str, ...]).

    Common fields go in the title, varying fields become per-config labels.
    """
    if not cfgs:
        return "", []
    if len(cfgs) == 1:
        return pprint(cfgs[0]), [pprint(cfgs[0])]

    all_fields = ["llm", "thk", "search", "crowd", "prior", "tools"]
    extractors = {
        "llm": lambda c: _LLM_FULL_TO_SHORT.get(c.llm, c.llm.split("/")[-1]),
        "thk": lambda c: _THK_FULL_TO_SHORT.get(c.reasoning_effort, str(c.reasoning_effort or "none")),
        "search": lambda c: _SEARCH_FULL_TO_SHORT.get(c.search_engine, c.search_engine),
        "crowd": lambda c: str(int(c.show_crowd)),
        "prior": lambda c: str(int(c.show_prior)),
        "tools": lambda c: str(int(c.use_tools)),
    }

    invariant = {}
    varying = []
    for field in all_fields:
        values = set(extractors[field](c) for c in cfgs)
        if len(values) == 1:
            invariant[field] = values.pop()
        else:
            varying.append(field)

    common_parts = []
    for field in all_fields:
        if field in invariant:
            v = invariant[field]
            if field in ("crowd", "tools"):
                common_parts.append(f"{field}={v}")
            else:
                common_parts.append(v)
    common = " / ".join(common_parts)

    per_config = []
    for cfg in cfgs:
        parts = []
        for field in varying:
            v = extractors[field](cfg)
            parts.append(v if field == "llm" else f"{field}={v}")
        per_config.append(", ".join(parts) if parts else pprint(cfg))

    return common, per_config


# ---------------------------------------------------------------------------
# Backward-compat re-exports (used by eval_html.py, eval_plots.py)
# ---------------------------------------------------------------------------

def model_short_name(llm: str) -> str:
    return _LLM_FULL_TO_SHORT.get(llm, llm.split("/")[-1])


def load_results_config(config_name: str) -> dict | None:
    """Load config dict from experiments/forecasts_raw/ or experiments/configs/."""
    base = (config_name.removesuffix("_calibrated")
            .removesuffix("_aggregated"))
    for path in [os.path.join("experiments", "forecasts_raw", base, "config.json"),
                 os.path.join("experiments", "configs", f"{base}.json")]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None
