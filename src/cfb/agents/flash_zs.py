"""Gemini Flash zero-shot agent.

Single LLM call per question, no tools, no ICL memory. Uses the Halawi et
al. (2024) zero-shot prompt — the same format BLF uses — so results are
directly comparable to Table 17 of the paper.

`crowd=True` adds the market `freeze value` to the prompt (= BLF's c=1
configuration). `crowd=False` is the c=0 configuration (no anchor).

Source is intentionally NOT included in the prompt: CFB hides it because in
real deployments questions arrive from arbitrary providers.
"""

from __future__ import annotations
import concurrent.futures as cf
from datetime import date
from ..schema import Question, Resolution
from ._llm import chat, parse_probability


HALAWI_SYSTEM = (
    "You are an expert superforecaster, familiar with the work of Tetlock "
    "and others. Make a prediction of the probability that the question "
    "will be resolved as true. You MUST give a probability estimate "
    "between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you "
    "can't answer, pick the base rate, but return a number between 0 and 1."
)

HALAWI_TEMPLATE = (
    "Question: {question}\n"
    "Question Background: {background}\n"
    "Resolution Criteria: {criteria}\n"
    "Question close date: {close_date}\n"
    "{freeze}"
    "Output your answer (a number between 0 and 1) with an asterisk at the "
    "beginning and end of the decimal. Do not output anything else.\n"
    "Answer:\n"
    "{{ Insert answer here }}"
)


def render_halawi(q: Question, crowd: bool = False) -> str:
    meta = q.meta or {}
    market_val = meta.get("market_value")
    has_market = market_val is not None and \
        str(market_val).strip() not in ("", "unknown", "None")

    freeze = ""
    if crowd and has_market:
        expl = meta.get("market_value_explanation") or ""
        if expl:
            freeze = f"The freeze value is {market_val}. {expl}\n"
        else:
            freeze = f"The freeze value is {market_val}.\n"

    return HALAWI_TEMPLATE.format(
        question=q.text,
        background=(meta.get("background") or "")[:2000],
        criteria=meta.get("resolution_criteria") or "",
        close_date=q.r.isoformat(),
        freeze=freeze,
    )


class FlashZSAgent:
    DEFAULT_MODEL = "openrouter/google/gemini-3-flash-preview"

    def __init__(self, model: str = None, crowd: bool = False,
                 max_tokens: int = 64, max_workers: int = 8):
        self.model = model or self.DEFAULT_MODEL
        self.crowd = crowd
        self.max_tokens = max_tokens
        self.max_workers = max_workers

    def _forecast_one(self, q: Question) -> tuple[str, float]:
        prompt = render_halawi(q, crowd=self.crowd)
        try:
            text = chat(prompt, model=self.model, system=HALAWI_SYSTEM,
                        max_tokens=self.max_tokens, temperature=0.0)
        except Exception:
            return q.u, 0.5
        return q.u, parse_probability(text, default=0.5)

    def act(self, questions: list[Question]) -> dict[str, float]:
        if not questions:
            return {}
        out: dict[str, float] = {}
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for u, p in ex.map(self._forecast_one, questions):
                out[u] = p
        return out

    def observe(self,
                questions: list[Question],
                forecasts: dict[str, float],
                resolutions: list[Resolution]) -> None:
        pass  # stateless


# Backwards-compat alias used by ICLAgent
SYSTEM = HALAWI_SYSTEM


def render_question(q: Question) -> str:
    """Halawi-format prompt without crowd anchor (c=0). ICL agent uses this
    for the trailing 'NEW question' block."""
    return render_halawi(q, crowd=False)
