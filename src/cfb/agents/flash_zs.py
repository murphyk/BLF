"""Gemini Flash zero-shot agent — single LLM call per question, no tools, no
in-context examples. The stateless baseline for the continual-learning plot.

Source is intentionally NOT included in the prompt: CFB hides it because in
real deployments questions arrive from arbitrary providers."""

from __future__ import annotations
import concurrent.futures as cf
from datetime import date
from ..schema import Question, Resolution
from ._llm import chat, parse_probability


SYSTEM = (
    "You are a careful binary forecaster. You will be given a yes/no question "
    "with a resolution criterion and resolution date. Estimate the probability "
    "the question resolves YES. Reply with a single line of the form\n"
    "  probability=<float in [0,1]>\n"
    "with no additional text, no explanation, no caveats."
)


def render_question(q: Question) -> str:
    meta = q.meta or {}
    parts = [
        f"Question: {q.text}",
    ]
    rc = meta.get("resolution_criteria") or ""
    if rc:
        parts.append(f"Resolution criterion: {rc}")
    bg = meta.get("background") or ""
    if bg:
        parts.append(f"Background: {bg[:1500]}")
    parts.append(f"Forecast date: {q.f.isoformat()}")
    parts.append(f"Resolution date: {q.r.isoformat()}")
    return "\n\n".join(parts)


class FlashZSAgent:
    DEFAULT_MODEL = "openrouter/google/gemini-3-flash-preview"

    def __init__(self, model: str = None, max_tokens: int = 256,
                 max_workers: int = 8):
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.max_workers = max_workers

    def _forecast_one(self, q: Question) -> tuple[str, float]:
        prompt = render_question(q)
        try:
            text = chat(prompt, model=self.model, system=SYSTEM,
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
