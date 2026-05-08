"""Market-value baseline: predict p = market_value (the freeze value /
crowd estimate) directly. Falls back to `default` for questions without
a usable market_value (e.g. dataset questions, or markets where the field
is missing). No learning, no LLM."""

from __future__ import annotations
from ..schema import Question, Resolution


class MarketValueAgent:
    def __init__(self, default: float = 0.5):
        self.default = float(default)

    @staticmethod
    def _read_mv(q: Question) -> float | None:
        mv = (q.meta or {}).get("market_value")
        if mv is None or str(mv).strip() in ("", "unknown", "None"):
            return None
        try:
            v = float(mv)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= v <= 1.0):
            return None
        return v

    def act(self, questions: list[Question]) -> dict[str, float]:
        out: dict[str, float] = {}
        for q in questions:
            mv = self._read_mv(q)
            out[q.u] = mv if mv is not None else self.default
        return out

    def observe(self,
                questions: list[Question],
                forecasts: dict[str, float],
                resolutions: list[Resolution]) -> None:
        pass
