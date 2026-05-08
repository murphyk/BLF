"""Global empirical-prior agent: a single running mean of all observed
outcomes across all sources, used as the forecast for every question.

Source-blind by design — CFB hides the source from the agent because in real
deployments questions arrive from arbitrary providers and per-source
specialisation is not generally available.
"""

from __future__ import annotations
from ..schema import Question, Resolution


class EmpiricalPriorAgent:
    """Predict the running mean of past outcomes. Falls back to `default`
    until the first resolution arrives."""

    def __init__(self, default: float = 0.5):
        self.default = float(default)
        self._n = 0
        self._sum = 0.0

    def _predict(self) -> float:
        if self._n == 0:
            return self.default
        return self._sum / self._n

    def act(self, questions: list[Question]) -> dict[str, float]:
        p = self._predict()
        return {q.u: p for q in questions}

    def observe(self,
                questions: list[Question],
                forecasts: dict[str, float],
                resolutions: list[Resolution]) -> None:
        for r in resolutions:
            self._n += 1
            self._sum += float(r.o)

    def state(self) -> dict:
        return {"n": self._n, "pi_hat": self._predict()}
