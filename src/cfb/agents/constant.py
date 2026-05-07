"""Trivial constant-p agent. Useful as a calibration anchor (constant 0.5
gives Brier 0.25, brier_index 0)."""

from datetime import date
from ..types import Question, Resolution


class ConstantAgent:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def act(self, questions: list[Question]) -> dict[str, float]:
        return {q.u: self.p for q in questions}

    def observe(self,
                questions: list[Question],
                forecasts: dict[str, float],
                resolutions: list[Resolution]) -> None:
        pass
