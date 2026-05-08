"""PlattWrapperAgent: post-hoc online calibration on top of any base agent.

  base.act(Q) -> p_raw
  agent.act(Q) returns OnlinePlatt(p_raw) for each question (calibrated)
  agent.observe(Q, P, R) trains the calibrator on (p_raw, o) pairs

The calibrator is a single global 2-parameter logistic regression — no
per-source machinery (CFB hides source).
"""

from __future__ import annotations
from ..schema import Question, Resolution
from .online_platt import OnlinePlatt


class PlattWrapperAgent:
    def __init__(self, base, ridge: float = 1.0, lr: float = 0.5):
        self.base = base
        self.platt = OnlinePlatt(ridge=ridge, lr=lr)
        self._raw: dict[str, float] = {}  # u -> base's pre-calibration p

    def act(self, questions: list[Question]) -> dict[str, float]:
        raw = self.base.act(questions)
        out: dict[str, float] = {}
        for u, p in raw.items():
            self._raw[u] = p
            out[u] = self.platt.predict(p)
        return out

    def observe(self,
                questions: list[Question],
                forecasts: dict[str, float],
                resolutions: list[Resolution]) -> None:
        # First, train Platt on the base's PRE-calibration forecasts using R.
        for r in resolutions:
            p = self._raw.pop(r.u, None)
            if p is None:
                continue
            self.platt.update(p, float(r.o))
        # Then let the base do its own learning (e.g. ICL memory update).
        self.base.observe(questions, forecasts, resolutions)

    def state(self) -> dict:
        return {"platt": self.platt.state(),
                "base": getattr(self.base, "state", lambda: {})()}
