"""Empirical-prior agent: per-(source, subtype) running mean of observed
outcomes, used as the forecast for new questions of that bucket.

Subtype classifier inlined here (duplicating src/config/empirical_prior.py)
so src/cfb/ stays self-contained for a future repo split.

Convergence target: Table 7 (π_q) of the paper, computed offline from all FB
resolutions. Online estimates should approach those numbers as the run
progresses.
"""

from __future__ import annotations
from collections import defaultdict
from ..schema import Question, Resolution


_DATASET_SOURCES = {"acled", "wikipedia", "fred", "yfinance", "dbnomics"}


def classify_subtype(source: str, text: str) -> str:
    if source == "acled":
        return "10x_spike" if "ten times" in text else "increase"
    if source == "wikipedia":
        t = text.lower()
        if "vaccine" in t:
            return "vaccine"
        if "at least 1%" in text or "Elo rating" in text:
            return "fide_elo"
        if "FIDE" in text or "ranking" in t:
            return "fide_rank"
        if "world record" in t or "swimming" in t:
            return "swimming"
        return "vaccine"
    return "all"


def _bucket(source: str, text: str) -> tuple[str, str]:
    if source in _DATASET_SOURCES:
        return source, classify_subtype(source, text)
    return source, "all"


class EmpiricalPriorAgent:
    """Maintains running (n, sum_o) per (source, subtype). Predicts the bucket
    mean; falls back to `default` when the bucket has no observations yet."""

    def __init__(self, default: float = 0.5):
        self.default = float(default)
        self._n: dict[tuple[str, str], int] = defaultdict(int)
        self._s: dict[tuple[str, str], float] = defaultdict(float)
        # u -> bucket, so we can update from R without re-classifying.
        self._u_bucket: dict[str, tuple[str, str]] = {}

    def _predict(self, bucket: tuple[str, str]) -> float:
        n = self._n[bucket]
        if n == 0:
            return self.default
        return self._s[bucket] / n

    def act(self, questions: list[Question]) -> dict[str, float]:
        out: dict[str, float] = {}
        for q in questions:
            b = _bucket(q.source, q.text)
            self._u_bucket[q.u] = b
            out[q.u] = self._predict(b)
        return out

    def observe(self,
                questions: list[Question],
                forecasts: dict[str, float],
                resolutions: list[Resolution]) -> None:
        for r in resolutions:
            b = self._u_bucket.pop(r.u, None)
            if b is None:
                # Resolution arrived for a question we never forecast (shouldn't
                # happen if act() ran on its asked-day). Reconstruct via source.
                b = (r.source, "all")
            self._n[b] += 1
            self._s[b] += float(r.o)

    def state(self) -> dict[tuple[str, str], tuple[int, float]]:
        return {b: (self._n[b], self._s[b] / self._n[b])
                for b in self._n if self._n[b] > 0}
