"""Evaluator: keeps the forecast log and computes running Brier as resolutions
arrive. Agent-independent — given the same (P, R) sequence it produces the same
loss trajectory.

Forecast log:
    F : u -> (p_u, f_u)

When a Resolution(u, f, r, o) arrives:
    B_u = (F[u].p - o)^2,
    n  += 1,
    SumB += B_u,
    Bbar = SumB / n.
"""

from datetime import date
from .schema import Resolution


class Evaluator:
    def __init__(self):
        self._F: dict[str, tuple[float, date]] = {}
        self._n = 0
        self._sum_b = 0.0
        self._scored: dict[str, float] = {}  # u -> B_u, for diagnostics

    def submit(self, t: date, forecasts: dict[str, float]) -> None:
        for u, p in forecasts.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"forecast for {u} out of range: {p}")
            if u in self._F:
                raise ValueError(f"duplicate forecast for {u}")
            self._F[u] = (float(p), t)

    def update_loss(self, resolutions: list[Resolution]) -> dict:
        new_n = 0
        new_sum = 0.0
        for res in resolutions:
            entry = self._F.get(res.u)
            if entry is None:
                # Question resolved but agent never forecast it. Penalize at 0.5
                # (Brier 0.25). This shouldn't happen if the agent plays every
                # question — flag it in diagnostics.
                p = 0.5
            else:
                p = entry[0]
            b = (p - res.o) ** 2
            self._scored[res.u] = b
            new_n += 1
            new_sum += b
        self._n += new_n
        self._sum_b += new_sum
        return {
            "delta_n": new_n,
            "delta_brier_mean": (new_sum / new_n) if new_n else None,
            "n": self._n,
            "brier_mean": (self._sum_b / self._n) if self._n else None,
        }

    def score(self) -> dict:
        return {
            "n": self._n,
            "brier_mean": (self._sum_b / self._n) if self._n else None,
            "brier_index": (1.0 - 4.0 * (self._sum_b / self._n))
                            if self._n else None,
        }
