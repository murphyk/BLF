"""Online shrink-toward-prior layer.

For each source independently, maintain a per-source mixing weight α that
combines the base agent's forecast with a prior (the market freeze value):

    p_shrunk = sigmoid( α · logit(p_base) + (1−α) · logit(p_prior) )

α is fitted by 1-D minimisation of Brier on the per-source history of
resolved (p_base, p_prior, o) tuples seen so far, plus a ridge penalty on
α² that biases the cold-start solution toward 0 (pure prior). As the
per-source history grows the data dominates the ridge and α can drift away
from zero — so when the base agent is reliable, α grows; when the base
agent is worse than the prior, α stays near zero.

Per-source α is admittedly source-specific machinery, but is intended as a
temporary expedient until we have a content-conditioned shrinkage model.
Sources with no usable prior (no market_value) bypass shrinkage and return
p_base unchanged.
"""

from __future__ import annotations
import math
from collections import defaultdict


_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _brent_alpha(history: list[tuple[float, float, float]],
                 ridge: float, n_grid: int = 21) -> float:
    """Minimise Σ Brier(α) + ridge·α² over α ∈ [0,1].
    Coarse grid + golden-section refinement around the best grid point —
    avoids a scipy dependency and is plenty for a 1-D smooth-enough loss."""
    def loss(a: float) -> float:
        L = ridge * a * a
        for p_b, p_c, o in history:
            p = _sigmoid(a * _logit(p_b) + (1.0 - a) * _logit(p_c))
            L += (p - o) ** 2
        return L

    # Coarse grid
    grid = [i / (n_grid - 1) for i in range(n_grid)]
    best = min(grid, key=loss)
    # Golden-section refinement around best ± step
    step = 1.0 / (n_grid - 1)
    lo = max(0.0, best - step)
    hi = min(1.0, best + step)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a = lo + (1.0 - phi) * (hi - lo)
    b = lo + phi * (hi - lo)
    fa, fb = loss(a), loss(b)
    for _ in range(20):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = lo + (1.0 - phi) * (hi - lo)
            fa = loss(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = loss(b)
    return (a + b) / 2.0


class OnlineShrinker:
    def __init__(self, ridge: float = 1.0, recompute_every: int = 1):
        """
        ridge: weight on α² penalty. Larger -> more conservative cold-start.
        recompute_every: refit α every N updates per source (1 = every update).
        """
        self.ridge = float(ridge)
        self.recompute_every = int(recompute_every)
        self._H: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        self._alpha: dict[str, float] = defaultdict(lambda: 0.0)
        self._since_fit: dict[str, int] = defaultdict(int)

    def predict(self, source: str, p_base: float,
                p_prior: float | None) -> float:
        if p_prior is None:
            return p_base
        a = self._alpha.get(source, 0.0)
        return _sigmoid(a * _logit(p_base) + (1.0 - a) * _logit(p_prior))

    def update(self, source: str, p_base: float,
               p_prior: float | None, o: float) -> None:
        if p_prior is None:
            return
        self._H[source].append((float(p_base), float(p_prior), float(o)))
        self._since_fit[source] += 1
        if self._since_fit[source] >= self.recompute_every:
            self._alpha[source] = _brent_alpha(self._H[source], self.ridge)
            self._since_fit[source] = 0

    def alpha(self, source: str) -> float:
        return self._alpha.get(source, 0.0)

    def state(self) -> dict[str, dict]:
        return {s: {"n": len(self._H[s]), "alpha": self._alpha.get(s, 0.0)}
                for s in self._H}
