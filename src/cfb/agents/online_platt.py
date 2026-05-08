"""Online Platt scaling: a 2-parameter logistic-regression calibrator
that updates as labelled data arrives.

Model:
    logit(q) = a * logit(p) + b
    q        = sigmoid(logit(q))            (calibrated probability)

where `p` is the base agent's pre-calibration forecast and `q` is what we
report. Initial (a, b) = (1, 0) — identity, so before any data we report
the base forecast unchanged.

Online estimator: Newton step on the binary-cross-entropy loss with a small
ridge prior, applied per (logit, label) pair as it arrives. This is the
standard "online logistic regression" derivation; with two parameters the
Hessian is a 2x2 matrix so each update is cheap.

API mirrors agents: `update(p, o)` and `predict(p)`.
"""

from __future__ import annotations
import math


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


class OnlinePlatt:
    """Recursive 2-parameter logistic regressor.

    ridge: L2 weight on (a-1, b) — pulls (a, b) toward identity (no calibration
           change) when data is sparse, so early updates can't overfit.
    lr:    Newton damping factor in (0, 1]. 1.0 is full step; <1 stabilises.
    """

    def __init__(self, ridge: float = 1.0, lr: float = 0.5):
        self.a = 1.0
        self.b = 0.0
        self.ridge = float(ridge)
        self.lr = float(lr)
        self.n = 0

    def predict(self, p: float) -> float:
        return _sigmoid(self.a * _logit(p) + self.b)

    def update(self, p: float, o: float) -> None:
        z = _logit(p)
        eta = self.a * z + self.b
        q = _sigmoid(eta)
        # Gradient of NLL + ridge prior on (a-1, b)
        ga = (q - o) * z + self.ridge * (self.a - 1.0)
        gb = (q - o)     + self.ridge * self.b
        # Hessian (2x2): w = q(1-q)
        w = q * (1.0 - q)
        Haa = w * z * z + self.ridge
        Hab = w * z
        Hbb = w + self.ridge
        det = Haa * Hbb - Hab * Hab
        if det <= 0.0:
            # Degenerate — fall back to gradient step
            self.a -= self.lr * ga
            self.b -= self.lr * gb
        else:
            # Newton step: theta -= H^{-1} g
            da = ( Hbb * ga - Hab * gb) / det
            db = (-Hab * ga + Haa * gb) / det
            self.a -= self.lr * da
            self.b -= self.lr * db
        # Clamp to a sane range so a single freak update can't blow up
        self.a = min(max(self.a, -10.0), 10.0)
        self.b = min(max(self.b, -10.0), 10.0)
        self.n += 1

    def state(self) -> dict:
        return {"a": self.a, "b": self.b, "n": self.n}
