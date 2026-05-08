"""CFB environment: deterministic daily stream of questions and resolutions.

The env is independent of the agent — same Q(t), R(t) regardless of forecasts.
Agents cannot influence what the env shows next.
"""

from datetime import date, timedelta
from collections import defaultdict
from typing import Iterable, Iterator

from .schema import Question, Resolution, PoolEntry


class Env:
    def __init__(self, pool: Iterable[PoolEntry], t0: date, t_max: date):
        self.t0 = t0
        self.t_max = t_max
        self._t = t0
        self._by_f: dict[date, list[PoolEntry]] = defaultdict(list)
        self._by_r: dict[date, list[PoolEntry]] = defaultdict(list)
        for e in pool:
            if e.f < t0 or e.r > t_max:
                continue
            self._by_f[e.f].append(e)
            self._by_r[e.r].append(e)
        for d in self._by_f:
            self._by_f[d].sort(key=lambda e: e.u)
        for d in self._by_r:
            self._by_r[d].sort(key=lambda e: e.u)

    @property
    def t(self) -> date:
        return self._t

    @property
    def done(self) -> bool:
        return self._t > self.t_max

    def reset(self) -> date:
        self._t = self.t0
        return self._t

    def obs_questions(self) -> list[Question]:
        """Q(t) — questions asked on day t."""
        return [
            Question(u=e.u, source=e.source, f=e.f, r=e.r, text=e.text, meta=e.meta)
            for e in self._by_f.get(self._t, [])
        ]

    def obs_resolutions(self) -> list[Resolution]:
        """R(t) — resolutions revealed on day t. Pure ground truth."""
        return [
            Resolution(u=e.u, source=e.source, f=e.f, r=e.r, o=e.o)
            for e in self._by_r.get(self._t, [])
        ]

    def step(self) -> date:
        self._t = self._t + timedelta(days=1)
        return self._t

    def advance_to(self, t: date) -> date:
        if not (self.t0 <= t <= self.t_max + timedelta(days=1)):
            raise ValueError(f"t={t} outside [{self.t0}, {self.t_max}]")
        self._t = t
        return self._t

    def event_days(self) -> Iterator[date]:
        """Days in [t0, t_max] with Q(t) or R(t) non-empty, in chronological
        order. Pure generator — does not advance env state."""
        days = sorted(set(self._by_f) | set(self._by_r))
        for d in days:
            if self.t0 <= d <= self.t_max:
                yield d
