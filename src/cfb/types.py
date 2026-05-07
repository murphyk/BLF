"""Frozen dataclasses for CFB.

Conventions:
  u   unique id for one (base_question, forecast_due_date, resolution_date)
      triple. Multi-resolution dataset questions are flattened so each (u, f, r)
      describes exactly one resolution event.
  f   forecast-due date (the "asked" day).
  r   resolution date, with r >= f.
  o   outcome in [0, 1]. FB allows fractional resolutions (e.g. partial market
      resolutions); Brier extends naturally.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class Question:
    u: str
    source: str
    f: date
    r: date
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Resolution:
    u: str
    source: str
    f: date
    r: date
    o: int  # 0 or 1


@dataclass(frozen=True)
class PoolEntry:
    """Single (u, f, r, o) record stored in the frozen pool. The text + meta
    are kept alongside so Env can reconstitute Question objects without
    re-reading per-question files."""
    u: str
    source: str
    f: date
    r: date
    o: int
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
