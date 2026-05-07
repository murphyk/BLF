"""CFB — Continual Forecasting Benchmark.

Online backtesting environment for binary forecasting agents. Three independent
objects: Env (deterministic stream of questions and resolutions), Agent
(policy Q -> P), Evaluator (matches P with R, tracks running Brier).
"""

from .types import Question, Resolution, PoolEntry
from .env import Env
from .evaluator import Evaluator

__all__ = ["Question", "Resolution", "PoolEntry", "Env", "Evaluator"]
