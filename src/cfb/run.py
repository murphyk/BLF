"""CFB driver: build pool (or load frozen), run env-agent-evaluator loop, print
trace + final score.

Usage:
    python -m src.cfb.run --t0 2025-10-26 --tmax 2026-03-29 --agent constant

The first run builds + freezes a pool under data/cfb/. Subsequent runs with
the same --t0/--tmax reuse the same frozen pool by hash.
"""

from __future__ import annotations
import argparse
import os
import sys
from datetime import date

from .types import PoolEntry
from .env import Env
from .evaluator import Evaluator
from .pool import build_pool, freeze, load_pool, _to_date
from .agents.constant import ConstantAgent
from .agents.empirical_prior import EmpiricalPriorAgent


def _parse_date(s: str) -> date:
    return _to_date(s)


def _resolve_pool(args) -> tuple[list[PoolEntry], str]:
    if args.pool:
        return load_pool(args.pool), args.pool
    questions_dir = args.questions_dir or os.path.join(
        os.path.expanduser("~/BLF"), "data", "questions")
    out_dir = args.cfb_dir or os.path.join(
        os.path.expanduser("~/BLF"), "data", "cfb")
    entries = build_pool(questions_dir, args.t0, args.tmax,
                         dedupe_base=args.dedupe_base)
    pool_path, meta_path = freeze(
        entries, out_dir,
        build_params={
            "t0": args.t0.isoformat(),
            "t_max": args.tmax.isoformat(),
            "questions_dir": questions_dir,
            "dedupe_base": args.dedupe_base,
        },
    )
    print(f"[pool] froze {len(entries)} entries -> {pool_path}", file=sys.stderr)
    return entries, pool_path


def _make_agent(name: str):
    if name == "constant":
        return ConstantAgent(0.5)
    if name == "empirical":
        return EmpiricalPriorAgent(default=0.5)
    raise SystemExit(f"unknown agent: {name}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--t0", type=_parse_date, default=date(2025, 10, 26))
    p.add_argument("--tmax", type=_parse_date, default=date(2026, 3, 29))
    p.add_argument("--agent", default="constant")
    p.add_argument("--pool", default=None,
                   help="load a pre-built frozen pool instead of rebuilding")
    p.add_argument("--questions-dir", default=None)
    p.add_argument("--cfb-dir", default=None)
    p.add_argument("--dedupe-base", action="store_true",
                   help="for each (source, base_id) keep only entries from the earliest forecast_due_date")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    entries, _ = _resolve_pool(args)
    env = Env(entries, t0=args.t0, t_max=args.tmax)
    agent = _make_agent(args.agent)
    ev = Evaluator()

    env.reset()
    for d in env.event_days():
        env.advance_to(d)
        Q = env.obs_questions()
        R = env.obs_resolutions()
        P = agent.act(Q) if Q else {}
        if Q:
            ev.submit(d, P)
        agent.observe(Q, P, R)
        L = ev.update_loss(R) if R else {"n": ev._n, "brier_mean": None}
        if not args.quiet:
            print(f"{d.isoformat()}  Q={len(Q):3d}  R={len(R):3d}  "
                  f"n={L['n']:5d}  Bbar={L['brier_mean']!s}")

    print()
    print("FINAL", ev.score())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
