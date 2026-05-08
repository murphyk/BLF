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

from .schema import PoolEntry
from .env import Env
from .evaluator import Evaluator
from .pool import build_pool, freeze, load_pool, _to_date, MARKET_SOURCES, ALL_SOURCES
from .agents.constant import ConstantAgent
from .agents.empirical_prior import EmpiricalPriorAgent
from .agents.platt_wrapper import PlattWrapperAgent


def _parse_date(s: str) -> date:
    return _to_date(s)


def _resolve_pool(args) -> tuple[list[PoolEntry], str]:
    if args.pool:
        return load_pool(args.pool), args.pool
    questions_dir = args.questions_dir or os.path.join(
        os.path.expanduser("~/BLF"), "data", "questions")
    out_dir = args.cfb_dir or os.path.join(
        os.path.expanduser("~/BLF"), "data", "cfb")
    sources = ALL_SOURCES if args.use_data_and_market else MARKET_SOURCES
    entries = build_pool(questions_dir, args.t0, args.tmax,
                         sources=sources,
                         dedupe_base=args.dedupe_base,
                         cap_per_source=args.cap_per_source)
    pool_path, meta_path = freeze(
        entries, out_dir,
        build_params={
            "t0": args.t0.isoformat(),
            "t_max": args.tmax.isoformat(),
            "questions_dir": questions_dir,
            "dedupe_base": args.dedupe_base,
            "cap_per_source": args.cap_per_source,
            "sources": list(sources),
        },
    )
    print(f"[pool] froze {len(entries)} entries -> {pool_path}", file=sys.stderr)
    return entries, pool_path


def _make_agent(name: str):
    if name == "constant":
        return ConstantAgent(0.5)
    if name == "empirical":
        return EmpiricalPriorAgent(default=0.5)
    if name == "empirical-platt":
        return PlattWrapperAgent(EmpiricalPriorAgent(default=0.5))
    if name == "market-value":
        from .agents.market_value import MarketValueAgent
        return MarketValueAgent()
    if name == "flash-zs":
        from .agents.flash_zs import FlashZSAgent
        return FlashZSAgent(crowd=False)
    if name == "flash-zs-c1":
        from .agents.flash_zs import FlashZSAgent
        return FlashZSAgent(crowd=True)
    if name == "flash-zs-icl":
        from .agents.icl import ICLAgent
        return ICLAgent()
    if name == "flash-zs-icl-platt":
        from .agents.icl import ICLAgent
        return PlattWrapperAgent(ICLAgent())
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
    p.add_argument("--use-data-and-market", action="store_true",
                   help="include dataset sources (acled/fred/dbnomics/wikipedia/yfinance) "
                        "alongside the default markets-only mix")
    p.add_argument("--dedupe-base", action="store_true",
                   help="for each (source, base_id) keep only entries from the earliest forecast_due_date")
    p.add_argument("--cap-per-source", type=int, default=None,
                   help="cap base questions per source; stratified round-robin across forecast_due_dates with positive-bases first")
    p.add_argument("--trajectory-out", default=None,
                   help="write per-resolution JSONL trajectory "
                        "(one row per scored event, in chronological order) "
                        "for use with plot_continual.py")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    entries, _ = _resolve_pool(args)
    env = Env(entries, t0=args.t0, t_max=args.tmax)
    agent = _make_agent(args.agent)
    ev = Evaluator()

    traj_fh = open(args.trajectory_out, "w") if args.trajectory_out else None
    import json as _json

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
        if traj_fh and R:
            for r in R:
                p_used = ev._F.get(r.u, (0.5, None))[0]
                b = (p_used - r.o) ** 2
                traj_fh.write(_json.dumps({
                    "i": ev._n - len(R) + R.index(r) + 1,
                    "u": r.u, "source": r.source,
                    "f": r.f.isoformat(), "r": r.r.isoformat(),
                    "p": p_used, "o": r.o, "b": b,
                }) + "\n")
        if not args.quiet:
            print(f"{d.isoformat()}  Q={len(Q):3d}  R={len(R):3d}  "
                  f"n={L['n']:5d}  Bbar={L['brier_mean']!s}")
    if traj_fh:
        traj_fh.close()
        print(f"[traj] wrote {args.trajectory_out}", file=sys.stderr)

    print()
    print("FINAL", ev.score())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
