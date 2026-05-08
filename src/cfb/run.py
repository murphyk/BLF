"""CFB driver: build pool (or load frozen), run env-agent-evaluator loop,
write the trajectory + manifest into data/cfb/runs/<xid>/, print final score.

Usage:
    python -m src.cfb.run --xid flash-zs-c1 --agent flash-zs-c1
    python -m src.cfb.run --xid emp --agent empirical --pool data/cfb/pool-be42200d.jsonl

If --pool is omitted the pool is rebuilt and frozen the first time.
The trajectory is the input to src.cfb.post (post-processing layers) and
src.cfb.plot_continual (reward-curve plot).
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
    p.add_argument("--xid", default=None,
                   help="experiment id; trajectory + manifest are written to "
                        "data/cfb/runs/<xid>/. defaults to <agent>")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    entries, pool_path = _resolve_pool(args)
    pool_idx = {e.u: e for e in entries}
    env = Env(entries, t0=args.t0, t_max=args.tmax)
    agent = _make_agent(args.agent)
    ev = Evaluator()

    xid = args.xid or args.agent
    runs_dir = args.runs_dir or os.path.join(
        os.path.expanduser("~/BLF"), "data", "cfb", "runs")
    out_dir = os.path.join(runs_dir, xid)
    os.makedirs(out_dir, exist_ok=True)
    traj_path = os.path.join(out_dir, "trajectory.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")
    traj_fh = open(traj_path, "w")

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
        if R:
            for ridx, r in enumerate(R):
                p_used = ev._F.get(r.u, (0.5, None))[0]
                pe = pool_idx.get(r.u)
                p_crowd = None
                if pe is not None:
                    mv = (pe.meta or {}).get("market_value")
                    if mv is not None and str(mv).strip() not in ("", "unknown", "None"):
                        try:
                            v = float(mv)
                            if 0.0 <= v <= 1.0:
                                p_crowd = v
                        except (TypeError, ValueError):
                            pass
                traj_fh.write(_json.dumps({
                    "i": ev._n - len(R) + ridx + 1,
                    "u": r.u, "source": r.source,
                    "f": r.f.isoformat(), "r": r.r.isoformat(),
                    "p": p_used, "p_raw": p_used, "p_crowd": p_crowd,
                    "o": r.o,
                }) + "\n")
        if not args.quiet:
            print(f"{d.isoformat()}  Q={len(Q):3d}  R={len(R):3d}  "
                  f"n={L['n']:5d}  Bbar={L['brier_mean']!s}")
    traj_fh.close()
    with open(manifest_path, "w") as fh:
        _json.dump({
            "xid": xid,
            "agent": args.agent,
            "t0": args.t0.isoformat(),
            "t_max": args.tmax.isoformat(),
            "pool": pool_path,
            "n_events": ev._n,
        }, fh, indent=2, default=str)

    # Final score (event-weighted + source-weighted)
    from .score import score, format_score, _read_traj
    s = score(_read_traj(traj_path))
    print()
    print(format_score(s))
    print()
    print(f"[run] xid={xid}  trajectory={traj_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
