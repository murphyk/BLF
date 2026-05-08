"""Run a single BASE agent end-to-end and write its trajectory under
data/cfb/runs/<xid>/.

Front-end (the user-facing one) is `run.py` — that takes string-form agent
configs like 'flash-zs-c1+shrink+platt' and chains a base run + post layers.
This module handles only the base-agent step: a single LLM-or-non-LLM
forecaster running through the env once.

Non-LLM bases:
    --agent constant
    --agent empirical
    --agent market-value

LLM base:
    --agent llm --llm <name> --zs [--icl] [--use-crowd]

LLM identifiers (--llm):
    flash, pro, sonnet, opus, haiku, gpt5
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import date
from typing import Any

from .schema import PoolEntry
from .env import Env
from .evaluator import Evaluator
from .pool import build_pool, freeze, load_pool, _to_date, MARKET_SOURCES, ALL_SOURCES
from .score import score, format_score, _read_traj


LLM_MAP = {
    "flash":  "openrouter/google/gemini-3-flash-preview",
    "pro":    "openrouter/google/gemini-3.1-pro-preview",
    "sonnet": "openrouter/anthropic/claude-sonnet-4-6",
    "opus":   "openrouter/anthropic/claude-opus-4-6",
    "haiku":  "openrouter/anthropic/claude-haiku-4-5-20251001",
    "gpt5":   "openrouter/openai/gpt-5.4",
}


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
    pool_path, _ = freeze(
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


def _make_agent(args):
    """Build the base agent from structured flags. No string parsing here."""
    kind = args.agent
    if kind == "constant":
        from .agents.constant import ConstantAgent
        return ConstantAgent(args.constant_p)
    if kind == "empirical":
        from .agents.empirical_prior import EmpiricalPriorAgent
        return EmpiricalPriorAgent(default=0.5)
    if kind == "market-value":
        from .agents.market_value import MarketValueAgent
        return MarketValueAgent(default=0.5)
    if kind == "llm":
        if args.llm not in LLM_MAP:
            raise SystemExit(f"unknown --llm {args.llm}; "
                             f"choices: {sorted(LLM_MAP)}")
        model = LLM_MAP[args.llm]
        if not args.zs:
            raise SystemExit("currently only --zs is supported for --agent llm")
        if args.icl:
            from .agents.icl import ICLAgent
            return ICLAgent(model=model, crowd=args.use_crowd)
        from .agents.flash_zs import FlashZSAgent
        return FlashZSAgent(model=model, crowd=args.use_crowd)
    raise SystemExit(f"unknown --agent {kind}")


def run_base(args) -> dict:
    """Run the env-agent-evaluator loop and write trajectory + manifest.
    Returns the score dict."""
    entries, pool_path = _resolve_pool(args)
    pool_idx = {e.u: e for e in entries}
    env = Env(entries, t0=args.t0, t_max=args.tmax)
    agent = _make_agent(args)
    ev = Evaluator()

    runs_dir = args.runs_dir or os.path.join(
        os.path.expanduser("~/BLF"), "data", "cfb", "runs")
    out_dir = os.path.join(runs_dir, args.xid)
    os.makedirs(out_dir, exist_ok=True)
    traj_path = os.path.join(out_dir, "trajectory.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")
    traj_fh = open(traj_path, "w")

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
                p_crowd: float | None = None
                if pe is not None:
                    mv = (pe.meta or {}).get("market_value")
                    if mv is not None and str(mv).strip() not in ("", "unknown", "None"):
                        try:
                            v = float(mv)
                            if 0.0 <= v <= 1.0:
                                p_crowd = v
                        except (TypeError, ValueError):
                            pass
                traj_fh.write(json.dumps({
                    "i": ev._n - len(R) + ridx + 1,
                    "u": r.u, "source": r.source,
                    "f": r.f.isoformat(), "r": r.r.isoformat(),
                    "p": p_used, "p_raw": p_used, "p_crowd": p_crowd,
                    "o": r.o,
                }) + "\n")
        if not args.quiet:
            print(f"[{args.xid}] {d.isoformat()}  Q={len(Q):3d}  R={len(R):3d}  "
                  f"n={L['n']:5d}  Bbar={L['brier_mean']!s}")
    traj_fh.close()
    manifest: dict[str, Any] = {
        "xid": args.xid,
        "agent": args.agent,
        "t0": args.t0.isoformat(),
        "t_max": args.tmax.isoformat(),
        "pool": pool_path,
        "n_events": ev._n,
    }
    if args.agent == "llm":
        manifest.update({
            "llm": args.llm, "model": LLM_MAP[args.llm],
            "mode": "icl" if args.icl else "zs",
            "use_crowd": args.use_crowd,
        })
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    s = score(_read_traj(traj_path))
    if not args.quiet:
        print()
        print(format_score(s))
    print(f"[run_base] xid={args.xid}  trajectory={traj_path}", file=sys.stderr)
    return s


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Pool / window
    p.add_argument("--t0", type=_parse_date, default=date(2025, 10, 26))
    p.add_argument("--tmax", type=_parse_date, default=date(2026, 3, 29))
    p.add_argument("--pool", default=None)
    p.add_argument("--questions-dir", default=None)
    p.add_argument("--cfb-dir", default=None)
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--use-data-and-market", action="store_true")
    p.add_argument("--dedupe-base", action="store_true", default=True)
    p.add_argument("--no-dedupe-base", action="store_false", dest="dedupe_base")
    p.add_argument("--cap-per-source", type=int, default=None)
    # Agent
    p.add_argument("--agent", required=True,
                   choices=["constant", "empirical", "market-value", "llm"])
    p.add_argument("--constant-p", type=float, default=0.5)
    p.add_argument("--llm", default=None,
                   choices=sorted(LLM_MAP.keys()))
    p.add_argument("--zs", action="store_true",
                   help="(--agent llm) zero-shot mode — currently the only LLM mode")
    p.add_argument("--icl", action="store_true",
                   help="(--agent llm) wrap with ICL memory of past (q, o) pairs")
    p.add_argument("--use-crowd", action="store_true",
                   help="(--agent llm) inject market freeze value into the prompt (c=1)")
    # Output
    p.add_argument("--xid", required=True,
                   help="experiment id; trajectory written to data/cfb/runs/<xid>/")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_base(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
