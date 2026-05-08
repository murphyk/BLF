"""User-facing CFB driver.

Takes one or more agent strings and runs each end-to-end, caching base
runs under data/cfb/runs/<xid>/ so you only pay LLM cost once. Multiple
agents run in parallel.

Agent string grammar:

    agent_string := base ("+" layer)*
    base         := atomic | llm
    atomic       := "constant" | "market-value" | "empirical"
    llm          := <name> "-zs" ["-icl"] ["-c1"]
    layer        := "shrink" | "platt"

LLM names (mapped to litellm models in run_base_agent.LLM_MAP):
    flash, pro, sonnet, opus, haiku, gpt5

`-c1` means crowd=1 (inject market_value into the prompt). Absence ⇒ c=0.
`-icl` wraps the LLM in an ICL memory layer.

Examples:
    python -m src.cfb.run --agent flash-zs
    python -m src.cfb.run --agent flash-zs-c1+shrink+platt
    python -m src.cfb.run --agent flash-zs flash-zs-c1+shrink flash-zs-icl

Caching:
    If data/cfb/runs/<xid>/trajectory.jsonl already exists, that step is
    skipped. To force rerun, pass --force or delete the directory.
"""

from __future__ import annotations
import argparse
import concurrent.futures as cf
import os
import re
import sys
from datetime import date
from typing import NamedTuple

from .pool import _to_date
from .run_base_agent import (build_arg_parser as _build_base_parser,
                             run_base, LLM_MAP)
from .post_process import post
from .score import format_score, score, _read_traj


_LLM_NAMES = set(LLM_MAP.keys())
_ATOMIC = {"constant", "empirical", "market-value"}
_LAYERS = {"shrink", "platt"}


class Step(NamedTuple):
    xid: str       # destination xid for this step
    kind: str      # "base" or "layer"
    base_args: dict | None  # for base step
    in_xid: str | None      # for layer step
    layer: str | None       # for layer step


def parse_agent_string(s: str) -> list[Step]:
    """Decompose an agent string into a sequence of steps to execute.
    Each step's xid is the cumulative prefix, e.g.

        flash-zs-c1+shrink+platt
            -> [Step(xid='flash-zs-c1',         kind='base'),
                Step(xid='flash-zs-c1+shrink',  kind='layer', layer='shrink'),
                Step(xid='flash-zs-c1+shrink+platt', kind='layer', layer='platt')]
    """
    parts = s.split("+")
    base = parts[0]
    layer_names = parts[1:]

    base_args = _parse_base(base)
    steps = [Step(xid=base, kind="base", base_args=base_args,
                  in_xid=None, layer=None)]

    cumulative = base
    for ln in layer_names:
        if ln not in _LAYERS:
            raise ValueError(
                f"unknown layer '{ln}' in '{s}'; choices: {sorted(_LAYERS)}")
        prev = cumulative
        cumulative = cumulative + "+" + ln
        steps.append(Step(xid=cumulative, kind="layer", base_args=None,
                          in_xid=prev, layer=ln))
    return steps


def _parse_base(base: str) -> dict:
    """Return the kwargs dict that becomes a Namespace for run_base."""
    if base in _ATOMIC:
        return {"agent": base}

    # llm-zs[-icl][-c1]
    tokens = base.split("-")
    if len(tokens) >= 2 and tokens[0] in _LLM_NAMES and tokens[1] == "zs":
        rest = set(tokens[2:])
        unknown = rest - {"icl", "c1", "c0"}
        if unknown:
            raise ValueError(
                f"unknown token(s) in '{base}': {sorted(unknown)}")
        return {
            "agent": "llm",
            "llm": tokens[0],
            "zs": True,
            "icl": "icl" in rest,
            "use_crowd": "c1" in rest,
        }

    raise ValueError(
        f"could not parse agent base '{base}'. "
        f"Expected one of {sorted(_ATOMIC)} or '<llm>-zs[-icl][-c1]'.")


def _runs_dir(args) -> str:
    return args.runs_dir or os.path.join(
        os.path.expanduser("~/BLF"), "data", "cfb", "runs")


def _has_trajectory(xid: str, runs_dir: str) -> bool:
    return os.path.exists(os.path.join(runs_dir, xid, "trajectory.jsonl"))


def _materialize_base_args(step: Step, common: argparse.Namespace
                           ) -> argparse.Namespace:
    """Take the common pool/window args + the per-step base kwargs and emit
    a Namespace usable by run_base."""
    base = step.base_args
    ns = argparse.Namespace(**vars(common))
    ns.xid = step.xid
    ns.agent = base["agent"]
    ns.constant_p = 0.5
    ns.llm = base.get("llm")
    ns.zs = base.get("zs", False)
    ns.icl = base.get("icl", False)
    ns.use_crowd = base.get("use_crowd", False)
    return ns


def _execute_pipeline(agent_string: str, common: argparse.Namespace,
                      force: bool, runs_dir: str) -> dict:
    """Run all steps for one agent string. Returns the final score dict."""
    steps = parse_agent_string(agent_string)
    last_score = None
    for step in steps:
        if not force and _has_trajectory(step.xid, runs_dir):
            print(f"[cache hit] {step.xid}", file=sys.stderr)
            last_score = score(_read_traj(
                os.path.join(runs_dir, step.xid, "trajectory.jsonl")))
            continue
        if step.kind == "base":
            ns = _materialize_base_args(step, common)
            ns.quiet = common.quiet
            print(f"[run base] {step.xid}", file=sys.stderr)
            last_score = run_base(ns)
        else:
            print(f"[apply {step.layer}] {step.in_xid} -> {step.xid}",
                  file=sys.stderr)
            last_score = post(in_xid=step.in_xid, out_xid=step.xid,
                              layer=step.layer, runs_dir=runs_dir)
    return last_score


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    # Variadic agent list
    p.add_argument("--agent", nargs="+", required=True,
                   help="one or more agent strings, e.g. flash-zs-c1+shrink")
    # Pool / window (passed through to run_base)
    p.add_argument("--t0", type=_to_date, default=date(2025, 10, 26))
    p.add_argument("--tmax", type=_to_date, default=date(2026, 3, 29))
    p.add_argument("--pool", default=None)
    p.add_argument("--questions-dir", default=None)
    p.add_argument("--cfb-dir", default=None)
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--use-data-and-market", action="store_true")
    p.add_argument("--dedupe-base", action="store_true", default=True)
    p.add_argument("--no-dedupe-base", action="store_false", dest="dedupe_base")
    p.add_argument("--cap-per-source", type=int, default=None)
    # Behaviour
    p.add_argument("--force", action="store_true",
                   help="rerun steps even if trajectory.jsonl already exists")
    p.add_argument("--max-parallel", type=int, default=4,
                   help="max number of agent strings to run concurrently")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    # Validate strings up-front — fail fast before launching any work.
    for s in args.agent:
        parse_agent_string(s)

    runs_dir = _runs_dir(args)
    os.makedirs(runs_dir, exist_ok=True)

    results: dict[str, dict] = {}
    if len(args.agent) == 1:
        results[args.agent[0]] = _execute_pipeline(
            args.agent[0], args, args.force, runs_dir)
    else:
        with cf.ThreadPoolExecutor(
                max_workers=min(args.max_parallel, len(args.agent))) as ex:
            futs = {ex.submit(_execute_pipeline, s, args, args.force, runs_dir): s
                    for s in args.agent}
            for fut in cf.as_completed(futs):
                s = futs[fut]
                try:
                    results[s] = fut.result()
                except Exception as e:
                    print(f"[ERROR] {s}: {e!r}", file=sys.stderr)
                    results[s] = None

    print()
    print("=" * 78)
    print(f"{'xid':50s}  {'BI ev':>7s}  {'BI src':>7s}  {'n':>5s}")
    print("-" * 78)
    for s in args.agent:
        r = results.get(s)
        if not r:
            print(f"{s:50s}  ERROR")
            continue
        print(f"{s:50s}  {r['bi_event']:7.3f}  {r['bi_source']:7.3f}  "
              f"{r['n']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
