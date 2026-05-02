#!/usr/bin/env python3
"""test_completion_rate.py — diagnose mid-loop completion failures cheaply.

Reads experiments/forecasts_raw/{config}/trial_*/ for one or more configs
and prints a per-source × per-config table showing:

  - Submit % (LLM successfully called the submit tool before max_steps)
  - Empty-completion % (LLM returned no tool call AND no text — agent loop
    exits at agent.py:140-144; fallback to belief.p in BLF mode, p=0.5 in NoBel)
  - Other failure % (max_steps timeout, LLM API error, tool error, etc.)

Background. Different frontier LLMs have very different rates of returning
empty completions during agent loops. We've observed:

  Pro 3.1 high (BLF): polymarket/manifold ≈ 24-33% submit rate (very poor).
  Pro 3.1 high (NoBel): polymarket/manifold ≈ 12-15% submit rate.
  Sonnet 4.6: 100% submit on every source we've tested.

The BLF architecture's 'anytime estimator' (belief.p tracked across steps)
absorbs these failures gracefully — when the LLM returns an empty
completion, the most recent belief.p is used as the forecast. NoBel falls
back to p=0.5, which is catastrophic on market questions.

Workflow.

    # 1. Run predict on the smoke-llm exam (~13 Q × 2 trials per LLM,
    #    ~$5-15 per LLM, ~15-30 min wall).
    python3 src/core/predict.py --exam smoke-llm \\
        --config pro/thk:high/crowd:1/prior:1/tools:1 --ntrials 2
    python3 src/core/predict.py --exam smoke-llm \\
        --config sonnet/thk:high/crowd:1/prior:1/tools:1 --ntrials 2
    # ...repeat for gpt5, grok4, etc.

    # 2. Diagnose:
    python3 src/testing/test_completion_rate.py \\
        --configs pro-high-brave-c1-p1-t1,sonnet-high-brave-c1-p1-t1,\\
                  gpt5-high-brave-c1-p1-t1,grok4-high-brave-c1-p1-t1

    # 3. Decision rule: if any (config × source) submit rate is < 80%,
    #    that LLM is unsuitable for NoBel-style agents on that source
    #    without an anytime-estimator fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


_FAIL_THRESHOLD = 0.80  # below this submit rate, flag as problematic


def _classify_trial(fc: dict) -> str:
    """Return one of: SUBMIT, NO_TOOL_CALL, MAX_STEPS, ERROR, OTHER."""
    if fc.get("submitted"):
        return "SUBMIT"
    tlog = fc.get("tool_log") or []
    if not tlog:
        return "OTHER"
    last = tlog[-1].get("type", "")
    if last == "no_tool_call":
        return "NO_TOOL_CALL"
    if last == "timeout":
        return "MAX_STEPS"
    if last == "error":
        return "ERROR"
    n_steps = fc.get("n_steps", 0) or 0
    if n_steps >= 10:  # max_steps default
        return "MAX_STEPS"
    return "OTHER"


def _exam_qids(exam: str | None) -> dict[str, set[str]] | None:
    """Return {source: {qid}} for exam, or None to disable filtering.
    Reads data/exams/{exam}/indices.json (the canonical exam definition).
    """
    if not exam:
        return None
    idx_path = os.path.join("data", "exams", exam, "indices.json")
    if not os.path.isfile(idx_path):
        print(f"  [exam={exam}] no data/exams/{exam}/indices.json — ignoring filter")
        return None
    with open(idx_path) as f:
        data = json.load(f)
    return {src: set(ids) for src, ids in data.items()}


def _matches_exam(src: str, fn: str, qids: dict[str, set[str]] | None) -> bool:
    if qids is None:
        return True
    base = fn[:-5] if fn.endswith(".json") else fn
    src_qids = qids.get(src, set())
    if base in src_qids:
        return True
    # Also accept exam qids that include a date suffix the trial filename lacks
    return any(q.startswith(base + "_") or base.startswith(q + "_") or q == base
               for q in src_qids)


def _gather(config: str, exam_qids: dict[str, set[str]] | None = None
            ) -> dict[str, dict[str, int]]:
    """Returns {source: {bucket: count}}."""
    by_src: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cfg_root = os.path.join("experiments", "forecasts_raw", config)
    if not os.path.isdir(cfg_root):
        print(f"  [{config}] no forecasts_raw/ dir — did you run predict?")
        return {}
    for trial in sorted(os.listdir(cfg_root)):
        trial_dir = os.path.join(cfg_root, trial)
        if not (trial.startswith("trial_") and os.path.isdir(trial_dir)):
            continue
        for src in os.listdir(trial_dir):
            sd = os.path.join(trial_dir, src)
            if not os.path.isdir(sd):
                continue
            for fn in os.listdir(sd):
                if not fn.endswith(".json"):
                    continue
                if not _matches_exam(src, fn, exam_qids):
                    continue
                with open(os.path.join(sd, fn)) as f:
                    fc = json.load(f)
                bucket = _classify_trial(fc)
                by_src[src][bucket] += 1
                by_src[src]["_total"] += 1
    return by_src


def _print_table(configs: list[str], stats: dict[str, dict],
                 sources_order: list[str]):
    """Print a per-(source × config) submit% table with flags."""
    cell_w = max(14, max(len(c) for c in configs) + 2)
    print(f"\nSubmit % per (source × config) — flagged with ⚠ if < {_FAIL_THRESHOLD:.0%}")
    print(f"{'source':<14s}", end="")
    for c in configs:
        print(f" {c:>{cell_w}s}", end="")
    print()
    print("-" * (14 + (cell_w + 1) * len(configs)))

    flagged: list[tuple[str, str, float]] = []
    for src in sources_order:
        row_cells = []
        for c in configs:
            d = stats.get(c, {}).get(src, {})
            tot = d.get("_total", 0)
            sub = d.get("SUBMIT", 0)
            if tot == 0:
                row_cells.append("    -    ")
                continue
            rate = sub / tot
            flag = "⚠" if rate < _FAIL_THRESHOLD else " "
            row_cells.append(f"{flag} {sub}/{tot} ({rate:>4.0%})")
            if rate < _FAIL_THRESHOLD:
                flagged.append((c, src, rate))
        print(f"{src:<14s}", end="")
        for cell in row_cells:
            print(f" {cell:>{cell_w}s}", end="")
        print()

    # Overall
    print("-" * (14 + (cell_w + 1) * len(configs)))
    print(f"{'TOTAL':<14s}", end="")
    for c in configs:
        d = stats.get(c, {})
        tot = sum(s.get("_total", 0) for s in d.values())
        sub = sum(s.get("SUBMIT", 0) for s in d.values())
        if tot == 0:
            print(f" {'    -    ':>{cell_w}s}", end="")
        else:
            rate = sub / tot
            flag = "⚠" if rate < _FAIL_THRESHOLD else " "
            cell = f"{flag} {sub}/{tot} ({rate:>4.0%})"
            print(f" {cell:>{cell_w}s}", end="")
    print("\n")

    if flagged:
        print("Flagged (LLM, source) pairs (submit rate < 80%):")
        for c, src, rate in sorted(flagged, key=lambda x: x[2]):
            print(f"  {c:<35s} {src:<12s} {rate:>4.0%}")
    else:
        print("All (LLM, source) pairs above 80% submit threshold ✓")


def _print_failure_breakdown(configs: list[str], stats: dict[str, dict]):
    print("\nFailure-mode breakdown (% of all trials per config):")
    print(f"{'config':<40s} {'SUBMIT':>8s} {'no_tool':>8s} {'max_step':>9s} "
          f"{'error':>7s} {'other':>7s} {'n':>7s}")
    for c in configs:
        d = stats.get(c, {})
        tot = sum(s.get("_total", 0) for s in d.values())
        if tot == 0:
            print(f"{c:<40s} (no data)")
            continue
        agg = defaultdict(int)
        for src, srcd in d.items():
            for k, v in srcd.items():
                if k != "_total":
                    agg[k] += v
        def pct(k):
            return f"{100 * agg.get(k, 0) / tot:>6.1f}%"
        print(f"{c:<40s} {pct('SUBMIT'):>8s} {pct('NO_TOOL_CALL'):>8s} "
              f"{pct('MAX_STEPS'):>9s} {pct('ERROR'):>7s} {pct('OTHER'):>7s} {tot:>7d}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Per-config × per-source completion-rate diagnostic.")
    p.add_argument("--configs", required=True,
                   help="Comma-separated config dir names "
                        "(e.g. 'pro-high-brave-c1-p1-t1,sonnet-high-brave-c1-p1-t1')")
    p.add_argument("--exam", default=None,
                   help="Restrict to qids in data/exams/{exam}/. Required when "
                        "forecasts_raw/{config}/ contains trials from multiple "
                        "exams (e.g. smoke-llm + aibq2).")
    args = p.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    exam_qids = _exam_qids(args.exam)
    stats = {c: _gather(c, exam_qids) for c in configs}

    # Use union of sources actually present, in canonical order
    canonical = ["polymarket", "manifold", "metaculus", "infer",
                 "acled", "dbnomics", "fred", "wikipedia", "yfinance"]
    seen = set()
    for c in configs:
        seen.update(stats[c].keys())
    sources_order = [s for s in canonical if s in seen]
    extra = sorted(seen - set(canonical) - {"_total"})
    sources_order += extra

    _print_table(configs, stats, sources_order)
    _print_failure_breakdown(configs, stats)

    # Exit nonzero if any flag fires (useful for CI)
    any_flag = any(
        (sum(stats[c].get(s, {}).get("SUBMIT", 0) for s in sources_order)
         / max(1, sum(stats[c].get(s, {}).get("_total", 0) for s in sources_order))) < _FAIL_THRESHOLD
        for c in configs
    )
    return 1 if any_flag else 0


if __name__ == "__main__":
    sys.exit(main())
