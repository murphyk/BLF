"""Stateful-vs-stateless reward-curve plot.

Reads one or more trajectory JSONL files (produced by `run.py
--trajectory-out`) and plots Brier-index-style reward curves over instance
index. Goal: show whether a stateful agent's reward improves over time
relative to a stateless baseline, in the spirit of
https://continual-learning-bench.com/news/cl-bench-1-0/.

Two views in the figure:
  • cumulative reward — running mean Brier index up to instance i
                        (smooth, shows asymptotic performance)
  • rolling-window reward — Brier index over the last K instances
                            (volatile but shows local improvement)

Usage:
    python -m src.cfb.plot_continual \
        --traj <name>=<path.jsonl> ... \
        --out compare.png  [--window 50]
"""

from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_traj(path: str) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def cumulative_brier(traj: list[dict]) -> np.ndarray:
    bs = np.array([(float(row["p"]) - float(row["o"])) ** 2 for row in traj], dtype=float)
    if not len(bs):
        return bs
    return np.cumsum(bs) / np.arange(1, len(bs) + 1)


def rolling_brier(traj: list[dict], window: int) -> np.ndarray:
    bs = np.array([(float(row["p"]) - float(row["o"])) ** 2 for row in traj], dtype=float)
    if not len(bs):
        return bs
    out = np.empty(len(bs))
    for i in range(len(bs)):
        lo = max(0, i + 1 - window)
        out[i] = bs[lo:i + 1].mean()
    return out


def brier_index(brier: np.ndarray) -> np.ndarray:
    return 1.0 - 4.0 * brier


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traj", action="append", required=True,
                   help="<label>=<path.jsonl>; repeat for multiple agents. "
                        "Append `:stateless` to dash the line.")
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--metric", default="brier_index",
                   choices=["brier_index", "brier"])
    p.add_argument("--ylim", default=None,
                   help="comma-separated lo,hi (default auto-clip to focus "
                        "on steady state)")
    p.add_argument("--warmup", type=int, default=20,
                   help="hide the first N instances on the rolling-window "
                        "panel — early indices are dominated by single-event "
                        "outliers")
    args = p.parse_args()

    runs = []
    for spec in args.traj:
        if "=" not in spec:
            raise SystemExit(f"--traj needs <label>=<path>, got: {spec}")
        label, path = spec.split("=", 1)
        is_stateless = label.endswith(":stateless")
        if is_stateless:
            label = label[:-len(":stateless")]
        runs.append((label, path, is_stateless, load_traj(path)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    ax_cum, ax_roll = axes

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(runs))))
    for i, (label, _, stateless, traj) in enumerate(runs):
        if not traj:
            continue
        x = np.arange(1, len(traj) + 1)
        cum = cumulative_brier(traj)
        roll = rolling_brier(traj, window=args.window)
        if args.metric == "brier_index":
            cum, roll = brier_index(cum), brier_index(roll)
        c = colors[i]
        ls = "--" if stateless else "-"
        lw_solid = 1.5 if stateless else 2.0
        ax_cum.plot(x, cum, lw=lw_solid, ls=ls, color=c, label=label)
        # rolling: hide warmup zone
        mask = np.arange(len(roll)) >= args.warmup
        ax_roll.plot(x[mask], roll[mask], lw=lw_solid * 0.8, ls=ls, color=c,
                     alpha=0.95, label=label)

    ylab = "Brier index  (1 − 4·Brier;  higher is better)" \
        if args.metric == "brier_index" else "Brier  (lower is better)"
    for ax in axes:
        ax.set_xlabel("instance index (resolved events, chronological)")
        ax.grid(True, alpha=0.25)
    ax_cum.set_ylabel(ylab)
    ax_cum.set_title("Cumulative reward")
    ax_roll.set_title(f"Rolling-window reward (k={args.window}, "
                      f"warmup={args.warmup} hidden)")
    ax_cum.legend(loc="lower right", fontsize=8, frameon=False)
    if args.metric == "brier_index":
        for ax in axes:
            ax.axhline(0.0, color="gray", lw=0.5, ls="--", alpha=0.6)

    if args.ylim:
        lo, hi = (float(x) for x in args.ylim.split(","))
        for ax in axes:
            ax.set_ylim(lo, hi)
    else:
        # Auto-clip to the rolling-window range, ignoring early outliers
        all_roll = []
        for _, _, _, traj in runs:
            r = rolling_brier(traj, window=args.window)
            if args.metric == "brier_index":
                r = brier_index(r)
            if len(r) > args.warmup:
                all_roll.extend(r[args.warmup:])
        if all_roll:
            lo, hi = float(np.min(all_roll)), float(np.max(all_roll))
            pad = 0.05 * max(0.1, hi - lo)
            for ax in axes:
                ax.set_ylim(lo - pad, hi + pad)

    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
