#!/usr/bin/env python3
"""bayes_online_demo.py — Compare direct-BLF vs online explicit-Bayes belief
evolution on a single question (default: the AIBQ2 Gulf-of-America question 0048
from the belief-trace figure).

Runs K trials of each method on the same question — direct (the LLM emits the
probability) vs config.bayes_update (the belief p is driven by the online
per-state likelihood update, conditioning matches the offline `sqhbcond` variant)
— and plots both belief-trajectory fans so we can compare the path, the actions,
the stopping point, and the *inter-trial variance* (the quantity that motivates
trial aggregation in the main paper).

Usage:
    caffeinate -s python3 src/analysis/bayes_online_demo.py --trials 5
    caffeinate -s python3 src/analysis/bayes_online_demo.py --trials 5 --llm flash --alpha 0.35 --workers 4
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import dotenv
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
except ImportError:
    pass

import numpy as np

from agent.agent import run_agent
from config.config import parse_config


def _probs(result):
    return [b.get("p") for b in result.get("belief_history", []) if isinstance(b, dict)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qfile", default="data/questions/aibq2/aibq2_0048.json")
    ap.add_argument("--llm", default="flash")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent agent runs (keep modest to respect search rate limits)")
    ap.add_argument("--out", default=os.path.join("experiments", "analysis", "bayes_demo"))
    args = ap.parse_args()

    question = json.load(open(args.qfile))
    os.makedirs(args.out, exist_ok=True)
    base = f"{args.llm}/thk:high/crowd:0/tools:0/steps:{args.steps}"
    outcome = question.get("resolved_to")
    outcome = float(outcome) if outcome is not None else None

    methods = {
        "direct": parse_config(base),
        "bayes": parse_config(f"{base}/bayes:1/balpha:{args.alpha}"),
    }
    for name, cfg in methods.items():
        cfg.name = name

    print(f"Question: {question.get('question')}")
    print(f"Outcome: {outcome}  (cutoff {question.get('forecast_due_date')})")
    print(f"K={args.trials} trials/method, llm={args.llm}, alpha={args.alpha}\n")

    # Build the 2K run specs and execute concurrently (run_agent is I/O-bound).
    specs = [(name, t) for name in methods for t in range(1, args.trials + 1)]

    def run_one(spec):
        name, t = spec
        odir = os.path.join(args.out, name, f"trial_{t}")
        res = run_agent(question, methods[name], odir, verbose=False)
        return name, t, res

    results = {name: {} for name in methods}
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in cf.as_completed([ex.submit(run_one, s) for s in specs]):
            name, t, res = fut.result()
            results[name][t] = res
            done += 1
            print(f"  [{done}/{len(specs)}] {name} trial {t}: "
                  f"p={res['forecast']:.3f} n_steps={res['n_steps']}", flush=True)

    # Stats + Brier
    print(f"\n{'method':<8} {'finals':<34} {'mean':>6} {'std σ':>7} "
          f"{'Brier(mean)':>11} {'mean Brier':>10}")
    print("-" * 80)
    stats = {}
    for name in methods:
        finals = [results[name][t]["forecast"] for t in sorted(results[name])]
        mean = float(np.mean(finals)); std = float(np.std(finals))
        briers = [(f - outcome) ** 2 for f in finals] if outcome is not None else []
        brier_mean = (mean - outcome) ** 2 if outcome is not None else float("nan")
        stats[name] = {"finals": finals, "mean": mean, "std": std,
                       "brier_mean": brier_mean,
                       "mean_brier": float(np.mean(briers)) if briers else float("nan")}
        fstr = "[" + ", ".join(f"{f:.2f}" for f in finals) + "]"
        print(f"{name:<8} {fstr:<34} {mean:>6.3f} {std:>7.3f} "
              f"{brier_mean:>11.3f} {stats[name]['mean_brier']:>10.3f}")

    # mean per-step cross-trial std (over the common step range)
    print()
    for name in methods:
        traj = [_probs(results[name][t]) for t in sorted(results[name])]
        L = min(len(p) for p in traj)
        arr = np.array([p[:L] for p in traj])
        step_std = arr.std(axis=0)
        print(f"{name}: mean cross-trial std over steps 1..{L-1} = "
              f"{float(step_std[1:].mean()):.3f}")

    # Two-panel belief-fan figure (Figure-4 style), shared axes
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, name in zip(axes, methods):
        for i, t in enumerate(sorted(results[name])):
            ps = _probs(results[name][t])
            ax.plot(range(len(ps)), ps, "o-", color=colors[i % len(colors)],
                    markersize=4, linewidth=1.4, alpha=0.85,
                    label=f"trial {t} (p={results[name][t]['forecast']:.2f})")
        ax.axhline(stats[name]["mean"], ls="--", color="black", alpha=0.6,
                   label=f"mean = {stats[name]['mean']:.2f}")
        if outcome is not None:
            ax.axhline(outcome, ls=":", color="gray", alpha=0.7,
                       label=f"outcome = {int(outcome)}")
        ax.set_xlabel("agent step"); ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{name}  (σ = {stats[name]['std']:.2f})")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("belief probability $p_t$")
    fig.suptitle(f"Belief evolution, {args.trials} trials/method (α={args.alpha}): "
                 f"{question.get('question','')[:60]}...", fontsize=11)
    fig.tight_layout()
    path = os.path.join(args.out, f"belief_fan_k{args.trials}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nplot -> {path}")


if __name__ == "__main__":
    main()
