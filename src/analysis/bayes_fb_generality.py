#!/usr/bin/env python3
"""bayes_fb_generality.py — Direct-BLF vs online explicit-Bayes across several
ForecastBench questions, K trials each, to check whether the single-question
overconfidence-taming effect generalizes.

The main agent LLM (--llm) and the likelihood LLM used for summarize+typicality
(--bllm) are separate, so we can run a strong agent (e.g. Pro, GPT-5) with a cheap
likelihood model (flash). For market questions the Bayes prior is anchored on the
market price (matching the offline market-anchored setup).

Usage:
    caffeinate -s python3 src/analysis/bayes_fb_generality.py --llm flash --bllm flash --trials 5
    caffeinate -s python3 src/analysis/bayes_fb_generality.py --llm pro --bllm flash --trials 5 --workers 4
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import dotenv
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
except ImportError:
    pass

import numpy as np

from agent.agent import run_agent
from config.config import parse_config

# Curated diverse tranche-A set (mix of market-right and market-wrong outcomes).
DEFAULT_QS = [
    "metaculus/39771_2025-10-26",      # out=1, mv=0.04  (market WRONG)
    "metaculus/21531_2025-10-26",      # out=0, mv=0.03
    "manifold/Zn0R05lcyR_2025-10-26",  # out=0, mv=0.17
    "infer/1563_2025-10-26",           # out=0, mv=0.43
    "polymarket/0x35aaaa9b77b25834ddf3647bd4c34376ba3f1dcd87e3d4801c829b3658b15b82_2025-03-02",  # out=0, mv=0.095
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=",".join(DEFAULT_QS),
                    help="comma list of source/id (resolves to data/questions/source/id.json)")
    ap.add_argument("--llm", default="flash", help="main agent LLM")
    ap.add_argument("--bllm", default="flash", help="likelihood LLM (summarize+typicality)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--crowd", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join("experiments", "analysis", "bayes_fb"))
    args = ap.parse_args()

    qfiles = []
    for spec in args.questions.split(","):
        spec = spec.strip()
        f = os.path.join("data", "questions", spec + ".json")
        if os.path.exists(f):
            qfiles.append(f)
        else:
            print(f"  (skip, not found) {f}")
    questions = [json.load(open(f)) for f in qfiles]
    os.makedirs(args.out, exist_ok=True)

    base = f"{args.llm}/thk:high/crowd:{args.crowd}/tools:0/steps:{args.steps}"
    cfgs = {"direct": parse_config(base),
            "bayes": parse_config(f"{base}/bayes:1/balpha:{args.alpha}/bllm:{args.bllm}")}
    for name, c in cfgs.items():
        c.name = name

    print(f"main LLM={args.llm}  likelihood LLM={args.bllm}  K={args.trials}  "
          f"crowd={args.crowd}  n_questions={len(questions)}", flush=True)

    # all (question, method, trial) runs, executed concurrently
    specs = [(qi, name, t) for qi in range(len(questions))
             for name in cfgs for t in range(1, args.trials + 1)]

    def short_stem(qi):
        return os.path.splitext(os.path.basename(qfiles[qi]))[0].split("_")[0][:14]

    def run_one(spec):
        qi, name, t = spec
        stem = os.path.splitext(os.path.basename(qfiles[qi]))[0]
        odir = os.path.join(args.out, args.llm, stem, name, f"trial_{t}")
        t0 = time.time()
        res = run_agent(questions[qi], cfgs[name], odir, verbose=False)
        return qi, name, t, res["forecast"], res.get("n_steps"), time.time() - t0

    # results[qi][name] = list of K finals
    results = [{"direct": [], "bayes": []} for _ in questions]
    done = 0
    t_start = time.time()
    print(f"launching {len(specs)} runs, {args.workers} concurrent "
          f"(each run is a full agent loop) ...", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in cf.as_completed([ex.submit(run_one, s) for s in specs]):
            qi, name, t, fc, nsteps, dt = fut.result()
            results[qi][name].append(fc)
            done += 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (len(specs) - done)
            print(f"  [{done:>2}/{len(specs)}] {short_stem(qi):<14} {name:<6} "
                  f"tr{t}: p={fc:.3f} ({nsteps} steps, {dt:.0f}s)  "
                  f"| elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m", flush=True)

    # Per-question + aggregate Brier (on the K-trial mean forecast)
    def brier(p, y):
        return (p - y) ** 2
    print(f"\n{'question':<26} {'out':>3} {'mv':>5} "
          f"{'direct p':>9} {'B_dir':>6} {'bayes p':>9} {'B_bay':>6} {'ΔB(bay-dir)':>11}")
    print("-" * 92)
    agg = {"direct": [], "bayes": []}
    for qi, q in enumerate(questions):
        y = float(q.get("resolved_to"))
        try:
            mv = float(q.get("market_value"))
        except (TypeError, ValueError):
            mv = float("nan")
        dp = float(np.mean(results[qi]["direct"])); bp = float(np.mean(results[qi]["bayes"]))
        bd, bb = brier(dp, y), brier(bp, y)
        agg["direct"].append(bd); agg["bayes"].append(bb)
        stem = os.path.splitext(os.path.basename(qfiles[qi]))[0][:24]
        print(f"{stem:<26} {int(y):>3} {mv:>5.2f} "
              f"{dp:>9.3f} {bd:>6.3f} {bp:>9.3f} {bb:>6.3f} {bb-bd:>+11.3f}")
    md, mb = float(np.mean(agg["direct"])), float(np.mean(agg["bayes"]))
    wins = sum(1 for i in range(len(questions)) if agg["bayes"][i] < agg["direct"][i])
    print("-" * 92)
    print(f"{'MEAN Brier':<26} {'':>3} {'':>5} {'':>9} {md:>6.3f} {'':>9} {mb:>6.3f} "
          f"{mb-md:>+11.3f}   (bayes better on {wins}/{len(questions)})")

    # Scatter: per-question direct vs bayes Brier
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], ls=":", color="gray")
    for qi, q in enumerate(questions):
        stem = os.path.splitext(os.path.basename(qfiles[qi]))[0].split("_")[0][:10]
        ax.scatter(agg["direct"][qi], agg["bayes"][qi], s=70, zorder=5)
        ax.annotate(stem, (agg["direct"][qi], agg["bayes"][qi]), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    ax.scatter([md], [mb], marker="*", s=260, color="red", zorder=6, label="mean")
    ax.set_xlabel("direct Brier (lower better)"); ax.set_ylabel("bayes Brier")
    ax.set_title(f"FB generality: {args.llm} agent, {args.bllm} likelihood, K={args.trials}\n"
                 f"below diagonal = Bayes better", fontsize=10)
    ax.legend(fontsize=9); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    path = os.path.join(args.out, f"fb_generality_{args.llm}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\nplot -> {path}")


if __name__ == "__main__":
    main()
