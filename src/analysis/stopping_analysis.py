#!/usr/bin/env python3
"""stopping_analysis.py — Offline analysis of when the agent could have stopped.

Zero-new-LLM-calls diagnostic. Replays the per-step belief trajectories already
logged in experiments/forecasts_raw/{config}/{source}/{id}.json to answer:

  1. Brier-vs-budget        — if every question were capped at k steps, what is
                              the mean Brier? (Is the curve flat after step 2-3?)
  2. Oracle best-stop       — lower bound: stop each question at its realized-best
                              step. Bounds how much ANY stopping rule could gain.
  3. Belief movement        — mean |p_k - p_{k-1}| per step; how fast beliefs
                              converge (≈ how much VOI is left late in the loop).
  4. Stopping frontier      — offline stand-in for a VOI controller. Stop at the
                              first step where the belief moved less than tau
                              (a one-sample proxy for Var_X(b') < threshold).
                              Sweeping tau traces the (mean #steps, mean Brier)
                              Pareto frontier — pick your operating point off it
                              instead of converting "cost of a search" to Brier.

The residual-Brier identity behind (4): for belief b, the expected Brier of
reporting honestly is b(1-b), and one search's expected reduction is Var_X(b').
Offline we observe the realized squared move (p_{k} - p_{k-1})^2 as a one-sample
estimate of that variance, so "stop when recent moves are small" ≈ "stop when
VOI is low."

Usage:
    python3 src/analysis/stopping_analysis.py --configs flash-high-brave-c1-t1
    python3 src/analysis/stopping_analysis.py \
        --configs flash-high-brave-c1-t1,pro-high-brave-c1-p1-t1 --exam tranche-a
    python3 src/analysis/stopping_analysis.py --configs ... --no-plots
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config.paths import FORECASTS_DIR, EXAMS_DIR

# Sources we expect as subdirectories under a config (everything else, e.g.
# `trial_*` or `config.json`, is skipped).
KNOWN_SOURCES = {
    "acled", "dbnomics", "fred", "infer", "manifold", "metaculus",
    "polymarket", "wikipedia", "yfinance", "aibq2",
}


def clamp(p, lo, hi):
    return max(lo, min(hi, p))


def load_exam_stems(exam):
    """Return the set of question-id stems in an exam, or None for no filter."""
    path = os.path.join(EXAMS_DIR, exam, "indices.json")
    if not os.path.exists(path):
        sys.exit(f"Exam indices not found: {path}")
    idx = json.load(open(path))
    stems = set()
    for ids in idx.values():
        stems.update(ids)
    return stems


def normalize_outcomes(resolved_to):
    """Coerce resolved_to (scalar or list, possibly with Nones) to a float list."""
    if resolved_to is None:
        return None
    vals = resolved_to if isinstance(resolved_to, list) else [resolved_to]
    out = [float(v) for v in vals if v is not None]
    return out or None


def load_trajectories(config, exam_stems, lo, hi):
    """Yield one record per resolved question with its per-step belief & Brier.

    record = {
        ps:      [p_0(prior), p_1, ..., p_T]   per-step belief (scalar)
        briers:  [B_0, ..., B_T]               Brier if stopped at that step
        source, stem, n_steps
    }
    """
    config_dir = os.path.join(FORECASTS_DIR, config)
    if not os.path.isdir(config_dir):
        sys.exit(f"No such config dir: {config_dir}")

    records = []
    n_skipped_unresolved = 0
    n_skipped_static = 0
    for source in sorted(os.listdir(config_dir)):
        if source not in KNOWN_SOURCES:
            continue  # skip trial_* dirs, config.json, etc.
        for f in sorted(glob.glob(os.path.join(config_dir, source, "*.json"))):
            stem = os.path.splitext(os.path.basename(f))[0]
            if exam_stems is not None and stem not in exam_stems:
                continue
            try:
                d = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            y = normalize_outcomes(d.get("resolved_to"))
            if y is None:
                n_skipped_unresolved += 1
                continue
            bh = d.get("belief_history", [])
            ps = [b.get("p", 0.5) for b in bh]
            if len(ps) < 2:
                # Only a prior logged (e.g. immediate failure) — not informative.
                continue
            briers = [float(np.mean([(clamp(p, lo, hi) - yj) ** 2 for yj in y]))
                      for p in ps]
            if max(ps) - min(ps) < 1e-9:
                n_skipped_static += 1  # belief never moved (e.g. nobelief config)
            records.append({
                "ps": ps, "briers": briers, "y": y,
                "source": source, "stem": stem, "n_steps": len(ps) - 1,
            })
    return records, n_skipped_unresolved, n_skipped_static


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def brier_vs_budget(records, k_max):
    """Mean Brier if every question were capped at exactly k steps (pad short
    trajectories with their last belief). Returns (ks, mean_brier, n_active)."""
    ks = list(range(k_max + 1))
    means, n_active = [], []
    for k in ks:
        vals, active = [], 0
        for r in records:
            kk = min(k, len(r["briers"]) - 1)
            if k <= r["n_steps"]:
                active += 1
            vals.append(r["briers"][kk])
        means.append(float(np.mean(vals)))
        n_active.append(active)
    return ks, means, n_active


def oracle_stop(records):
    """Per-question realized-best stopping step. Returns summary dict."""
    oracle_briers, oracle_ks, full_briers, full_ks = [], [], [], []
    for r in records:
        b = r["briers"]
        k_star = int(np.argmin(b))
        oracle_briers.append(b[k_star])
        oracle_ks.append(k_star)
        full_briers.append(b[-1])      # what production actually did (stop at end)
        full_ks.append(r["n_steps"])
    return {
        "oracle_brier": float(np.mean(oracle_briers)),
        "oracle_steps": float(np.mean(oracle_ks)),
        "full_brier": float(np.mean(full_briers)),
        "full_steps": float(np.mean(full_ks)),
        "oracle_k_dist": oracle_ks,
    }


def belief_movement(records, k_max):
    """Mean |p_k - p_{k-1}| at each step k, and frac of questions still moving."""
    deltas_by_k = {k: [] for k in range(1, k_max + 1)}
    for r in records:
        ps = r["ps"]
        for k in range(1, len(ps)):
            if k <= k_max:
                deltas_by_k[k].append(abs(ps[k] - ps[k - 1]))
    ks = sorted(deltas_by_k)
    mean_delta = [float(np.mean(deltas_by_k[k])) if deltas_by_k[k] else 0.0
                  for k in ks]
    frac_moving = [float(np.mean([dd > 1e-6 for dd in deltas_by_k[k]]))
                   if deltas_by_k[k] else 0.0 for k in ks]
    return ks, mean_delta, frac_moving


def stopping_frontier(records, taus, k_min=1):
    """Offline VOI stand-in (movement rule): stop at first step k>=k_min whose
    move |p_k - p_{k-1}| < tau.

    Returns list of (tau, mean_steps_used, mean_brier)."""
    frontier = []
    for tau in taus:
        steps_used, briers = [], []
        for r in records:
            ps, b = r["ps"], r["briers"]
            stop_k = len(ps) - 1  # default: ran to the end
            for k in range(max(1, k_min), len(ps)):
                if abs(ps[k] - ps[k - 1]) < tau:
                    stop_k = k
                    break
            steps_used.append(stop_k)
            briers.append(b[stop_k])
        frontier.append((tau, float(np.mean(steps_used)), float(np.mean(briers))))
    return frontier


def uncertainty_frontier(records, thetas, k_min=1):
    """Offline test of the paper's eqn-74 rule, which for binary+Brier reduces to
    VOI ≈ ρ·b(1−b): stop at first step k>=k_min where b_k(1−b_k) < theta
    (i.e. once the belief is confident enough). theta = c_z/ρ is the single
    tuned threshold. Returns list of (theta, mean_steps_used, mean_brier)."""
    frontier = []
    for theta in thetas:
        steps_used, briers = [], []
        for r in records:
            ps, b = r["ps"], r["briers"]
            stop_k = len(ps) - 1
            for k in range(max(1, k_min), len(ps)):
                pk = ps[k]
                if pk * (1.0 - pk) < theta:
                    stop_k = k
                    break
            steps_used.append(stop_k)
            briers.append(b[stop_k])
        frontier.append((theta, float(np.mean(steps_used)), float(np.mean(briers))))
    return frontier


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(config, records, n_unresolved, n_static, k_max, taus, thetas, out_dir, plots):
    n = len(records)
    print(f"\n{'=' * 70}\nCONFIG: {config}")
    print(f"{'=' * 70}")
    print(f"resolved questions analyzed : {n}")
    print(f"  skipped (unresolved)      : {n_unresolved}")
    if n_static:
        print(f"  ⚠ static belief (never moved): {n_static} "
              f"(stopping analysis is vacuous for these — nobelief config?)")
    if n == 0:
        print("No resolved trajectories — nothing to analyze.")
        return None

    # 1. Brier vs budget
    ks, means, n_active = brier_vs_budget(records, k_max)
    print(f"\n[1] Brier vs step budget  (cap every question at k steps)")
    print(f"    {'k':>3}  {'mean Brier':>10}  {'Δ vs k-1':>9}  {'# still active':>14}")
    for i, k in enumerate(ks):
        d = "" if i == 0 else f"{means[i] - means[i-1]:+.4f}"
        print(f"    {k:>3}  {means[i]:>10.4f}  {d:>9}  {n_active[i]:>14}")

    # 2. Oracle
    o = oracle_stop(records)
    print(f"\n[2] Oracle best-stop (lower bound on any stopping rule)")
    print(f"    production (stop at end) : Brier {o['full_brier']:.4f}  "
          f"@ {o['full_steps']:.2f} steps")
    print(f"    oracle  (best per-q stop): Brier {o['oracle_brier']:.4f}  "
          f"@ {o['oracle_steps']:.2f} steps")
    gain = o['full_brier'] - o['oracle_brier']
    saved = o['full_steps'] - o['oracle_steps']
    print(f"    ceiling: Brier −{gain:.4f}  AND  {saved:.2f} fewer steps "
          f"({100*saved/max(o['full_steps'],1e-9):.0f}% cheaper)")
    dist = np.bincount(o["oracle_k_dist"], minlength=k_max + 1)
    print(f"    oracle stop-step histogram: "
          + " ".join(f"k{kk}:{c}" for kk, c in enumerate(dist) if c))

    # 3. Movement
    mks, mdelta, fmov = belief_movement(records, k_max)
    print(f"\n[3] Belief movement per step")
    print(f"    {'k':>3}  {'mean |Δp|':>10}  {'frac moving':>11}")
    for k, md, fm in zip(mks, mdelta, fmov):
        print(f"    {k:>3}  {md:>10.4f}  {fm:>11.2f}")

    # 4. Movement-rule frontier (|Δp| < tau)
    fr = stopping_frontier(records, taus)
    print(f"\n[4] Movement-rule frontier (stop when |Δp| < tau)")
    print(f"    {'tau':>6}  {'mean steps':>10}  {'mean Brier':>10}  "
          f"{'Δ Brier vs prod':>15}")
    for tau, st, br in fr:
        print(f"    {tau:>6.3f}  {st:>10.2f}  {br:>10.4f}  "
              f"{br - o['full_brier']:>+15.4f}")

    # 5. Eqn-74 uncertainty-rule frontier (b(1−b) < theta ≈ ρ·b(1−b) VOI rule)
    uf = uncertainty_frontier(records, thetas)
    print(f"\n[5] Eqn-74 frontier (stop when b(1−b) < theta;  θ = c_z/ρ)")
    print(f"    {'theta':>6}  {'mean steps':>10}  {'mean Brier':>10}  "
          f"{'Δ Brier vs prod':>15}")
    for th, st, br in uf:
        print(f"    {th:>6.3f}  {st:>10.2f}  {br:>10.4f}  "
              f"{br - o['full_brier']:>+15.4f}")

    if plots:
        _make_plots(config, ks, means, o, mks, mdelta, fr, uf, out_dir)

    return {"config": config, "n": n, "brier_vs_budget": (ks, means),
            "oracle": o, "frontier": fr, "uncertainty_frontier": uf}


def _make_plots(config, ks, means, o, mks, mdelta, fr, uf, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(ks, means, "o-", label="cap at k steps")
    ax[0].axhline(o["oracle_brier"], ls="--", color="green",
                  label=f"oracle ({o['oracle_brier']:.3f})")
    ax[0].axhline(o["full_brier"], ls=":", color="gray",
                  label=f"production ({o['full_brier']:.3f})")
    ax[0].set_xlabel("step budget k"); ax[0].set_ylabel("mean Brier")
    ax[0].set_title("Brier vs budget"); ax[0].legend(fontsize=8)

    ax[1].plot(mks, mdelta, "o-", color="purple")
    ax[1].set_xlabel("step k"); ax[1].set_ylabel("mean |Δp|")
    ax[1].set_title("Belief movement per step")

    ax[2].plot([f[1] for f in fr], [f[2] for f in fr], "o-", color="darkorange",
               label="|Δp|<tau (movement)")
    ax[2].plot([f[1] for f in uf], [f[2] for f in uf], "s-", color="teal",
               label="b(1−b)<θ (eqn 74)")
    ax[2].axhline(o["oracle_brier"], ls="--", color="green", label="oracle")
    ax[2].axhline(o["full_brier"], ls=":", color="gray", label="production")
    ax[2].set_xlabel("mean steps used"); ax[2].set_ylabel("mean Brier")
    ax[2].set_title("Stopping frontiers (lower-left = better)")
    ax[2].legend(fontsize=8)

    fig.suptitle(f"Stopping analysis — {config}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f"stopping_{config}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n    plot → {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True,
                    help="comma-separated config dir names under forecasts_raw/")
    ap.add_argument("--exam", default=None,
                    help="restrict to question stems in this exam (data/exams/{exam})")
    ap.add_argument("--k-max", type=int, default=10, help="max step budget to tabulate")
    ap.add_argument("--clamp", type=float, default=0.05,
                    help="probability clamp (matches production [clamp, 1-clamp])")
    ap.add_argument("--taus", default="0.0,0.005,0.01,0.02,0.05,0.1,0.2",
                    help="comma-separated tau grid for the |Δp| movement-rule frontier")
    ap.add_argument("--thetas", default="0.0,0.04,0.09,0.16,0.21,0.24,0.25",
                    help="comma-separated theta grid for the eqn-74 b(1−b) frontier")
    ap.add_argument("--out", default=os.path.join("experiments", "analysis", "stopping"),
                    help="output dir for plots")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    lo, hi = args.clamp, 1.0 - args.clamp
    exam_stems = load_exam_stems(args.exam) if args.exam else None
    taus = [float(t) for t in args.taus.split(",")]
    thetas = [float(t) for t in args.thetas.split(",")]

    for config in args.configs.split(","):
        config = config.strip()
        records, n_unres, n_static = load_trajectories(config, exam_stems, lo, hi)
        report(config, records, n_unres, n_static, args.k_max, taus, thetas,
               args.out, plots=not args.no_plots)


if __name__ == "__main__":
    main()
