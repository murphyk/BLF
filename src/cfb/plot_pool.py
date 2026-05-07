"""Three-panel timeseries view of a CFB pool.

  top    |Q(t)|  stacked bars by source  — when the agent must act
  middle |R(t)|  stacked bars by source, outcome-split
                 (positive bars = o=1, negative bars = o=0) — when reward arrives
  bottom lifelines: each question is a horizontal segment from f_u to r_u,
                    sorted (f, r), colored by source, with a green/red dot at
                    the resolution endpoint. Visualises the asynchronous
                    reward-delay structure.

Usage:
    python -m src.cfb.plot_pool --pool data/cfb/pool-<sha>.jsonl --out data/cfb/plot_pool.png
"""

from __future__ import annotations
import argparse
import os
from collections import defaultdict
from datetime import date, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .pool import load_pool


# Stable colour map. Markets in a warm palette, datasets in a cool palette.
SOURCE_ORDER = [
    "polymarket", "manifold", "metaculus", "infer",
    "yfinance", "fred", "dbnomics", "wikipedia", "acled",
]
SOURCE_COLORS = {
    "polymarket": "#d62728",
    "manifold":   "#ff7f0e",
    "metaculus":  "#bcbd22",
    "infer":      "#e377c2",
    "yfinance":   "#1f77b4",
    "fred":       "#17becf",
    "dbnomics":   "#2ca02c",
    "wikipedia":  "#9467bd",
    "acled":      "#8c564b",
}
OUTCOME_TRUE  = "#2ca02c"
OUTCOME_FALSE = "#d62728"


def _stacked_by_source(by_day_src: dict[date, dict[str, int]],
                       sources: list[str], days: list[date]):
    """Returns 2-D array (n_sources, n_days) for stacked bars."""
    M = np.zeros((len(sources), len(days)), dtype=int)
    for j, d in enumerate(days):
        s = by_day_src.get(d, {})
        for i, src in enumerate(sources):
            M[i, j] = s.get(src, 0)
    return M


def make_figure(pool_path: str, out_path: str) -> None:
    pool = load_pool(pool_path)
    if not pool:
        raise SystemExit("empty pool")

    sources = [s for s in SOURCE_ORDER if any(e.source == s for e in pool)]

    # Daily counts
    Q_by_day_src: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    R_pos: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    R_neg: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in pool:
        Q_by_day_src[e.f][e.source] += 1
        if e.o == 1:
            R_pos[e.r][e.source] += 1
        else:
            R_neg[e.r][e.source] += 1

    t0 = min(min(e.f for e in pool), min(e.r for e in pool))
    t1 = max(max(e.f for e in pool), max(e.r for e in pool))
    days = [t0 + timedelta(days=i) for i in range((t1 - t0).days + 1)]

    Q_M    = _stacked_by_source(Q_by_day_src, sources, days)
    Rpos_M = _stacked_by_source(R_pos, sources, days)
    Rneg_M = _stacked_by_source(R_neg, sources, days)

    # --- figure ---
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [1.0, 1.4, 2.5], "hspace": 0.32},
        sharex=True,
    )
    ax_q, ax_r, ax_lf = axes

    xs = days

    # --- top: |Q(t)| stacked bars ---
    bottoms = np.zeros(len(days))
    for i, src in enumerate(sources):
        ax_q.bar(xs, Q_M[i], bottom=bottoms, color=SOURCE_COLORS[src],
                 width=1.0, label=src, edgecolor="none")
        bottoms += Q_M[i]
    ax_q.set_ylabel("|Q(t)|  (asked)")
    ax_q.set_title(f"CFB pool — {os.path.basename(pool_path)}  "
                   f"(n={len(pool)}, sources={len(sources)})", fontsize=11)
    ax_q.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                fontsize=8, frameon=False, title="source")

    # --- middle: |R(t)|, positive up (green tint), negative down (red tint) ---
    bottoms_pos = np.zeros(len(days))
    bottoms_neg = np.zeros(len(days))
    for i, src in enumerate(sources):
        ax_r.bar(xs, Rpos_M[i], bottom=bottoms_pos,
                 color=SOURCE_COLORS[src], width=1.0,
                 edgecolor="none")
        ax_r.bar(xs, -Rneg_M[i], bottom=-bottoms_neg,
                 color=SOURCE_COLORS[src], width=1.0,
                 edgecolor="none", alpha=0.55)
        bottoms_pos += Rpos_M[i]
        bottoms_neg += Rneg_M[i]
    ax_r.axhline(0, color="black", lw=0.5)
    ax_r.set_ylabel("|R(t)|   o=1↑   o=0↓")

    # --- bottom: lifelines ---
    pool_sorted = sorted(pool, key=lambda e: (e.f, e.r, e.u))
    n = len(pool_sorted)
    # vectorised collection by (source, outcome) for speed
    for src in sources:
        xs_seg, ys_seg = [], []
        for idx, e in enumerate(pool_sorted):
            if e.source != src:
                continue
            xs_seg.extend([e.f, e.r, None])
            ys_seg.extend([idx, idx, None])
        if xs_seg:
            ax_lf.plot(xs_seg, ys_seg, color=SOURCE_COLORS[src], lw=0.4,
                       alpha=0.55, solid_capstyle="butt", label=src)

    # outcome dots at right endpoint (larger so they read against line density)
    pos_x = [e.r for e in pool_sorted if e.o == 1]
    pos_y = [i for i, e in enumerate(pool_sorted) if e.o == 1]
    neg_x = [e.r for e in pool_sorted if e.o == 0]
    neg_y = [i for i, e in enumerate(pool_sorted) if e.o == 0]
    ax_lf.scatter(neg_x, neg_y, s=3.5, c=OUTCOME_FALSE, marker="o", lw=0,
                  label=f"o=0 ({len(neg_x)})", alpha=0.85, zorder=3)
    ax_lf.scatter(pos_x, pos_y, s=3.5, c=OUTCOME_TRUE,  marker="o", lw=0,
                  label=f"o=1 ({len(pos_x)})", alpha=0.85, zorder=4)

    # forecast-due-date guides
    fdds = sorted(set(e.f for e in pool_sorted))
    for fdd in fdds:
        ax_lf.axvline(fdd, color="black", lw=0.3, alpha=0.15, zorder=0)

    ax_lf.set_ylim(-2, n + 2)
    ax_lf.set_ylabel("question (sorted by f, r)")
    ax_lf.set_xlabel("date")
    ax_lf.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                 fontsize=8, frameon=False, markerscale=3,
                 title="outcome")

    # x-axis
    ax_lf.xaxis.set_major_locator(mdates.MonthLocator())
    ax_lf.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for ax in axes:
        ax.grid(True, alpha=0.25, axis="x")
    fig.autofmt_xdate()

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or args.pool.replace(".jsonl", ".png")
    make_figure(args.pool, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
