"""CFB pool visualisations. Three separate figures so each can be read at
its own scale:

  pool_q.png         |Q(t)| stacked bars by source — when the agent must act.
  pool_r.png         |R(t)| stacked bars by source, outcome-split (o=1 up,
                     o=0 down) — when reward arrives.
  pool_lifelines.png Per-source swim lanes. Each source occupies one lane of
                     uniform height; within a lane its question lifelines
                     (segment from f_u to r_u) are spread to fill the lane,
                     coloured by outcome. Right-edge label shows event /
                     base counts. Rare sources (infer, metaculus) read as
                     clearly as common ones.

Usage:
    python -m src.cfb.plot_pool --pool data/cfb/pool-<sha>.jsonl
                                [--out-dir data/cfb] [--prefix pool_]
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


def _date_range(pool):
    t0 = min(min(e.f for e in pool), min(e.r for e in pool))
    t1 = max(max(e.f for e in pool), max(e.r for e in pool))
    return t0, t1, [t0 + timedelta(days=i) for i in range((t1 - t0).days + 1)]


def _stacked_by_source(by_day_src, sources, days):
    M = np.zeros((len(sources), len(days)), dtype=int)
    for j, d in enumerate(days):
        s = by_day_src.get(d, {})
        for i, src in enumerate(sources):
            M[i, j] = s.get(src, 0)
    return M


def _present_sources(pool):
    return [s for s in SOURCE_ORDER if any(e.source == s for e in pool)]


# --- Q(t) ---------------------------------------------------------------------

def plot_q(pool, out_path):
    sources = _present_sources(pool)
    _, _, days = _date_range(pool)
    by_day_src = defaultdict(lambda: defaultdict(int))
    for e in pool:
        by_day_src[e.f][e.source] += 1
    M = _stacked_by_source(by_day_src, sources, days)

    fig, ax = plt.subplots(figsize=(11, 4))
    bottoms = np.zeros(len(days))
    for i, src in enumerate(sources):
        ax.bar(days, M[i], bottom=bottoms, color=SOURCE_COLORS[src],
               width=1.0, label=src, edgecolor="none")
        bottoms += M[i]
    ax.set_ylabel("|Q(t)|  questions asked on day t")
    ax.set_xlabel("date")
    ax.set_title(f"Asked-day cadence — n={len(pool)} events", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False, title="source")
    _format_xaxis(ax)
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _format_xaxis(ax):
    """Monthly major ticks (labelled), weekly minor ticks (unlabelled)."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.SU))
    ax.grid(True, which="major", alpha=0.30, axis="x")
    ax.grid(True, which="minor", alpha=0.10, axis="x", linestyle="-")
    ax.tick_params(axis="x", which="minor", length=3)
    ax.tick_params(axis="x", which="major", length=6)


# --- R(t) ---------------------------------------------------------------------

def plot_r(pool, out_path):
    sources = _present_sources(pool)
    _, _, days = _date_range(pool)
    R_pos = defaultdict(lambda: defaultdict(int))
    R_neg = defaultdict(lambda: defaultdict(int))
    for e in pool:
        (R_pos if e.o == 1 else R_neg)[e.r][e.source] += 1
    Rpos_M = _stacked_by_source(R_pos, sources, days)
    Rneg_M = _stacked_by_source(R_neg, sources, days)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bottoms_pos = np.zeros(len(days))
    bottoms_neg = np.zeros(len(days))
    for i, src in enumerate(sources):
        ax.bar(days,  Rpos_M[i], bottom= bottoms_pos,
               color=SOURCE_COLORS[src], width=1.0, edgecolor="none",
               label=src)
        ax.bar(days, -Rneg_M[i], bottom=-bottoms_neg,
               color=SOURCE_COLORS[src], width=1.0, edgecolor="none",
               alpha=0.55)
        bottoms_pos += Rpos_M[i]
        bottoms_neg += Rneg_M[i]
    ax.axhline(0, color="black", lw=0.5)
    n_pos = sum(1 for e in pool if e.o == 1)
    n_neg = sum(1 for e in pool if e.o == 0)
    ax.set_ylabel(f"|R(t)|     o=1 ↑ ({n_pos})     o=0 ↓ ({n_neg})")
    ax.set_xlabel("date")
    ax.set_title("Resolution arrivals — outcomes split above/below zero",
                 fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, frameon=False, title="source")
    _format_xaxis(ax)
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --- per-source swim lanes ----------------------------------------------------

def plot_swimlanes(pool, out_path):
    """One horizontal lane per source. Each lane has uniform height so rare
    sources (infer, metaculus) read as clearly as common ones. Within a lane,
    questions are sorted by (f, r) and stacked vertically to fill the lane;
    each lifeline (f_u -> r_u) is coloured by outcome (green=1, red=0)."""
    sources = _present_sources(pool)
    t0, t1, _ = _date_range(pool)

    # Per-source entry lists, sorted (f, r, u)
    by_src = defaultdict(list)
    for e in pool:
        by_src[e.source].append(e)
    for src in by_src:
        by_src[src].sort(key=lambda e: (e.f, e.r, e.u))

    n_src = len(sources)
    fig, ax = plt.subplots(figsize=(12, 0.85 * n_src + 1.5))

    # lanes from top (i=0) to bottom (i=n_src-1) -> reverse for natural reading
    # We want first source in SOURCE_ORDER at top.
    sources_top_first = sources

    # Plot per-source
    for lane_idx, src in enumerate(sources_top_first):
        es = by_src[src]
        n = len(es)
        bases = len(set(e.meta["base_id"] for e in es))
        n_pos = sum(1 for e in es if e.o == 1)
        n_neg = n - n_pos

        # y-range for this lane: top edge is (n_src - lane_idx), bottom edge is (n_src - lane_idx - 1)
        y_top = n_src - lane_idx
        y_bot = y_top - 1
        # Lane background tint
        ax.axhspan(y_bot, y_top, color=SOURCE_COLORS[src], alpha=0.06, zorder=0)
        ax.axhline(y_bot, color="black", lw=0.3, alpha=0.35, zorder=1)

        # Spread n entries evenly inside [y_bot + pad, y_top - pad]
        pad = 0.06
        ys = np.linspace(y_bot + pad, y_top - pad, max(n, 1))

        # Draw lifelines coloured by outcome
        # Build separate batches for o=0 and o=1 to use vectorised plot
        seg_x_pos, seg_y_pos = [], []
        seg_x_neg, seg_y_neg = [], []
        for j, e in enumerate(es):
            if e.o == 1:
                seg_x_pos.extend([e.f, e.r, None])
                seg_y_pos.extend([ys[j], ys[j], None])
            else:
                seg_x_neg.extend([e.f, e.r, None])
                seg_y_neg.extend([ys[j], ys[j], None])

        lw = max(0.25, min(1.4, 1.4 / max(np.sqrt(n / 30.0), 1.0)))
        line_alpha = 0.55  # equal for both outcomes — was biased toward pos
        if seg_x_neg:
            ax.plot(seg_x_neg, seg_y_neg, color=OUTCOME_FALSE, lw=lw,
                    alpha=line_alpha, solid_capstyle="butt")
        if seg_x_pos:
            ax.plot(seg_x_pos, seg_y_pos, color=OUTCOME_TRUE, lw=lw,
                    alpha=line_alpha, solid_capstyle="butt")

        # Endpoint dots — give an unbiased outcome read regardless of horizon
        # length. Size adapts to lane density so dots don't blob together.
        dot_size = max(2.0, min(14.0, 22.0 / max(np.sqrt(n / 30.0), 1.0)))
        ep_x_pos, ep_y_pos, ep_x_neg, ep_y_neg = [], [], [], []
        for j, e in enumerate(es):
            if e.o == 1:
                ep_x_pos.append(e.r); ep_y_pos.append(ys[j])
            else:
                ep_x_neg.append(e.r); ep_y_neg.append(ys[j])
        if ep_x_neg:
            ax.scatter(ep_x_neg, ep_y_neg, s=dot_size, c=OUTCOME_FALSE,
                       marker="o", lw=0, alpha=0.9, zorder=3)
        if ep_x_pos:
            ax.scatter(ep_x_pos, ep_y_pos, s=dot_size, c=OUTCOME_TRUE,
                       marker="o", lw=0, alpha=0.9, zorder=4)

        # Source label on left margin
        ax.text(-0.005, (y_top + y_bot) / 2, src,
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10, fontweight="bold", color=SOURCE_COLORS[src])

        # Right edge: count tag — events / bases / +pos
        tag = f"n_events={n}   n_bases={bases}   o=1: {n_pos}  o=0: {n_neg}   π̂={n_pos/n:.2f}"
        ax.text(1.005, (y_top + y_bot) / 2, tag,
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=9, color="black", family="monospace")

    # Top boundary
    ax.axhline(n_src, color="black", lw=0.3, alpha=0.35)

    ax.set_xlim(t0 - timedelta(days=3), t1 + timedelta(days=3))
    ax.set_ylim(0, n_src)
    ax.set_yticks([])
    ax.set_xlabel("date")

    # Legend for outcome colors
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=OUTCOME_TRUE,  lw=3, label="o=1"),
        Line2D([0], [0], color=OUTCOME_FALSE, lw=3, label="o=0"),
    ]
    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(1.0, -0.06), ncol=2, frameon=False,
              fontsize=9)

    ax.set_title(f"Per-source question lifelines  (n={len(pool)} events, "
                 f"{len(set((e.source, e.meta['base_id']) for e in pool))} bases)",
                 fontsize=11)
    _format_xaxis(ax)
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --- driver ------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--out-dir", default=None,
                   help="output dir (defaults to dir of --pool)")
    p.add_argument("--prefix", default=None,
                   help="filename prefix (defaults to <pool basename>_)")
    args = p.parse_args()

    pool = load_pool(args.pool)
    if not pool:
        raise SystemExit("empty pool")

    out_dir = args.out_dir or os.path.dirname(args.pool) or "."
    base = args.prefix or os.path.basename(args.pool).replace(".jsonl", "_")

    plot_q(pool,         os.path.join(out_dir, f"{base}q.png"))
    plot_r(pool,         os.path.join(out_dir, f"{base}r.png"))
    plot_swimlanes(pool, os.path.join(out_dir, f"{base}lifelines.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
