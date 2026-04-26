#!/usr/bin/env python3
"""plot_exams.py — Generate data visualizations for exam question sets.

Produces scatter plots, histograms, and tag distributions for a given exam.
These are the same plots previously embedded in eval_plots.py, but factored
out since they operate on exam data, not forecasts.

Usage:
    python3 src/plot_exams.py --name aibq2-all
    python3 src/plot_exams.py --name tranche-a1
    python3 src/plot_exams.py --name tranche-a1 --name tranche-b1  # multiple

Outputs go to data/exams/{name}/:
    rdate_by_fdate_scatter.png  — Resolution date vs forecast date
    horizon_histogram.png       — Forecast horizon distribution
    tag_{version}_distribution.png — Topic tag distribution (if tags exist)
"""

import argparse
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Tag category ordering (for consistent heatmap layout)
# ---------------------------------------------------------------------------

_TAG_CATEGORIES = {
    "kevin": [
        "Geopolitics & Conflict",
        "Domestic Politics",
        "Financial Markets",
        "Macroeconomics",
        "Weather & Climate",
        "Science & Technology",
        "Sports & Entertainment",
        "Health & Biology",
        "Business & Industry",
        "Society & Law",
    ],
}


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def generate_tag_distribution(exam_name, tags, tag_version="kevin",
                              n_total=None):
    """Generate source x category heatmap with marginal totals.

    Saved to data/exams/{exam}/tag_{version}_distribution.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy.ma as ma
    except ImportError:
        return None

    if not tags:
        return None

    sources = sorted({src for src, _ in tags})
    all_cats = set(tags.values())
    known_order = _TAG_CATEGORIES.get(tag_version, [])
    cats_seen = [c for c in known_order if c in all_cats]
    extras = sorted(all_cats - set(cats_seen))
    cats_seen += extras

    n_cats = len(cats_seen)
    n_src = len(sources)
    matrix = np.zeros((n_cats, n_src), dtype=int)
    for (src, _), cat in tags.items():
        if cat in cats_seen and src in sources:
            matrix[cats_seen.index(cat), sources.index(src)] += 1

    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    grand_total = matrix.sum()

    aug = np.zeros((n_cats + 1, n_src + 1), dtype=int)
    aug[:n_cats, :n_src] = matrix
    aug[:n_cats, n_src] = row_totals
    aug[n_cats, :n_src] = col_totals
    aug[n_cats, n_src] = grand_total

    main_mask = np.ones_like(aug, dtype=bool)
    main_mask[:n_cats, :n_src] = False
    marginal_mask = ~main_mask

    fig_w = max(8, n_src * 1.8 + 3.5)
    fig_h = max(5, n_cats * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    main_data = ma.masked_where(main_mask, aug)
    im = ax.imshow(main_data, aspect="auto", cmap="YlOrRd")

    marginal_data = ma.masked_where(marginal_mask, aug)
    ax.imshow(marginal_data, aspect="auto", cmap="Greys", alpha=0.15,
              vmin=0, vmax=1)

    ax.axhline(n_cats - 0.5, color="#999", linewidth=1.5)
    ax.axvline(n_src - 0.5, color="#999", linewidth=1.5)

    y_labels = cats_seen + ["Total"]
    x_labels = sources + ["Total"]
    ax.set_xticks(range(n_src + 1))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_cats + 1))
    ax.set_yticklabels(y_labels, fontsize=8)
    for label in ax.get_xticklabels():
        if label.get_text() == "Total":
            label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        if label.get_text() == "Total":
            label.set_fontweight("bold")

    vmax = matrix.max() if matrix.max() > 0 else 1
    for i in range(n_cats + 1):
        for j in range(n_src + 1):
            val = aug[i, j]
            if val == 0:
                continue
            is_marginal = (i == n_cats or j == n_src)
            if is_marginal:
                color, weight = "black", "bold"
            else:
                color = "white" if val > vmax * 0.6 else "black"
                weight = "normal"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)

    n_tagged = len(tags)
    if n_total and n_total > n_tagged:
        title_n = f"n={n_tagged} of {n_total} tagged"
    else:
        title_n = f"n={n_tagged}"
    ax.set_title(f"Tag distribution — {tag_version} ({exam_name}, {title_n})",
                 fontsize=9, pad=8)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Count", pad=0.02)

    exam_dir = os.path.join("data", "exams", exam_name)
    out_path = os.path.join(exam_dir, f"tag_{tag_version}_distribution.png")
    os.makedirs(exam_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_rdate_by_fdate_scatter(exam_name, exam):
    """Scatter plot of resolution date vs forecast due date, colored by outcome.

    Saved to data/exams/{exam}/rdate_by_fdate_scatter.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        import numpy as np
    except ImportError:
        return None

    rows = []
    for source, ids in exam.items():
        for qid in ids:
            safe_id = re.sub(r'[/\\:]', '_', str(qid))
            q_path = os.path.join("data", "questions", source, f"{safe_id}.json")
            if not os.path.exists(q_path):
                continue
            with open(q_path) as f:
                q = json.load(f)
            fdd = q.get("forecast_due_date", "")
            if not fdd:
                continue
            rdates = q.get("resolution_dates", [])
            rdate = q.get("resolution_date", "")
            resolved_to = q.get("resolved_to")
            if rdates and isinstance(rdates, list):
                outcomes = resolved_to if isinstance(resolved_to, list) else [resolved_to]
                for i, rd in enumerate(rdates):
                    o = outcomes[i] if i < len(outcomes) else None
                    rows.append({"source": source, "fdate": fdd, "rdate": rd, "outcome": o})
            elif rdate:
                o = resolved_to[0] if isinstance(resolved_to, list) else resolved_to
                rows.append({"source": source, "fdate": fdd, "rdate": rdate, "outcome": o})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["fdate"] = pd.to_datetime(df["fdate"], errors="coerce")
    df["rdate"] = pd.to_datetime(df["rdate"], errors="coerce")
    df = df.dropna(subset=["fdate", "rdate"])

    n = len(df)
    npos = int((df["outcome"] == 1.0).sum())
    nneg = int((df["outcome"] == 0.0).sum())

    fig, ax = plt.subplots(figsize=(10, 7))

    rng = np.random.default_rng(42)
    jitter_days = 0.8
    fdate_jitter = pd.to_timedelta(rng.uniform(-jitter_days, jitter_days, len(df)), unit="D")
    rdate_jitter = pd.to_timedelta(rng.uniform(-jitter_days, jitter_days, len(df)), unit="D")
    df = df.copy()
    df["fdate_j"] = df["fdate"] + fdate_jitter
    df["rdate_j"] = df["rdate"] + rdate_jitter

    mask_false = df["outcome"] == 0.0
    mask_true = df["outcome"] == 1.0
    mask_unk = ~(mask_false | mask_true)

    if mask_unk.any():
        ax.scatter(df.loc[mask_unk, "fdate_j"], df.loc[mask_unk, "rdate_j"],
                   c="gray", alpha=0.4, label=f"Unresolved ({n - npos - nneg})", s=12, edgecolors="none")
    if mask_false.any():
        ax.scatter(df.loc[mask_false, "fdate_j"], df.loc[mask_false, "rdate_j"],
                   c="red", alpha=0.5, label=f"False ({nneg})", s=12, edgecolors="none")
    if mask_true.any():
        ax.scatter(df.loc[mask_true, "fdate_j"], df.loc[mask_true, "rdate_j"],
                   c="green", alpha=0.5, label=f"True ({npos})", s=12, edgecolors="none")

    fpad = max((df["fdate"].max() - df["fdate"].min()) * 0.05, pd.Timedelta(days=7))
    rpad = max((df["rdate"].max() - df["rdate"].min()) * 0.05, pd.Timedelta(days=7))
    xmin, xmax = df["fdate"].min() - fpad, df["fdate"].max() + fpad
    ymin, ymax = df["rdate"].min() - rpad, df["rdate"].max() + rpad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    diag_min = max(xmin, ymin)
    diag_max = min(xmax, ymax)
    if diag_min < diag_max:
        ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', alpha=0.2, label="horizon=0")

    ax.grid(True, alpha=0.15)
    date_range = max(xmax - xmin, ymax - ymin)
    if date_range < pd.Timedelta(days=60):
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.yaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    elif date_range < pd.Timedelta(days=180):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.yaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax.set_xlabel("Forecast due date")
    ax.set_ylabel("Resolution date")
    ax.set_title(f"Resolution date vs Forecast date — {exam_name}\n"
                 f"n={n}, True={npos}, False={nneg}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()

    exam_dir = os.path.join("data", "exams", exam_name)
    os.makedirs(exam_dir, exist_ok=True)
    out_path = os.path.join(exam_dir, "rdate_by_fdate_scatter.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_horizon_histogram(exam_name, exam):
    """Histogram of forecast horizons, colored by outcome.

    Saved to data/exams/{exam}/horizon_histogram.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
    except ImportError:
        return None

    rows = []
    for source, ids in exam.items():
        for qid in ids:
            safe_id = re.sub(r'[/\\:]', '_', str(qid))
            q_path = os.path.join("data", "questions", source, f"{safe_id}.json")
            if not os.path.exists(q_path):
                continue
            with open(q_path) as f:
                q = json.load(f)
            fdd = q.get("forecast_due_date", "")
            if not fdd:
                continue
            rdates = q.get("resolution_dates", [])
            rdate = q.get("resolution_date", "")
            resolved_to = q.get("resolved_to")
            if rdates and isinstance(rdates, list):
                outcomes = resolved_to if isinstance(resolved_to, list) else [resolved_to]
                for i, rd in enumerate(rdates):
                    o = outcomes[i] if i < len(outcomes) else None
                    rows.append({"source": source, "fdate": fdd, "rdate": rd, "outcome": o})
            elif rdate:
                o = resolved_to[0] if isinstance(resolved_to, list) else resolved_to
                rows.append({"source": source, "fdate": fdd, "rdate": rdate, "outcome": o})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["fdate"] = pd.to_datetime(df["fdate"], errors="coerce")
    df["rdate"] = pd.to_datetime(df["rdate"], errors="coerce")
    df = df.dropna(subset=["fdate", "rdate"])
    df["horizon_days"] = (df["rdate"] - df["fdate"]).dt.days

    n = len(df)
    npos = int((df["outcome"] == 1.0).sum())
    nneg = int((df["outcome"] == 0.0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                              gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    bins = np.arange(0, df["horizon_days"].max() + 14, 14)

    mask_true = df["outcome"] == 1.0
    mask_false = df["outcome"] == 0.0
    mask_unresolved = ~(mask_true | mask_false)
    n_unres = int(mask_unresolved.sum())

    data_list, labels, colors = [], [], []
    if mask_false.any():
        data_list.append(df.loc[mask_false, "horizon_days"].values)
        labels.append(f"False ({nneg})")
        colors.append("#e74c3c")
    if mask_true.any():
        data_list.append(df.loc[mask_true, "horizon_days"].values)
        labels.append(f"True ({npos})")
        colors.append("#27ae60")
    if mask_unresolved.any():
        data_list.append(df.loc[mask_unresolved, "horizon_days"].values)
        labels.append(f"Unresolved ({n_unres})")
        colors.append("#7f8c8d")

    if data_list:
        ax.hist(data_list, bins=bins, stacked=True, color=colors, label=labels,
                edgecolor="white", linewidth=0.5, alpha=0.8)
    else:
        ax.text(0.5, 0.5, "no horizon data",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("Number of questions")
    ax.legend(fontsize=8)
    ax.set_title(f"Forecast horizon distribution — {exam_name}\n"
                 f"n={n}, True={npos}, False={nneg} "
                 f"({npos/(npos+nneg)*100:.0f}% positive)" if (npos+nneg) > 0
                 else f"Forecast horizon — {exam_name} (n={n})",
                 fontsize=10)

    ax2 = axes[1]
    source_counts = df.groupby("source").size().sort_values()
    source_pos = df[df["outcome"] == 1.0].groupby("source").size()
    source_neg = df[df["outcome"] == 0.0].groupby("source").size()

    y_pos = range(len(source_counts))
    neg_vals = [source_neg.get(s, 0) for s in source_counts.index]
    pos_vals = [source_pos.get(s, 0) for s in source_counts.index]

    ax2.barh(list(y_pos), neg_vals, color="#e74c3c", alpha=0.8, label="False")
    ax2.barh(list(y_pos), pos_vals, left=neg_vals, color="#27ae60", alpha=0.8, label="True")
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(source_counts.index, fontsize=8)
    ax2.set_xlabel("Count")
    ax2.set_title("By source", fontsize=10)
    ax2.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    exam_dir = os.path.join("data", "exams", exam_name)
    os.makedirs(exam_dir, exist_ok=True)
    out_path = os.path.join(exam_dir, "horizon_histogram.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_exam_indices(exam_name):
    """Load exam indices from data/exams/{name}/indices.json."""
    path = os.path.join("data", "exams", exam_name, "indices.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: exam not found: {path}")
    with open(path) as f:
        return json.load(f)


def plot_exam(exam_name):
    """Generate all plots for one exam."""
    indices = load_exam_indices(exam_name)
    total = sum(len(v) for v in indices.values())
    print(f"Plotting {exam_name}: {total} questions across {len(indices)} sources")

    from config.tags import get_tags_for_exam, discover_classified_spaces

    p = generate_rdate_by_fdate_scatter(exam_name, indices)
    if p:
        print(f"  {p}")

    p = generate_horizon_histogram(exam_name, indices)
    if p:
        print(f"  {p}")

    for ls in discover_classified_spaces():
        tags = get_tags_for_exam(indices, ls)
        if tags:
            p = generate_tag_distribution(exam_name, tags, tag_version=ls,
                                          n_total=total)
            if p:
                print(f"  {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate data visualizations for exam question sets.")
    parser.add_argument("--name", required=True, action="append",
                        help="Exam name(s) to plot (can specify multiple)")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    for name in args.name:
        plot_exam(name)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
