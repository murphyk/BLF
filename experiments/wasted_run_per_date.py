#!/usr/bin/env python3
"""wasted_run_per_date.py — per-date BI and resolution-rate trends.

Companion to wasted_run_analysis.py. Uses the same 2 trials × 9399
questions × 14 ask dates from the 2026-04-26 over-run, and produces:

  figs/per_date_bi.png            BI per source × ask date (lines)
  figs/per_date_resolved.png      number of resolved entries per source × date
  figs/per_date_summary.txt       text dump of the same numbers

Looks for trends across the timeline:
  - is BI lower for ask dates AFTER the model's knowledge cutoff
    (Gemini-3.1-Pro: 2025-01-31)?
  - does the resolved-question count taper as ask date approaches the
    present (less time for resolution)?
"""

import os, json, math, sys
from collections import defaultdict

CONFIG = "pro-high-brave-c1-p1-t1"
TRIALS = [1, 2]
RAW = f"experiments/forecasts_raw/{CONFIG}"
KNOWLEDGE_CUTOFF = "2025-01-31"
TRANCHE_A_DATE = "2025-10-26"
TRANCHE_B_DATE = "2025-11-09"
OUT_DIR = "experiments/wasted_run_figs"


def _logit(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit_mean(ps):
    if not ps:
        return None
    return _sigmoid(sum(_logit(p) for p in ps) / len(ps))


def bi_from_bs(bs):
    if not bs:
        return None
    return 100 * (1 - math.sqrt(max(0.0, sum(bs) / len(bs))))


def gather():
    print(f"Reading wasted-run trials {TRIALS}...")
    by_q = defaultdict(dict)
    for t in TRIALS:
        td = f"{RAW}/trial_{t}"
        if not os.path.isdir(td):
            continue
        for src in os.listdir(td):
            sp = os.path.join(td, src)
            if not os.path.isdir(sp):
                continue
            for fn in os.listdir(sp):
                if fn.endswith(".json"):
                    with open(os.path.join(sp, fn)) as f:
                        by_q[(src, fn.removesuffix(".json"))][t] = json.load(f)

    records = []
    for (src, qid), trials in by_q.items():
        if len(trials) < 2:
            continue
        ref = next(iter(trials.values()))
        rdates = ref.get("resolution_dates") or [ref.get("resolution_date", "")]
        outcomes = ref.get("resolved_to")
        if not isinstance(outcomes, list):
            outcomes = [outcomes]
        fdd = (ref.get("forecast_due_date", "") or "")[:10]
        for i in range(len(rdates)):
            ps = []
            for fc in trials.values():
                fl = fc.get("forecasts")
                if isinstance(fl, list) and i < len(fl) and fl[i] is not None:
                    ps.append(float(fl[i]))
                elif i == 0 and fc.get("forecast") is not None:
                    ps.append(float(fc["forecast"]))
            o = outcomes[i] if i < len(outcomes) else None
            if not ps or o is None:
                continue
            records.append({"source": src, "fdd": fdd,
                            "p": logit_mean(ps), "o": float(o)})
    return records


def main():
    records = gather()
    print(f"  total resolved entries: {len(records):,}")

    # Indexed: bs[source][fdd] = list of BS values
    bs = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        bs[r["source"]][r["fdd"]].append((r["p"] - r["o"]) ** 2)
        counts[r["source"]][r["fdd"]] += 1

    sources = sorted(bs)
    # Filter to FB-cycle ask dates only — drop ones that contain ONLY
    # aibq2 entries (AIBQ2 has its own daily ask dates that pollute the
    # per-date FB analysis).
    fb_sources = set(sources) - {"aibq2"}
    dates = sorted({
        r["fdd"] for r in records
        if r["source"] in fb_sources
    })

    # Per-date overall (equal-weighted mean of market vs dataset BI)
    market = {"infer", "manifold", "metaculus", "polymarket"}
    overall_bi = {}
    for d in dates:
        m_bs = [b for s in market for b in bs[s].get(d, [])]
        d_bs = [b for s in (set(sources) - market) for b in bs[s].get(d, [])]
        m_bi = bi_from_bs(m_bs)
        d_bi = bi_from_bs(d_bs)
        ov = ((m_bi + d_bi) / 2) if (m_bi is not None and d_bi is not None) else (m_bi or d_bi)
        overall_bi[d] = (m_bi, d_bi, ov, len(m_bs), len(d_bs))

    os.makedirs(OUT_DIR, exist_ok=True)

    # Text dump
    txt_lines = ["per-date summary  (BI = 100·(1 - sqrt(mean BS)))",
                 f"knowledge cutoff: {KNOWLEDGE_CUTOFF}", ""]
    txt_lines.append(f"{'date':<12} {'mkt-BI':>7} {'dat-BI':>7} {'overall':>8}  "
                     f"{'n-mkt':>6} {'n-dat':>6}")
    for d in dates:
        m_bi, d_bi, ov, nm, nd = overall_bi[d]
        marker = "" if d > KNOWLEDGE_CUTOFF else "  ←pre-cutoff"
        txt_lines.append(f"{d:<12} "
                         f"{m_bi if m_bi is not None else 0:>7.2f} "
                         f"{d_bi if d_bi is not None else 0:>7.2f} "
                         f"{ov if ov is not None else 0:>8.2f}  "
                         f"{nm:>6,d} {nd:>6,d}{marker}")
    txt_lines.append("")
    txt_lines.append("per-source BI per date:")
    txt_lines.append(f"{'source':<12} " +
                     " ".join(f"{d[5:]:>7}" for d in dates))
    for s in sources:
        line = f"{s:<12} "
        for d in dates:
            v = bi_from_bs(bs[s].get(d, []))
            line += f"{v if v is not None else 0:>7.2f} "
        txt_lines.append(line)
    txt_lines.append("")
    txt_lines.append("per-source resolved counts per date:")
    txt_lines.append(f"{'source':<12} " +
                     " ".join(f"{d[5:]:>7}" for d in dates) + "   total")
    for s in sources:
        per = [counts[s].get(d, 0) for d in dates]
        line = f"{s:<12} " + " ".join(f"{c:>7,d}" for c in per)
        line += f"   {sum(per):>6,d}"
        txt_lines.append(line)

    txt = "\n".join(txt_lines)
    with open(f"{OUT_DIR}/per_date_summary.txt", "w") as f:
        f.write(txt + "\n")
    print(txt)
    print()

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from datetime import datetime
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    xs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    cutoff = datetime.strptime(KNOWLEDGE_CUTOFF, "%Y-%m-%d")
    tranche_a = datetime.strptime(TRANCHE_A_DATE, "%Y-%m-%d")
    tranche_b = datetime.strptime(TRANCHE_B_DATE, "%Y-%m-%d")

    # Plot 1: BI per source over time
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(sources):
        ys = [bi_from_bs(bs[s].get(d, [])) for d in dates]
        # plot only points where we have data
        x_valid = [x for x, y in zip(xs, ys) if y is not None]
        y_valid = [y for y in ys if y is not None]
        ax.plot(x_valid, y_valid, marker="o", linewidth=1.5,
                markersize=5, label=s, color=cmap(i))
    # Overall BI as a thick black line
    ov_xs, ov_ys = [], []
    for x, d in zip(xs, dates):
        ov = overall_bi[d][2]
        if ov is not None:
            ov_xs.append(x); ov_ys.append(ov)
    ax.plot(ov_xs, ov_ys, color="black", linewidth=2.5,
            marker="s", markersize=6, label="overall (eq.-wt mkt/dat)")
    # Shaded band covering the two tranche dates (A=2025-10-26, B=2025-11-09)
    # so the eye can read off how the paper-evaluation tranches sit
    # relative to the broader corpus.
    ax.axvspan(tranche_a, tranche_b, color="gold", alpha=0.18, zorder=0,
               label=f"tranche A∪B ({TRANCHE_A_DATE} → {TRANCHE_B_DATE})")
    ax.axvline(tranche_a, color="goldenrod", linestyle="-",
               alpha=0.7, linewidth=1.0)
    ax.axvline(tranche_b, color="goldenrod", linestyle="-",
               alpha=0.7, linewidth=1.0)
    ax.axvline(cutoff, color="gray", linestyle="--", alpha=0.6,
               label=f"knowledge cutoff ({KNOWLEDGE_CUTOFF})")
    ax.set_xlabel("Forecast due date")
    ax.set_ylabel("Brier Index (×100)")
    ax.set_title(f"BLF SOTA per-source BI by ask date  ({CONFIG}, mean of 2 trials)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    p1 = f"{OUT_DIR}/per_date_bi.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print(f"wrote {p1}")

    # Plot 2: resolved count per source over time
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, s in enumerate(sources):
        ys = [counts[s].get(d, 0) for d in dates]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=5,
                label=s, color=cmap(i))
    overall_counts = [sum(counts[s].get(d, 0) for s in sources) for d in dates]
    ax.plot(xs, overall_counts, color="black", linewidth=2.5,
            marker="s", markersize=6, label="overall")
    ax.axvspan(tranche_a, tranche_b, color="gold", alpha=0.18, zorder=0,
               label=f"tranche A∪B ({TRANCHE_A_DATE} → {TRANCHE_B_DATE})")
    ax.axvline(tranche_a, color="goldenrod", linestyle="-",
               alpha=0.7, linewidth=1.0)
    ax.axvline(tranche_b, color="goldenrod", linestyle="-",
               alpha=0.7, linewidth=1.0)
    ax.axvline(cutoff, color="gray", linestyle="--", alpha=0.6,
               label=f"knowledge cutoff ({KNOWLEDGE_CUTOFF})")
    ax.set_xlabel("Forecast due date")
    ax.set_ylabel("Resolved (source, qid, res_date) entries")
    ax.set_title(f"Resolution coverage by ask date  ({CONFIG}, mean of 2 trials)")
    ax.set_yscale("symlog", linthresh=10)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    p2 = f"{OUT_DIR}/per_date_resolved.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    print(f"wrote {p2}")


if __name__ == "__main__":
    sys.exit(main())
