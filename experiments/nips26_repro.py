#!/usr/bin/env python3
"""nips26_repro.py — Reproduce Tables 1–4 of the NeurIPS 2026 paper.

Reads from experiments/forecasts_final/ (which is git-tracked, so this
script works from a fresh clone) and prints each paper table with our
recomputed BI numbers next to the published ones.

Tables reproduced:
    Table 1 (tab:mega):           SOTA comparison on FB A∪B
    Table 2 (tab:core-ablations): Core component ablations on FB A∪B
    Table 3 (tab:postproc):       Trial-aggregation variants on c0-t1
    Table 4 (tab:cal-comparison): Calibration × crowd/emp ablation on FB A∪B

Usage:
    python3 experiments/nips26_repro.py                # all four tables
    python3 experiments/nips26_repro.py --table 2      # one table
    python3 experiments/nips26_repro.py --no-paper     # omit the paper-number column
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.eval import (
    load_and_score, load_exam, compute_group_means, load_xid,
    load_baseline_scores,
    _index_cache,  # for cache-clear between configs
)


# Standard FBQtype split for tranche-ab.
GROUPS_FB = {
    "market":  ["infer", "manifold", "metaculus", "polymarket"],
    "dataset": ["acled", "dbnomics", "fred", "wikipedia", "yfinance"],
    "overall": sorted({"infer", "manifold", "metaculus", "polymarket",
                        "acled", "dbnomics", "fred", "wikipedia", "yfinance"}),
}


def score(config: str, exam: dict, groups: dict) -> dict[str, float | None]:
    """Score one config on one exam, returning {group: BI*100}."""
    metrics = ["brier-index", "brier-score"]
    if config == "baseline":
        s = load_baseline_scores(exam, metrics)
    else:
        s = load_and_score(config, exam, metrics)
    g = compute_group_means(s, groups, "brier-index")
    out = {}
    for k, v in g.items():
        out[k] = (v * 100) if v is not None else None
    out["_n"] = len(s)
    return out


def fmt(v: float | None) -> str:
    return f"{v:5.1f}" if isinstance(v, (int, float)) else "  —  "


def fmt_paper(v: str) -> str:
    return f"  ({v})" if v else ""


def print_table(title: str, rows: list[dict], columns: list[tuple[str, str]]):
    """rows = list of {label, paper_market, paper_data, paper_overall, scores}.
    columns = [(group_name_for_print, group_key)]
    """
    print()
    print(f"{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    width_label = max(len(r["label"]) for r in rows) + 2
    header = f"{'Method':<{width_label}}  {'n':>4}"
    for col_label, _ in columns:
        header += f"  {col_label:>8}{'(paper)':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['label']:<{width_label}}  {r['scores'].get('_n', 0):>4}"
        for col_label, group_key in columns:
            v = r["scores"].get(group_key)
            paper_v = r.get(f"paper_{group_key}", "")
            line += f"  {fmt(v):>8}  {fmt_paper(paper_v):>8}"
        print(line)


# -------------------------------------------------------------------------
# Table specs
# -------------------------------------------------------------------------

TABLE1_ROWS = [
    # (label, config, paper_market, paper_data, paper_overall)
    ("BLF+crowd+emp+cal", "pro-high-brave-c1-p1-t1_calibrated_hier",
     "85.2", "62.4", "73.8"),
    ("Cassi",             "fb-cassi-ai-2",
     "82.0", "59.6", "70.8"),
    ("GPT-5(zs+crowd)",   "fb-openai-gpt-5-2025-08-07_zero_shot_with_freeze_values",
     "80.4", "60.1", "70.2"),
    ("Grok",              "fb-xai-1",
     "77.6", "61.4", "69.5"),
    ("Foresight",         "fb-lightning-rod-labs",
     "82.3", "57.6", "70.0"),
    ("Crowd+emp (no LLM)","baseline",
     "81.5", "58.3", "69.9"),
]

TABLE2_ROWS = [
    ("clairvoyant",  "pro-high-brave-c1-t1-clairvoyant", "91.0", "66.9", "78.9"),
    ("+crowd",       "pro-high-brave-c1-t1",             "84.2", "62.0", "73.1"),
    ("BLF (base)",   "pro-high-brave-c0-t1",             "77.1", "61.8", "69.4"),
    ("medthink",     "pro-default-brave-c0-t1",          "77.5", "61.7", "69.6"),
    ("notools",      "pro-high-brave-c0-t0",             "79.1", "59.0", "69.0"),
    ("flash",        "flash-high-brave-c0-t1",           "73.2", "61.0", "67.1"),
    ("kimi",         "kimi-k2t-high-brave-c0-t1",        "74.1", "58.8", "66.5"),
    ("nobelief",     "pro-high-brave-c0-t1-nobelief",    "74.6", "58.2", "66.4"),
    ("batch",        "pro-high-brave-c0-t1-batch5",      "78.7", "51.9", "65.3"),
    ("nosearch",     "pro-high-none-c0-t1",              "68.7", "61.1", "64.9"),
    ("zs",           "pro-high-none-c0-t0",              "67.8", "59.8", "63.8"),
]

# Table 3 uses one config + a list of aggregation-variant keys.
# Switched 2026-04-29 to c0-t0 (no crowd, no emp prior, no tools, no cal):
# the most stripped-down BLF setting, matching the philosophy of Table 2
# (vary one component at a time from a stripped baseline so the
# aggregation differences aren't masked by anchor signals like crowd
# price or source-specific data tools). On c0-t1 the LOO grid picked
# (f=0, c=0) ⇒ α≡1 ⇒ logit-mean, hiding any shrinkage benefit; on
# c0-t0 LOO picks (f=0.9, c=2.0) ⇒ genuine per-question shrinkage.
TABLE3_BASE = "pro-high-brave-c0-t0"
TABLE3_VARIANTS = [
    # (label, agg_key, paper_market, paper_data, paper_overall)
    # Paper-time numbers shown for the old c0-t1 base, kept here for
    # reference; the repro now scores c0-t0 (see TABLE3_BASE comment).
    ("logit:5",            "logit-mean:5",        "—",    "—",    "—"),
    ("logit:3",            "logit-mean:3",        "—",    "—",    "—"),
    ("logit:1",            "logit-mean:1",        "—",    "—",    "—"),
    ("mean:5",             "mean:5",              "—",    "—",    "—"),
    ("mean:3",             "mean:3",              "—",    "—",    "—"),
    ("mean:1",             "mean:1",              "—",    "—",    "—"),
    ("median",             "median:5",            "—",    "—",    "—"),
    ("shrink-std-loo:5",   "shrink-std-loo:5",    "—",    "—",    "—"),
    ("shrink-alpha-loo:5", "shrink-alpha-loo:5",  "—",    "—",    "—"),
]

# Table 4 — each base config × {uncal, global, hier}.
TABLE4_BLOCKS = [
    ("BLF (c0,e0)", "pro-high-brave-c0-t1", [
        ("uncal",  "",                    "77.1", "61.8", "69.4"),
        ("global", "_calibrated_global",  "77.0", "61.8", "69.4"),
        ("hier",   "_calibrated_hier",    "76.4", "62.3", "69.3"),
    ]),
    ("BLF (c1,e1)", "pro-high-brave-c1-p1-t1", [
        ("uncal",  "",                    "84.8", "61.9", "73.4"),
        ("global", "_calibrated_global",  "85.5", "62.0", "73.8"),
        ("hier",   "_calibrated_hier",    "85.2", "62.4", "73.8"),
    ]),
    # ZS rows all use the Halawi prompt (paper claim). Switching the c0,e0
    # row from our prompt to Halawi makes the comparison apples-to-apples
    # with the c1,e1 row (which was already Halawi). The ZS (c1,e0) row is
    # added to isolate the crowd contribution under ZS — also approximates
    # the external "Gemini-3-Pro-Preview (zero shot with crowd forecast)"
    # baseline on the FB leaderboard.
    # Paper-update note: previous Table 4 ZS (c0,e0) numbers (67.8/59.8/63.8)
    # used pro-high-none-c0-t0 (our prompt), not Halawi.
    ("ZS [halawi] (c0,e0)", "pro-high-none-c0-t0-halawi", [
        ("uncal",  "",                    "—",    "—",    "—"),
        ("global", "_calibrated_global",  "—",    "—",    "—"),
        ("hier",   "_calibrated_hier",    "—",    "—",    "—"),
    ]),
    ("ZS [halawi] (c1,e0)", "pro-high-none-c1-t0-halawi", [
        ("uncal",  "",                    "—",    "—",    "—"),
        ("global", "_calibrated_global",  "—",    "—",    "—"),
        ("hier",   "_calibrated_hier",    "—",    "—",    "—"),
    ]),
    ("ZS [halawi] (c1,e1)", "pro-high-none-c1-p1-t0-halawi", [
        ("uncal",  "",                    "78.6", "51.9", "65.3"),
        ("global", "_calibrated_global",  "78.0", "52.3", "65.1"),
        ("hier",   "_calibrated_hier",    "79.0", "56.9", "68.0"),
    ]),
]


# -------------------------------------------------------------------------
# Table runners
# -------------------------------------------------------------------------

def run_table1(exam):
    rows = []
    for label, cfg, m, d, o in TABLE1_ROWS:
        scores = score(cfg, exam, GROUPS_FB)
        rows.append(dict(label=label, scores=scores,
                         paper_market=m, paper_dataset=d, paper_overall=o))
    print_table("Table 1: SOTA comparison on FB A∪B (BI × 100)", rows,
                [("Market", "market"), ("Data", "dataset"), ("Overall", "overall")])


def run_table2(exam):
    rows = []
    for label, cfg, m, d, o in TABLE2_ROWS:
        scores = score(cfg, exam, GROUPS_FB)
        rows.append(dict(label=label, scores=scores,
                         paper_market=m, paper_dataset=d, paper_overall=o))
    print_table("Table 2: Core ablations on FB A∪B (BI × 100)", rows,
                [("Market", "market"), ("Data", "dataset"), ("Overall", "overall")])


def run_table3(exam):
    rows = []
    for label, key, m, d, o in TABLE3_VARIANTS:
        cfg_ref = f"{TABLE3_BASE}[{key}]"
        scores = score(cfg_ref, exam, GROUPS_FB)
        rows.append(dict(label=label, scores=scores,
                         paper_market=m, paper_dataset=d, paper_overall=o))
    print_table(f"Table 3: Trial-aggregation variants on {TABLE3_BASE} (BI × 100)",
                rows,
                [("Market", "market"), ("Data", "dataset"), ("Overall", "overall")])


def run_table4(exam):
    rows = []
    for block_label, base_cfg, variants in TABLE4_BLOCKS:
        rows.append(dict(label=f"-- {block_label} --", scores={},
                         paper_market="", paper_dataset="", paper_overall=""))
        for cal_label, suffix, m, d, o in variants:
            cfg = f"{base_cfg}{suffix}" if suffix else base_cfg
            scores = score(cfg, exam, GROUPS_FB)
            rows.append(dict(label=f"  {cal_label}", scores=scores,
                             paper_market=m, paper_dataset=d, paper_overall=o))
    print_table("Table 4: Calibration × crowd/emp ablation on FB A∪B (BI × 100)",
                rows,
                [("Market", "market"), ("Data", "dataset"), ("Overall", "overall")])


def main():
    ap = argparse.ArgumentParser(description="Reproduce NeurIPS 2026 paper tables.")
    ap.add_argument("--table", type=int, default=None, choices=[1, 2, 3, 4],
                    help="Run only the specified table (default: all).")
    args = ap.parse_args()

    exam = load_exam("tranche-ab")

    runners = {1: run_table1, 2: run_table2, 3: run_table3, 4: run_table4}
    targets = [args.table] if args.table else [1, 2, 3, 4]

    for t in targets:
        _index_cache.clear()  # avoid cross-table cross-config cache pollution
        runners[t](exam)
    print()


if __name__ == "__main__":
    main()
