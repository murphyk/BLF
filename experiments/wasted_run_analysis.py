#!/usr/bin/env python3
"""wasted_run_analysis.py — silver-lining audit of the 2026-04-26 wasted run.

The 2026-04-26 compete.py run accidentally re-ran the SOTA agent on every
question on disk (9399 questions across 14 ask dates, ~$2.4k of
OpenRouter spend) before the date-filter bug was caught. trials 1 & 2
finished for all 9399; trial 3 finished ~950.

This script extracts what's salvageable:

1. **Reproduction check** — does today's BLF pipeline reproduce the
   paper Table 1 / 4 numbers when re-running the SOTA agent on the
   same questions used in the paper (tranche-a, tranche-b, tranche-ab)?
   The migrated paper data lives in forecasts_final/{date}/pro-high-
   brave-c1-p1-t1.json (5 trials averaged); the wasted run gives us 2
   fresh trials. We average the 2 trials with logit-mean and compute
   per-question BI, then aggregate per (source, FBQtype).

2. **Wider per-source BI** — using the full ~8400 resolved entries
   from the wasted run (vs ~800 in tranche-ab), do per-source BI
   patterns hold? Is tranche AuB representative of the broader corpus?

Read-only: does NOT touch forecasts_final/ or anything tracked.
"""

import os, json, math, glob, sys
from collections import defaultdict

CONFIG = "pro-high-brave-c1-p1-t1"
TRIALS = [1, 2]  # use the two complete trials from the wasted run
RAW = f"experiments/forecasts_raw/{CONFIG}"
FINAL = "experiments/forecasts_final"
MARKET = {"infer", "manifold", "metaculus", "polymarket"}


def _logit(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit_mean(ps):
    if not ps:
        return None
    return _sigmoid(sum(_logit(p) for p in ps) / len(ps))


def bi_from_bs(bs_values):
    """Population BI from a list of BS values."""
    if not bs_values:
        return None
    return 100 * (1 - math.sqrt(max(0.0, sum(bs_values) / len(bs_values))))


# ---------------------------------------------------------------------------
# 1. Build per-(source, qid, res_idx) records from the wasted-run RAW data
# ---------------------------------------------------------------------------

def gather_wasted_records():
    """For every (source, qid) touched by the wasted run, average the
    available trial probabilities (logit-mean) per resolution date and
    return a list of (source, qid, fdd, res_idx, p_logit_mean, outcome)."""
    print(f"Reading wasted-run raw trials {TRIALS}...")
    by_q = defaultdict(dict)  # (src, qid) -> {trial: fc}
    for t in TRIALS:
        td = f"{RAW}/trial_{t}"
        for src in os.listdir(td):
            sp = os.path.join(td, src)
            if not os.path.isdir(sp):
                continue
            for fn in os.listdir(sp):
                if not fn.endswith(".json"):
                    continue
                with open(os.path.join(sp, fn)) as f:
                    fc = json.load(f)
                qid = fn.removesuffix(".json")
                by_q[(src, qid)][t] = fc
    print(f"  loaded {len(by_q)} (source, qid) pairs")

    records = []
    for (src, qid), trials in by_q.items():
        if len(trials) < 2:
            continue
        # forecasts is a list (one per resolution date) per trial.
        # Get the resolved_to per res_date from the first trial's fc.
        ref = next(iter(trials.values()))
        res_dates = ref.get("resolution_dates") or [ref.get("resolution_date", "")]
        outcomes = ref.get("resolved_to")
        if not isinstance(outcomes, list):
            outcomes = [outcomes]
        fdd = ref.get("forecast_due_date", "")
        # Per res_date, average the trial forecasts
        for i in range(len(res_dates)):
            ps = []
            for t, fc in trials.items():
                fl = fc.get("forecasts")
                if isinstance(fl, list) and i < len(fl) and fl[i] is not None:
                    ps.append(float(fl[i]))
                elif i == 0 and fc.get("forecast") is not None:
                    ps.append(float(fc["forecast"]))
            if not ps:
                continue
            o = outcomes[i] if i < len(outcomes) else None
            if o is None:
                continue
            records.append({
                "source": src, "qid": qid, "fdd": fdd,
                "res_idx": i, "p": logit_mean(ps), "o": float(o),
                "n_trials": len(ps),
            })
    return records


def per_source_bi(records, sources=None):
    by_src = defaultdict(list)
    for r in records:
        if sources is not None and r["source"] not in sources:
            continue
        by_src[r["source"]].append((r["p"] - r["o"]) ** 2)
    return {s: (bi_from_bs(bs), len(bs)) for s, bs in by_src.items()}


def overall_bi_two_split(records):
    mkt_bs = [(r["p"] - r["o"]) ** 2 for r in records if r["source"] in MARKET]
    dat_bs = [(r["p"] - r["o"]) ** 2 for r in records if r["source"] not in MARKET]
    mkt_bi = bi_from_bs(mkt_bs)
    dat_bi = bi_from_bs(dat_bs)
    overall = (mkt_bi + dat_bi) / 2 if mkt_bi is not None and dat_bi is not None else None
    return mkt_bi, dat_bi, overall, len(mkt_bs), len(dat_bs)


# ---------------------------------------------------------------------------
# 2. Compare to migrated paper numbers stored in forecasts_final
# ---------------------------------------------------------------------------

def load_paper_records(date):
    """Read migrated forecasts_final/{date}/{CONFIG}.json (paper run, 5 trials
    aggregated) and return per-(source, qid, res_idx) records."""
    p = f"{FINAL}/{date}/{CONFIG}.json"
    if not os.path.exists(p):
        return []
    with open(p) as f:
        d = json.load(f)
    out = []
    for e in d.get("forecasts", []):
        rt = e.get("resolved_to")
        if isinstance(rt, list):
            o = rt[0] if rt else None
        else:
            o = rt
        if o is None or e.get("forecast") is None:
            continue
        out.append({"source": e["source"], "qid": e["id"],
                    "fdd": e.get("forecast_due_date", "")[:10],
                    "p": float(e["forecast"]), "o": float(o),
                    "res_date": str(e.get("resolution_date", ""))})
    return out


def main():
    print("=" * 72)
    print("Section 1: Reproduction check — does the wasted run match paper?")
    print("=" * 72)
    wasted = gather_wasted_records()
    paper_dates = ["2025-10-26", "2025-11-09"]  # tranche-a + tranche-b ask dates

    for date in paper_dates:
        print(f"\n--- ask date {date} ---")
        wasted_d = [r for r in wasted if r["fdd"] == date]
        paper_d = load_paper_records(date)
        print(f"  wasted records: {len(wasted_d)}, paper records: {len(paper_d)}")

        for label, recs in [("wasted (2 trials)", wasted_d),
                            ("paper   (5 trials)", paper_d)]:
            mkt_bi, dat_bi, ov_bi, n_m, n_d = overall_bi_two_split(recs)
            print(f"  {label:<22}  mkt={mkt_bi:5.2f} (n={n_m})  "
                  f"dat={dat_bi:5.2f} (n={n_d})  overall={ov_bi:5.2f}")

    print()
    print("=" * 72)
    print("Section 2: Per-source BI across the broader corpus")
    print("=" * 72)
    print()
    print(f"  total resolved records (wasted, 2 trials avg): {len(wasted):,}")
    print(f"  spans {len({r['fdd'] for r in wasted})} ask dates")
    print()
    print(f"  per-source population BI (wasted, all dates):")
    full = per_source_bi(wasted)
    for s in sorted(full):
        bi, n = full[s]
        print(f"    {s:<12} {bi:5.2f}  (n={n:,})")

    print()
    mkt_bi, dat_bi, ov_bi, n_m, n_d = overall_bi_two_split(wasted)
    print(f"  market split:  {mkt_bi:5.2f} (n={n_m:,})")
    print(f"  dataset split: {dat_bi:5.2f} (n={n_d:,})")
    print(f"  overall (eq.-wt of m/d): {ov_bi:5.2f}")

    print()
    print("=" * 72)
    print("Section 3: Per-source BI on tranche AuB only (paper subset)")
    print("=" * 72)
    aub = [r for r in wasted if r["fdd"] in paper_dates]
    print(f"  records: {len(aub):,}")
    sub = per_source_bi(aub)
    print(f"  per-source population BI (wasted, tranche AuB):")
    for s in sorted(sub):
        bi, n = sub[s]
        bi_full, n_full = full.get(s, (None, 0))
        delta = (bi - bi_full) if (bi is not None and bi_full is not None) else None
        delta_str = f" (Δ vs full = {delta:+5.2f})" if delta is not None else ""
        print(f"    {s:<12} {bi:5.2f}  (n={n:>3}){delta_str}")


if __name__ == "__main__":
    sys.exit(main())
