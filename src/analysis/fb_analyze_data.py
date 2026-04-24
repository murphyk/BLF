"""
fb_analyze_data.py -- summarize question and resolution counts by source.

Usage:
    python3 fb_analyze_data.py --start-date 2026-01-04
    python3 fb_analyze_data.py --start-date 2025-08-03 --end-date 2025-12-31


Writes:
    datasets/stats/statistics_{start}_{end}.csv
    datasets/stats/statistics_{start}_{end}.html
"""

import argparse
from collections import defaultdict
from datetime import date as date_type
import json
import os
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.fb_make_data import (
    list_available_dates, load_questions, load_resolutions,
    MARKET_SOURCES, DATASET_SOURCES,
)

SOURCE_NAMES = {
    "infer": "RFI",
    "manifold": "Manifold Markets",
    "metaculus": "Metaculus",
    "polymarket": "Polymarket",
    "acled": "ACLED",
    "dbnomics": "DBnomics",
    "fred": "FRED",
    "wikipedia": "Wikipedia",
    "yfinance": "Yahoo! Finance",
}

SOURCE_URLS = {
    "infer": "randforecastinginitiative.org",
    "manifold": "manifold.markets",
    "metaculus": "metaculus.com",
    "polymarket": "polymarket.com",
    "acled": "acleddata.com",
    "dbnomics": "db.nomics.world",
    "fred": "fred.stlouisfed.org",
    "wikipedia": "wikipedia.org",
    "yfinance": "finance.yahoo.com",
}

MARKET_ORDER = ["infer", "manifold", "metaculus", "polymarket"]
DATASET_ORDER = ["acled", "dbnomics", "fred", "wikipedia", "yfinance"]


def count_questions_and_resolutions(dates):
    """Count unique questions and resolved questions per source across dates.

    Returns (q_by_source, r_by_source) where each maps source -> set of question ids.
    """
    q_by_source = defaultdict(set)
    r_by_source = defaultdict(set)
    dates_with_data = []

    for date in dates:
        try:
            qs_data = load_questions(date)
        except Exception:
            continue

        dates_with_data.append(date)
        for q in qs_data["questions"]:
            if isinstance(q.get("id"), list):
                continue
            src = q.get("source", "")
            qid = str(q.get("id", ""))
            q_by_source[src].add(qid)

        try:
            res_data = load_resolutions(date)
        except Exception:
            continue

        if res_data is None:
            continue
        for r in res_data["resolutions"]:
            if isinstance(r.get("id"), list):
                continue
            if r.get("resolved"):
                src = r.get("source", "")
                qid = str(r.get("id", ""))
                r_by_source[src].add(qid)

    return q_by_source, r_by_source, dates_with_data


def build_table(start_date, end_date, all_dates):
    """Build list of row dicts for the statistics table."""
    recent_dates = [d for d in all_dates if d >= start_date and d <= end_date]
    q_recent, r_recent, actual_recent = count_questions_and_resolutions(recent_dates)
    q_total, r_total, actual_total = count_questions_and_resolutions(all_dates)

    range_label = f"{start_date}..{end_date}"
    col_n_recent = f"n({range_label})"
    col_nres_recent = f"nres({range_label})"

    rows = []

    def add_source(src):
        rows.append({
            "Source": SOURCE_NAMES[src],
            "URL": SOURCE_URLS[src],
            col_n_recent: len(q_recent.get(src, set())),
            col_nres_recent: len(r_recent.get(src, set())),
            "n(total)": len(q_total.get(src, set())),
            "nres(total)": len(r_total.get(src, set())),
        })

    def add_subtotal(label, sources):
        rows.append({
            "Source": label,
            "URL": "",
            col_n_recent: sum(len(q_recent.get(s, set())) for s in sources),
            col_nres_recent: sum(len(r_recent.get(s, set())) for s in sources),
            "n(total)": sum(len(q_total.get(s, set())) for s in sources),
            "nres(total)": sum(len(r_total.get(s, set())) for s in sources),
        })

    for src in MARKET_ORDER:
        add_source(src)
    add_subtotal("Market Total", MARKET_ORDER)

    for src in DATASET_ORDER:
        add_source(src)
    add_subtotal("Dataset Total", DATASET_ORDER)

    add_subtotal("Overall Total", MARKET_ORDER + DATASET_ORDER)

    return rows, col_n_recent, col_nres_recent, len(actual_recent), len(actual_total)


def save_csv(rows, path):
    """Write rows as CSV."""
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved -> {path}")


def save_html(rows, path, start_date, end_date, n_recent_dates, n_total_dates):
    """Write rows as a styled HTML table."""
    if not rows:
        return
    cols = list(rows[0].keys())

    # Identify total rows for bold styling
    total_sources = {"Market Total", "Dataset Total", "Overall Total"}

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>ForecastBench Question Statistics</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2em; }}
  h2 {{ margin-bottom: 0.3em; }}
  .subtitle {{ color: #666; margin-bottom: 1.5em; }}
  table {{ border-collapse: collapse; }}
  th, td {{ padding: 6px 14px; text-align: right; border-bottom: 1px solid #ddd; }}
  th {{ background: #f5f5f5; border-bottom: 2px solid #333; }}
  td:first-child, td:nth-child(2) {{ text-align: left; }}
  th:first-child, th:nth-child(2) {{ text-align: left; }}
  tr.total td {{ font-weight: bold; border-top: 2px solid #333; }}
  tr.grand-total td {{ font-weight: bold; border-top: 3px double #333; }}
</style>
</head><body>
<h2>ForecastBench Question Statistics</h2>
<p class="subtitle">Selected: {n_recent_dates} question sets ({start_date} .. {end_date}) &middot;
Total: {n_total_dates} question sets</p>
<table>
<tr>"""
    for c in cols:
        html += f"<th>{c}</th>"
    html += "</tr>\n"

    for r in rows:
        src = r["Source"]
        if src == "Overall Total":
            cls = ' class="grand-total"'
        elif src in total_sources:
            cls = ' class="total"'
        else:
            cls = ""
        html += f"<tr{cls}>"
        for c in cols:
            val = r[c]
            if isinstance(val, int):
                val = f"{val:,}"
            html += f"<td>{val}</td>"
        html += "</tr>\n"

    html += "</table>\n</body></html>"

    with open(path, "w") as f:
        f.write(html)
    print(f"Saved -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize question and resolution counts by source."
    )
    parser.add_argument("--start-date", required=True,
                        help="Start date for recent counts (e.g. 2026-01-04)")
    parser.add_argument("--end-date", default=None,
                        help="End date for recent counts (default: today)")
    args = parser.parse_args()

    end_date = args.end_date or str(date_type.today())

    # List all available dates from GitHub
    all_dates = list_available_dates()

    rows, col_n_recent, col_nres_recent, n_recent, n_total = build_table(
        args.start_date, end_date, all_dates)
    print(f"Selected: {n_recent} question sets ({args.start_date} .. {end_date})")
    print(f"Total: {n_total} question sets")

    # Print table to stdout
    date_range = f"{args.start_date}..{end_date}"
    print(f"\n{'Source':<20s} {'n(sel)':>8s} {'nres(sel)':>10s} {'n(all)':>8s} {'nres(all)':>10s}")
    print(f"{'':20s} {'(' + date_range + ')':>{18 + len(date_range)}s}")
    print("-" * 60)
    for r in rows:
        n1 = f"{r[col_n_recent]:,}" if isinstance(r[col_n_recent], int) else str(r[col_n_recent])
        r1 = f"{r[col_nres_recent]:,}" if isinstance(r[col_nres_recent], int) else str(r[col_nres_recent])
        nt = f"{r['n(total)']:,}" if isinstance(r["n(total)"], int) else str(r["n(total)"])
        rt = f"{r['nres(total)']:,}" if isinstance(r["nres(total)"], int) else str(r["nres(total)"])
        src = r['Source']
        # Add separator lines for subtotals
        if src in ("Market Total", "Dataset Total", "Overall Total"):
            print(f"  {'─'*56}")
            print(f"  {src:<18s} {n1:>8s} {r1:>10s} {nt:>8s} {rt:>10s}")
        else:
            print(f"  {src:<18s} {n1:>8s} {r1:>10s} {nt:>8s} {rt:>10s}")

    os.makedirs("data/stats", exist_ok=True)
    stem = f"data/stats/statistics_{args.start_date}_{end_date}"
    save_csv(rows, f"{stem}.csv")
    save_html(rows, f"{stem}.html", args.start_date, end_date, n_recent, n_total)


if __name__ == "__main__":
    main()
