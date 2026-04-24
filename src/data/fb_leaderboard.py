#!/usr/bin/env python3
"""fb_leaderboard.py — ForecastBench leaderboard: discover, compare, and import methods.

Downloads processed forecast sets from ForecastBench, finds which methods
have forecasts overlapping a given xid's exam, scores them, and generates
a sortable HTML leaderboard.

Usage:
    # Generate leaderboard (HTML + markdown)
    python3 src/fb_leaderboard.py --xid xid-ben-rich
    python3 src/fb_leaderboard.py --xid xid-market-both

    # Import specific methods as local configs for eval
    python3 src/fb_leaderboard.py --xid xid-ben-rich \
        --import-method "external.Google DeepMind.2" --score
"""

import argparse
import glob
import json
import os
import re
import tarfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FB_CACHE = os.path.join("data", "fb_cache")
_TAR_NAME = "processed_forecast_sets.tar.gz"
_TAR_PATH = os.path.join(_FB_CACHE, _TAR_NAME)
_EXTRACTED_DIR = os.path.join(_FB_CACHE, "forecastbench-processed-forecast-sets")
_TAR_URL = ("https://www.forecastbench.org/assets/data/"
            "processed-forecast-sets/processed_forecast_sets.tar.gz")
_FORECASTS_DIR = os.path.join("experiments", "forecasts_raw")
_OUTPUT_DIR = os.path.join("experiments", "fb_leaderboard")


_REFRESH = False  # set by CLI --refresh flag

def _ensure_extracted(refresh: bool = None):
    """Download and extract FB processed forecast sets if not already present."""
    if refresh is None:
        refresh = _REFRESH
    if refresh:
        import shutil
        if os.path.isdir(_EXTRACTED_DIR):
            shutil.rmtree(_EXTRACTED_DIR)
            print(f"  Deleted {_EXTRACTED_DIR}")
        if os.path.exists(_TAR_PATH):
            os.remove(_TAR_PATH)
            print(f"  Deleted {_TAR_PATH}")
    if os.path.isdir(_EXTRACTED_DIR):
        return
    if not os.path.exists(_TAR_PATH):
        print(f"Downloading {_TAR_URL} ...")
        import requests
        resp = requests.get(_TAR_URL, stream=True)
        resp.raise_for_status()
        os.makedirs(_FB_CACHE, exist_ok=True)
        with open(_TAR_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print(f"  Saved to {_TAR_PATH}")
    print(f"Extracting {_TAR_PATH} ...")
    with tarfile.open(_TAR_PATH) as tf:
        tf.extractall(_FB_CACHE)
    print(f"  Extracted to {_EXTRACTED_DIR}")


# ---------------------------------------------------------------------------
# Exam helpers
# ---------------------------------------------------------------------------

def _load_exam_keys(exam_name: str) -> dict[str, dict[tuple[str, str], str]]:
    """Load exam and return per-date mappings.

    Returns {forecast_due_date: {(source, base_id): full_qid}}.
    """
    from core.eval import load_exam
    exam = load_exam(exam_name)
    by_date = {}
    for source, ids in exam.items():
        for qid in ids:
            m = re.search(r'_(\d{4}-\d{2}-\d{2})$', qid)
            if m:
                fdd = m.group(1)
                base = qid[:m.start()]
            else:
                safe_id = re.sub(r'[/\\:]', '_', str(qid))
                q_path = os.path.join("data", "questions", source, f"{safe_id}.json")
                if os.path.exists(q_path):
                    with open(q_path) as f:
                        q = json.load(f)
                    fdd = q.get("forecast_due_date", "unknown")
                else:
                    fdd = "unknown"
                base = qid
            by_date.setdefault(fdd, {})[(source, base)] = qid
    return by_date


def _flat_exam_keys(by_date: dict) -> dict[tuple[str, str], str]:
    flat = {}
    for keys in by_date.values():
        flat.update(keys)
    return flat


def _load_outcomes(all_keys: dict) -> dict[tuple[str, str], float]:
    """Load resolved outcomes for exam questions."""
    questions = {}
    for (source, base_id), full_qid in all_keys.items():
        safe_id = re.sub(r'[/\\:]', '_', str(full_qid))
        q_path = os.path.join("data", "questions", source, f"{safe_id}.json")
        if os.path.exists(q_path):
            with open(q_path) as f:
                q = json.load(f)
            rt = q.get("resolved_to")
            if isinstance(rt, list):
                rt = rt[0] if rt else None
            if rt is not None:
                questions[(source, base_id)] = float(rt)
    return questions


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

def gather_results(exam_name: str):
    """Gather all FB methods' results for an exam.

    Returns (meta, results) where:
        meta = {exam_name, n_total, date_str, dates}
        results = [{method_key, org, model, n_overlap, bi, is_external}, ...]
    """
    _ensure_extracted()

    by_date = _load_exam_keys(exam_name)
    all_keys = _flat_exam_keys(by_date)
    dates = sorted(by_date.keys())
    n_total = len(all_keys)
    outcomes = _load_outcomes(all_keys)

    method_data = {}
    for fdd in dates:
        date_dir = os.path.join(_EXTRACTED_DIR, fdd)
        if not os.path.isdir(date_dir):
            continue
        date_keys = by_date[fdd]
        for fpath in sorted(glob.glob(os.path.join(date_dir, "*.json"))):
            data = _load_json_file(fpath)
            org = data.get("organization", "?")
            model = data.get("model", "?")
            forecasts = data.get("forecasts", [])

            fname = os.path.basename(fpath)
            method_key = fname.removeprefix(f"{fdd}.").removesuffix(".json")

            if method_key not in method_data:
                method_data[method_key] = {
                    "org": org, "model": model,
                    "n_overlap": 0, "bi_vals": [],
                    "is_external": method_key.startswith("external."),
                }

            md = method_data[method_key]
            seen = md.setdefault("_seen", set())
            for fc in forecasts:
                key = (fc["source"], fc["id"])
                if key not in date_keys or key in seen:
                    continue
                seen.add(key)
                md["n_overlap"] += 1
                if key in outcomes and fc["forecast"] is not None:
                    md["bi_vals"].append(1 - abs(fc["forecast"] - outcomes[key]))

    results = []
    for method_key, md in method_data.items():
        bi_vals = md["bi_vals"]
        bi = sum(bi_vals) / len(bi_vals) if bi_vals else None
        results.append({
            "method_key": method_key,
            "org": md["org"],
            "model": md["model"],
            "n_overlap": md["n_overlap"],
            "bi": bi,
            "is_external": md["is_external"],
        })

    results.sort(key=lambda x: -(x["bi"] or -1))

    date_str = dates[0] if len(dates) == 1 else f"{dates[0]} .. {dates[-1]}"
    meta = {"exam_name": exam_name, "n_total": n_total,
            "date_str": date_str, "dates": dates}
    return meta, results


# Workaround: json.load needs file handle, not path
def _load_json_file(fpath):
    with open(fpath) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Output: HTML
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>FB Leaderboard — {xid}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em; }}
h1 {{ font-size: 1.5em; }}
h2 {{ font-size: 1.2em; margin-top: 2em; }}
.meta {{ color: #666; margin-bottom: 1.5em; }}
table {{ border-collapse: collapse; margin-bottom: 2em; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f5f5f5; cursor: pointer; user-select: none; white-space: nowrap; }}
th:hover {{ background: #e8e8e8; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.sort-arrow {{ color: #bbb; font-size: 0.8em; }}
tr:hover {{ background: #f9f9f9; }}
.model-trunc {{ max-width: 250px; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; cursor: pointer; }}
.model-trunc.expanded {{ white-space: normal; max-width: none; }}
td.key {{ font-family: monospace; font-size: 0.85em; color: #555; }}
</style>
<script>
function sortTable(tableId, colIdx, numeric) {{
  var table = document.getElementById(tableId);
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var th = table.querySelectorAll('th')[colIdx];
  var asc = th.dataset.sortDir !== 'asc';
  th.dataset.sortDir = asc ? 'asc' : 'desc';
  rows.sort(function(a, b) {{
    var av = a.cells[colIdx].getAttribute('data-val') || a.cells[colIdx].textContent.trim();
    var bv = b.cells[colIdx].getAttribute('data-val') || b.cells[colIdx].textContent.trim();
    if (numeric) {{ av = parseFloat(av) || -1e9; bv = parseFloat(bv) || -1e9; }}
    if (av < bv) return asc ? -1 : 1;
    if (av > bv) return asc ? 1 : -1;
    return 0;
  }});
  rows.forEach(function(r) {{ tbody.appendChild(r); }});
}}
function toggleModel(el) {{ el.classList.toggle('expanded'); }}
</script>
</head><body>
<h1>ForecastBench Leaderboard</h1>
<p class="meta">
XID: <b>{xid}</b> | Exam: <b>{exam_name}</b> |
Questions: <b>{n_total}</b> | Dates: {date_str}
</p>
{tables}
</body></html>
"""


def _html_table(table_id: str, title: str, rows: list[dict],
                n_total: int) -> str:
    """Generate one sortable HTML table."""
    if not rows:
        return ""

    parts = [f"<h2>{title} ({len(rows)} methods)</h2>"]
    parts.append(f'<table id="{table_id}"><thead><tr>')

    cols = [
        ("#", False), ("Organization", False), ("Model", False),
        ("Method Key", False), ("Overlap", True), ("BI", True),
    ]
    for ci, (label, numeric) in enumerate(cols):
        parts.append(
            f'<th onclick="sortTable(\'{table_id}\',{ci},{str(numeric).lower()})">'
            f'{label} <span class="sort-arrow">&#x21C5;</span></th>')
    parts.append("</tr></thead><tbody>")

    for i, r in enumerate(rows, 1):
        bi_str = f"{r['bi']:.3f}" if r['bi'] is not None else "\u2014"
        bi_val = f"{r['bi']:.4f}" if r['bi'] is not None else "-1"
        model_esc = _esc(r['model'])
        parts.append(
            f"<tr>"
            f"<td class='num'>{i}</td>"
            f"<td>{_esc(r['org'])}</td>"
            f"<td><div class='model-trunc' onclick='toggleModel(this)' "
            f"title='{model_esc}'>{model_esc}</div></td>"
            f"<td class='key'>{_esc(r['method_key'])}</td>"
            f"<td class='num' data-val='{r['n_overlap']}'>"
            f"{r['n_overlap']}/{n_total}</td>"
            f"<td class='num' data-val='{bi_val}'>{bi_str}</td>"
            f"</tr>")

    parts.append("</tbody></table>")
    return "\n".join(parts)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_html(xid: str, meta: dict, results: list[dict]) -> str:
    external = [r for r in results if r["is_external"] and r["n_overlap"] > 0]
    official = [r for r in results if not r["is_external"] and r["n_overlap"] > 0]

    tables = _html_table("external", "External Methods", external, meta["n_total"])
    tables += _html_table("official", "ForecastBench Official", official, meta["n_total"])

    return _HTML_TEMPLATE.format(
        xid=xid, exam_name=meta["exam_name"],
        n_total=meta["n_total"], date_str=meta["date_str"],
        tables=tables)


# ---------------------------------------------------------------------------
# Output: Markdown
# ---------------------------------------------------------------------------

def generate_md(xid: str, meta: dict, results: list[dict]) -> str:
    lines = [
        f"# ForecastBench Leaderboard — {xid}",
        f"",
        f"Exam: **{meta['exam_name']}** | "
        f"Questions: **{meta['n_total']}** | "
        f"Dates: {meta['date_str']}",
        "",
    ]

    for title, rows in [
        ("External Methods",
         [r for r in results if r["is_external"] and r["n_overlap"] > 0]),
        ("ForecastBench Official",
         [r for r in results if not r["is_external"] and r["n_overlap"] > 0]),
    ]:
        if not rows:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"| # | Organization | Model | Method Key | Overlap | BI |")
        lines.append(f"|---|---|---|---|---|---|")
        for i, r in enumerate(rows, 1):
            bi_str = f"{r['bi']:.3f}" if r['bi'] is not None else "\u2014"
            lines.append(
                f"| {i} | {r['org']} | {r['model']} | "
                f"`{r['method_key']}` | "
                f"{r['n_overlap']}/{meta['n_total']} | {bi_str} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# History: list all submissions by an organization across dates
# ---------------------------------------------------------------------------

_MARKET_SOURCES = {"infer", "manifold", "metaculus", "polymarket"}
_DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}


def gather_history(org_queries: list[str]) -> list[dict]:
    """Find all submissions by one or more orgs across all dates.

    Matches org_queries against the method_key suffix (the part after
    "external." for external methods, or the org prefix like "Anthropic."
    for ForecastBench official methods).

    Returns list of {date, org, model, method_key, n_forecasts,
                     n_resolved, n_resolved_market, n_resolved_dataset}.
    """
    _ensure_extracted()
    org_lowers = [q.lower() for q in org_queries]

    def _matches(method_key: str, file_org: str) -> bool:
        mk_lower = method_key.lower()
        org_lower = file_org.lower()
        for q in org_lowers:
            # Match against method_key components (e.g. "sirbayes" in
            # "external.sirbayes.1" or "Anthropic" in "Anthropic.claude...")
            if q in mk_lower or q in org_lower:
                return True
        return False

    results = []
    for date_name in sorted(os.listdir(_EXTRACTED_DIR)):
        date_dir = os.path.join(_EXTRACTED_DIR, date_name)
        if not os.path.isdir(date_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(date_dir, "*.json"))):
            fname = os.path.basename(fpath)
            data = _load_json_file(fpath)
            file_org = data.get("organization", "")
            method_key = fname.removeprefix(f"{date_name}.").removesuffix(".json")
            if not _matches(method_key, file_org):
                continue
            model = data.get("model", "?")
            forecasts = data.get("forecasts", [])
            n_forecasts = len(forecasts)
            n_resolved = 0
            n_resolved_market = 0
            n_resolved_dataset = 0
            for fc in forecasts:
                if fc.get("resolved"):
                    n_resolved += 1
                    src = fc.get("source", "")
                    if src in _MARKET_SOURCES:
                        n_resolved_market += 1
                    elif src in _DATASET_SOURCES:
                        n_resolved_dataset += 1
            results.append({
                "date": date_name,
                "org": file_org,
                "model": model,
                "method_key": method_key,
                "n_forecasts": n_forecasts,
                "n_resolved": n_resolved,
                "n_resolved_market": n_resolved_market,
                "n_resolved_dataset": n_resolved_dataset,
            })
    return results


_HISTORY_CSS_JS = """\
<style>
body { font-family: -apple-system, system-ui, sans-serif; margin: 2em; }
h1 { font-size: 1.5em; }
h2 { font-size: 1.2em; margin-top: 2em; }
.meta { color: #666; margin-bottom: 1.5em; }
table { border-collapse: collapse; margin-bottom: 2em; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
th { background: #f5f5f5; cursor: pointer; user-select: none; white-space: nowrap; }
th:hover { background: #e8e8e8; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.key { font-family: monospace; font-size: 0.85em; color: #555; }
.sort-arrow { color: #bbb; font-size: 0.8em; }
tr:hover { background: #f9f9f9; }
</style>
<script>
function sortTable(tableId, colIdx, numeric) {
  var table = document.getElementById(tableId);
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var th = table.querySelectorAll('th')[colIdx];
  var asc = th.dataset.sortDir !== 'asc';
  th.dataset.sortDir = asc ? 'asc' : 'desc';
  rows.sort(function(a, b) {
    var av = a.cells[colIdx].getAttribute('data-val') || a.cells[colIdx].textContent.trim();
    var bv = b.cells[colIdx].getAttribute('data-val') || b.cells[colIdx].textContent.trim();
    if (numeric) { av = parseFloat(av) || -1e9; bv = parseFloat(bv) || -1e9; }
    if (av < bv) return asc ? -1 : 1;
    if (av > bv) return asc ? 1 : -1;
    return 0;
  });
  rows.forEach(function(r) { tbody.appendChild(r); });
}
</script>"""

_HISTORY_COLS = [("Forecast Date", False), ("Organization", False),
                 ("Model", False), ("Method Key", False),
                 ("Questions", True), ("Resolved", True),
                 ("Market", True), ("Dataset", True)]


def _history_html_table(table_id: str, title: str, rows: list[dict]) -> str:
    if not rows:
        return ""
    parts = [f"<h2>{_esc(title)} ({len(rows)})</h2>"]
    parts.append(f'<table id="{table_id}"><thead><tr>')
    for ci, (label, numeric) in enumerate(_HISTORY_COLS):
        parts.append(
            f'<th onclick="sortTable(\'{table_id}\',{ci},{str(numeric).lower()})">'
            f'{label} <span class="sort-arrow">&#x21C5;</span></th>')
    parts.append("</tr></thead><tbody>")
    for r in rows:
        parts.append(
            f"<tr>"
            f"<td>{r['date']}</td>"
            f"<td>{_esc(r['org'])}</td>"
            f"<td>{_esc(r['model'])}</td>"
            f"<td class='key'>{_esc(r['method_key'])}</td>"
            f"<td class='num'>{r['n_forecasts']}</td>"
            f"<td class='num'>{r['n_resolved']}</td>"
            f"<td class='num'>{r['n_resolved_market']}</td>"
            f"<td class='num'>{r['n_resolved_dataset']}</td>"
            f"</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _history_md_table(title: str, rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = [
        f"## {title} ({len(rows)})", "",
        "| Forecast Date | Organization | Model | Method Key | Questions | Resolved | Market | Dataset |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['date']} | {r['org']} | {r['model']} | `{r['method_key']}` | "
            f"{r['n_forecasts']} | {r['n_resolved']} | "
            f"{r['n_resolved_market']} | {r['n_resolved_dataset']} |")
    lines.append("")
    return "\n".join(lines)


def generate_history_html(title: str, rows: list[dict]) -> str:
    external = [r for r in rows if r["method_key"].startswith("external.")]
    official = [r for r in rows if not r["method_key"].startswith("external.")]

    tables = ""
    if external:
        tables += _history_html_table("external", "External Methods", external)
    if official:
        tables += _history_html_table("official", "ForecastBench Official", official)

    return (f"<!DOCTYPE html>\n<html><head>\n"
            f"<meta charset='utf-8'>\n"
            f"<title>FB Submissions — {_esc(title)}</title>\n"
            f"{_HISTORY_CSS_JS}\n"
            f"</head><body>\n"
            f"<h1>Submission History: {_esc(title)}</h1>\n"
            f"<p class='meta'>{len(rows)} submissions found</p>\n"
            f"{tables}\n"
            f"</body></html>")


def generate_history_md(title: str, rows: list[dict]) -> str:
    external = [r for r in rows if r["method_key"].startswith("external.")]
    official = [r for r in rows if not r["method_key"].startswith("external.")]

    lines = [
        f"# Submission History: {title}", "",
        f"{len(rows)} submissions found.", "",
    ]
    if external:
        lines.append(_history_md_table("External Methods", external))
    if official:
        lines.append(_history_md_table("ForecastBench Official", official))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Import / Score
# ---------------------------------------------------------------------------

def import_method(exam_name: str, method_key: str, config_name: str | None = None):
    """Import a FB method's forecasts as a local config for eval."""
    _ensure_extracted()

    by_date = _load_exam_keys(exam_name)

    if config_name is None:
        config_name = "fb-" + re.sub(
            r'[^a-zA-Z0-9]+', '-',
            method_key.removeprefix("external.")).strip('-').lower()

    n_written = 0
    model_name = None
    for fdd, date_keys in sorted(by_date.items()):
        date_dir = os.path.join(_EXTRACTED_DIR, fdd)
        if not os.path.isdir(date_dir):
            continue

        pattern = os.path.join(date_dir, f"{fdd}.{method_key}.json")
        matches = glob.glob(pattern)
        if not matches:
            all_files = glob.glob(os.path.join(date_dir, "*.json"))
            matches = [f for f in all_files if method_key in os.path.basename(f)]
        if not matches:
            continue

        data = _load_json_file(matches[0])
        model = data.get("model", "?")
        org = data.get("organization", "?")
        if model_name is None:
            model_name = f"{org}/{model}"

        for fc in data["forecasts"]:
            key = (fc["source"], fc["id"])
            if key not in date_keys:
                continue
            full_qid = date_keys[key]
            source = fc["source"]

            safe_id = re.sub(r'[/\\:]', '_', str(full_qid))
            out_dir = os.path.join(_FORECASTS_DIR, config_name, source)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{safe_id}.json")

            out = {
                "forecast": fc["forecast"],
                "resolved_to": fc.get("resolved_to"),
                "source": source, "id": full_qid,
                "fb_method": method_key, "fb_model": model,
                "fb_organization": org,
                "imputed": fc.get("imputed", False),
            }
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            n_written += 1

    print(f"  Importing {model_name} as '{config_name}'")
    print(f"  Wrote {n_written} forecasts -> {_FORECASTS_DIR}/{config_name}/")
    return n_written


def score_config(exam_name: str, config_name: str, metrics=None):
    """Score an imported config and print per-source results."""
    import math
    from core.eval import load_exam, load_and_score

    if metrics is None:
        metrics = ["brier-index"]

    exam = load_exam(exam_name)
    scores = load_and_score(config_name, exam, metrics)
    if not scores:
        print(f"  No scores for {config_name}")
        return

    source_vals = {}
    for (source, qid), rec in scores.items():
        for m in metrics:
            if m in rec and not (isinstance(rec[m], float) and math.isnan(rec[m])):
                source_vals.setdefault(m, {}).setdefault(source, []).append(rec[m])

    for m in metrics:
        print(f"\n  {m}:")
        sv = source_vals.get(m, {})
        all_vals = []
        for source in sorted(sv):
            vals = sv[source]
            mean = sum(vals) / len(vals)
            all_vals.extend(vals)
            print(f"    {source:>15s} (n={len(vals):3d}): {mean:.4f}")
        if all_vals:
            overall = sum(all_vals) / len(all_vals)
            print(f"    {'OVERALL':>15s} (n={len(all_vals):3d}): {overall:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ForecastBench leaderboard: discover, compare, and import methods")
    parser.add_argument("--xid", default=None, help="Experiment ID")
    parser.add_argument("--import-method", default=None,
                        help="Import method(s), comma-separated method keys")
    parser.add_argument("--config-name", default=None,
                        help="Override config name for imported method")
    parser.add_argument("--score", action="store_true",
                        help="Score imported methods")
    parser.add_argument("--metrics", default="brier-index",
                        help="Metrics to score (comma-separated)")
    parser.add_argument("--history", action="store_true",
                        help="Show submission history for an organization")
    parser.add_argument("--org", default=None,
                        help="Organization name(s), comma-separated (used with --history)")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download ForecastBench data (delete cache and fetch latest)")
    args = parser.parse_args()

    global _REFRESH
    _REFRESH = args.refresh

    # --history mode: no xid needed
    if args.history:
        if not args.org:
            parser.error("--history requires --org")
        org_list = [o.strip() for o in args.org.split(",") if o.strip()]
        rows = gather_history(org_list)
        if not rows:
            print(f"No submissions found for: {org_list}")
            return

        title = ", ".join(org_list)
        os.makedirs(_OUTPUT_DIR, exist_ok=True)

        html = generate_history_html(title, rows)
        html_path = os.path.join(_OUTPUT_DIR, "submissions.html")
        with open(html_path, "w") as f:
            f.write(html)
        print(f"  HTML: {html_path}")

        md = generate_history_md(title, rows)
        md_path = os.path.join(_OUTPUT_DIR, "submissions.md")
        with open(md_path, "w") as f:
            f.write(md)
        print(f"  Markdown: {md_path}")

        # Terminal summary
        print(f"\n  {len(rows)} submissions:")
        for r in rows:
            print(f"    {r['date']}  {r['org']:20s}  {r['model']:40s}  "
                  f"{r['n_forecasts']:>4d} q, {r['n_resolved']:>4d} resolved "
                  f"({r['n_resolved_market']} mkt, {r['n_resolved_dataset']} data)")
        return

    if not args.xid:
        parser.error("--xid is required (unless using --history)")

    from core.eval import load_xid
    xid_data = load_xid(args.xid)
    exam_name = xid_data["exam"]

    if args.import_method:
        methods = [m.strip() for m in args.import_method.split(",")]
        metrics = [m.strip() for m in args.metrics.split(",")]
        for method_key in methods:
            cfg = args.config_name if len(methods) == 1 else None
            n = import_method(exam_name, method_key, config_name=cfg)
            if n and args.score:
                if cfg is None:
                    cfg = "fb-" + re.sub(r'[^a-zA-Z0-9]+', '-',
                                         method_key.removeprefix("external.")).strip('-').lower()
                score_config(exam_name, cfg, metrics)
        return

    # Default: generate leaderboard
    meta, results = gather_results(exam_name)

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    xid_base = args.xid.removeprefix("xid-")

    html = generate_html(args.xid, meta, results)
    html_path = os.path.join(_OUTPUT_DIR, f"{args.xid}.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  HTML: {html_path}")

    md = generate_md(args.xid, meta, results)
    md_path = os.path.join(_OUTPUT_DIR, f"{args.xid}.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Markdown: {md_path}")

    # Summary to terminal
    external = [r for r in results if r["is_external"] and r["n_overlap"] > 0]
    official = [r for r in results if not r["is_external"] and r["n_overlap"] > 0]
    print(f"\n  {len(external)} external + {len(official)} official methods")
    print(f"  Top 5 external:")
    for r in external[:5]:
        bi = f"{r['bi']:.3f}" if r['bi'] is not None else "?"
        print(f"    {r['org']:25s} {r['model']:40s} "
              f"{r['n_overlap']:>3d}/{meta['n_total']}  BI={bi}")


if __name__ == "__main__":
    main()
