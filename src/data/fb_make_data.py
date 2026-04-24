"""
fb_make_data.py — Download questions+resolutions and write one JSON per question.

Downloads question sets and resolution sets from the ForecastBench GitHub repo
(if not already cached locally), joins them, preprocesses, and writes one file
per question to data/questions/{source}/{id}.json.

Usage (backtesting — resolved questions across a date range):
    python3 src/fb_make_data.py --start-date 2025-10-26
    python3 src/fb_make_data.py --start-date 2025-10-26 --end-date 2026-03-01

Usage (specific dates):
    python3 src/fb_make_data.py --date 2026-01-04,2026-02-01

Usage (from a specific GitHub commit/URL):
    python3 src/fb_make_data.py --github-url https://github.com/forecastingresearch/forecastbench-datasets/blob/COMMIT/datasets/resolution_sets/DATE_resolution_set.json

Usage (live forecasting — include unresolved):
    python3 src/fb_make_data.py --date 2026-03-01 --unresolved

Usage (with live data lookups for dataset questions):
    python3 src/fb_make_data.py --start-date 2025-10-26 --lookup-dataset-ref-values

Options:
    --exam NAME    Also create experiments/exams/NAME/indices.json and meta.json

Reads (downloading if absent):
    data/fb_cache/question_sets/{date}-llm.json
    data/fb_cache/resolution_sets/{date}_resolution_set.json

Writes:
    data/questions/{source}/{id}.json   (one file per question)
"""

import argparse
from collections import defaultdict
from datetime import date as date_type
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import requests

QUESTION_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets"
    "/main/datasets/question_sets"
)
RESOLUTION_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets"
    "/main/datasets/resolution_sets"
)

COMBINATION_QUESTION_CUTOFF = "2025-10-26"

GITHUB_API_QUESTION_SETS = (
    "https://api.github.com/repos/forecastingresearch/forecastbench-datasets"
    "/contents/datasets/question_sets"
)

MARKET_SOURCES = {"infer", "manifold", "metaculus", "polymarket"}
DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}


# ---------------------------------------------------------------------------
# GitHub blob URL helpers
# ---------------------------------------------------------------------------

def _blob_to_raw(blob_url: str) -> str:
    """Convert a GitHub blob URL to a raw content URL.

    e.g. https://github.com/OWNER/REPO/blob/COMMIT/path/file.json
      -> https://raw.githubusercontent.com/OWNER/REPO/COMMIT/path/file.json
    """
    m = re.match(
        r"https://github\.com/([^/]+/[^/]+)/blob/([^/]+)/(.*)", blob_url
    )
    if not m:
        raise ValueError(f"Cannot parse GitHub blob URL: {blob_url}")
    return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"


def _resolution_url_to_question_url(res_url: str, question_set_name: str) -> str:
    """Given a raw resolution URL, derive the corresponding question set URL."""
    # Replace resolution_sets/DATE_resolution_set.json with question_sets/DATE-llm.json
    return re.sub(
        r"resolution_sets/[^/]+$",
        f"question_sets/{question_set_name}",
        res_url,
    )

from config.knowledge_cutoffs import KNOWLEDGE_CUTOFFS
EARLIEST_KNOWLEDGE_CUTOFF = max(KNOWLEDGE_CUTOFFS.values())


# ---------------------------------------------------------------------------
# Date listing (from GitHub)
# ---------------------------------------------------------------------------

def list_available_dates():
    """List all available question set dates from the ForecastBench GitHub repo."""
    resp = requests.get(GITHUB_API_QUESTION_SETS)
    resp.raise_for_status()
    dates = set()
    for entry in resp.json():
        m = re.match(r"^(\d{4}-\d{2}-\d{2})-llm\.json$", entry["name"])
        if m:
            dates.add(m.group(1))
    return sorted(dates)


# ---------------------------------------------------------------------------
# Loaders (download from GitHub if not cached locally)
# ---------------------------------------------------------------------------

def load_questions(date):
    """Load question set for date, downloading if absent."""
    local_path = f"data/fb_cache/question_sets/{date}-llm.json"
    if not os.path.exists(local_path):
        url = f"{QUESTION_RAW_BASE_URL}/{date}-llm.json"
        print(f"Downloading {url} ...")
        resp = requests.get(url)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            f.write(resp.text)
        print(f"Saved -> {local_path}")
    with open(local_path) as f:
        return json.load(f)


def load_resolutions(date):
    """Load resolution set for date, downloading if absent.
    Returns None if the resolution set does not exist on GitHub (404)."""
    local_path = f"data/fb_cache/resolution_sets/{date}_resolution_set.json"
    if not os.path.exists(local_path):
        url = f"{RESOLUTION_RAW_BASE_URL}/{date}_resolution_set.json"
        print(f"Downloading {url} ...")
        resp = requests.get(url)
        if resp.status_code == 404:
            print(f"  [{date}] No resolution set available (404)")
            return None
        resp.raise_for_status()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            f.write(resp.text)
        print(f"Saved -> {local_path}")
    with open(local_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core: process one date
# ---------------------------------------------------------------------------

def build_resolution_lookup(dates):
    """Load resolution sets for all dates, return lookup: (id, source, resolution_date) -> resolved_to."""
    lookup = {}
    for date in dates:
        res_data = load_resolutions(date)
        if res_data is None:
            continue
        for r in res_data["resolutions"]:
            if r.get("resolved") is True and not isinstance(r.get("id"), list):
                key = (str(r["id"]), r["source"], r.get("resolution_date", ""))
                if key not in lookup:
                    lookup[key] = r.get("resolved_to")
    return lookup


def process_date(date, *, unresolved=False, resolution_lookup=None):
    """Load questions and resolutions for one date, join them, return records.

    Each returned record has a "forecast_due_date" field.
    If unresolved=False, only questions with at least one resolution are included.
    If unresolved=True, all questions are included (live forecasting).
    """
    tag = f"[{date}] "

    # Load questions
    qs_data = load_questions(date)
    questions = qs_data["questions"]
    forecast_due_date = qs_data.get("forecast_due_date", date)

    # Filter combination questions for old dates
    if date < COMBINATION_QUESTION_CUTOFF:
        before = len(questions)
        questions = [q for q in questions if not isinstance(q.get("id"), list)]
        filtered = before - len(questions)
        if filtered:
            print(f"{tag}Filtered {filtered} combination question(s)")

    # Load resolutions for this date
    res_data = load_resolutions(date)
    if res_data is not None:
        resolutions = [
            r for r in res_data["resolutions"]
            if r.get("resolved") is True and not isinstance(r.get("id"), list)
        ]
    else:
        resolutions = []

    # Build per-question resolution index from this date's resolution set
    local_res = {}
    for r in resolutions:
        qid = str(r["id"])
        src = r["source"]
        key = (qid, src)
        if src in MARKET_SOURCES:
            local_res[key] = (r.get("resolved_to"), r.get("resolution_date"))
        else:
            local_res.setdefault(key, {})[r.get("resolution_date", "")] = r.get("resolved_to")

    print(f"{tag}Questions: {len(questions)}  |  Resolved entries: {len(resolutions)}")

    records = []
    n_resolved = 0
    for q in questions:
        qid = str(q.get("id", ""))
        src = q.get("source", "")
        rec = dict(q)
        rec["id"] = qid
        rec["forecast_due_date"] = forecast_due_date

        if src in DATASET_SOURCES:
            rdates = q.get("resolution_dates", [])
            if not isinstance(rdates, list):
                rdates = []
            resolved_vals = []
            for rd in rdates:
                val = None
                if resolution_lookup:
                    val = resolution_lookup.get((qid, src, rd))
                if val is None:
                    local = local_res.get((qid, src))
                    if isinstance(local, dict):
                        val = local.get(rd)
                resolved_vals.append(val)

            is_resolved = any(v is not None for v in resolved_vals)
            rec["resolved_to"] = resolved_vals
            rec.pop("resolution_date", None)
        else:
            local = local_res.get((qid, src))
            if local is not None:
                rec["resolved_to"] = [local[0]]
                rec["resolution_dates"] = [local[1]]
                is_resolved = True
            else:
                rec["resolved_to"] = None
                is_resolved = False
            rec.pop("resolution_date", None)

        rec.pop("resolved", None)

        if is_resolved:
            n_resolved += 1

        if not unresolved and not is_resolved:
            continue
        records.append(rec)

    print(f"{tag}Output: {len(records)} records ({n_resolved} resolved)")
    return records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup_key(r):
    """Deduplication key: (id, source, forecast_due_date) for all questions.

    This ensures the same question asked on different dates gets separate entries.
    """
    return (str(r.get("id", "")), r.get("source", ""), r.get("forecast_due_date", ""))


def deduplicate(all_records):
    """Deduplicate records, merging resolution_dates for dataset questions."""
    seen = {}
    deduped = []
    for r in all_records:
        key = _dedup_key(r)
        src = r.get("source", "")
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(r)
        elif src in DATASET_SOURCES:
            existing = deduped[seen[key]]
            ex_rdates = existing.get("resolution_dates", [])
            ex_resolved = existing.get("resolved_to", [])
            if not isinstance(ex_rdates, list):
                ex_rdates = []
            if not isinstance(ex_resolved, list):
                ex_resolved = []
            new_rdates = r.get("resolution_dates", [])
            new_resolved = r.get("resolved_to", [])
            if not isinstance(new_rdates, list):
                new_rdates = []
            if not isinstance(new_resolved, list):
                new_resolved = []
            ex_set = set(ex_rdates)
            for j, rd in enumerate(new_rdates):
                if rd not in ex_set:
                    ex_rdates.append(rd)
                    ex_resolved.append(new_resolved[j] if j < len(new_resolved) else None)
                    ex_set.add(rd)
            pairs = sorted(zip(ex_rdates, ex_resolved))
            existing["resolution_dates"] = [p[0] for p in pairs]
            existing["resolved_to"] = [p[1] for p in pairs]
            existing.pop("resolved", None)
    if len(deduped) < len(all_records):
        print(f"\nDeduplicated: {len(all_records)} -> {len(deduped)} records")
    return deduped


# ---------------------------------------------------------------------------
# Preprocessing and normalization
# ---------------------------------------------------------------------------

def strip_unresolved_dates(records):
    """Remove unresolved resolution dates from dataset questions."""
    for r in records:
        if r.get("source", "") not in DATASET_SOURCES:
            continue
        rdates = r.get("resolution_dates", [])
        rvals = r.get("resolved_to", [])
        if not isinstance(rdates, list) or not isinstance(rvals, list):
            continue
        pairs = [(d, v) for d, v in zip(rdates, rvals) if v is not None]
        r["resolution_dates"] = [p[0] for p in pairs]
        r["resolved_to"] = [p[1] for p in pairs]
        r.pop("resolved", None)
    return records


def preprocess_records(records, *, live_lookup=False):
    """Preprocess all records: dataset question expansion, market field cleanup, field normalization."""
    from agent.data_tools import DATASET_SOURCES as _DS_SOURCES, preprocess_dataset_question

    preprocessed = []
    for r in records:
        if r.get("source") in _DS_SOURCES:
            preprocessed.append(preprocess_dataset_question(r, live_lookup=live_lookup))
        else:
            # Market questions: flatten single-element lists
            if isinstance(r.get("resolved_to"), list) and len(r["resolved_to"]) == 1:
                r["resolved_to"] = r["resolved_to"][0]
            if isinstance(r.get("resolution_dates"), list) and len(r["resolution_dates"]) == 1:
                r["resolution_date"] = r["resolution_dates"][0]
                del r["resolution_dates"]
            if "freeze_datetime_value" in r:
                r["market_value"] = r.pop("freeze_datetime_value")
            if "freeze_datetime" in r:
                r["market_date"] = r.pop("freeze_datetime")[:10]
            if "freeze_datetime_value_explanation" in r:
                r["market_value_explanation"] = r.pop("freeze_datetime_value_explanation")
            r.pop("source_intro", None)
            r.pop("market_info_close_datetime", None)
            r.pop("market_info_open_datetime", None)
            # Add forecast_due_date suffix to ID (same as dataset questions)
            # so the same question asked on different dates gets a unique file
            fdd = r.get("forecast_due_date", "")
            raw_id = str(r.get("id", ""))
            if fdd and not raw_id.endswith(fdd):
                r["id"] = f"{raw_id}_{fdd}"
            preprocessed.append(r)
    return preprocessed


def normalize_fields(records):
    """Normalize resolution_criteria and background fields per source."""
    for r in records:
        src = r.get("source", "")
        mirc = r.get("market_info_resolution_criteria", "")
        bg_orig = r.get("background", "")

        if src == "metaculus":
            if mirc and mirc != "N/A":
                r["resolution_criteria"] = mirc
            else:
                r["resolution_criteria"] = ""
        elif src in ("polymarket", "manifold"):
            r["resolution_criteria"] = bg_orig
            r["background"] = ""
        elif src == "infer":
            split_match = re.search(
                r'(?:<div><b>Resolution Criteria:?\s*(?:&nbsp;)?</b></div>|'
                r'<p><strong>Resolution Criteria:?</strong></p>)',
                bg_orig, re.IGNORECASE)
            if split_match:
                bg_part = bg_orig[:split_match.start()].strip()
                rc_part = bg_orig[split_match.end():].strip()
                bg_part = re.sub(r'<[^>]+>', ' ', bg_part)
                bg_part = re.sub(r'\s+', ' ', bg_part).strip()
                rc_part = re.sub(r'<[^>]+>', ' ', rc_part)
                rc_part = re.sub(r'\s+', ' ', rc_part).strip()
                r["resolution_criteria"] = rc_part
                r["background"] = bg_part
            else:
                r["resolution_criteria"] = bg_orig
                r["background"] = ""

        r.pop("market_info_resolution_criteria", None)

    return records


# ---------------------------------------------------------------------------
# Write one JSON per question
# ---------------------------------------------------------------------------

def write_questions(records):
    """Write each record to data/questions/{source}/{id}.json.

    For dataset questions, id includes the forecast_due_date suffix.
    Returns count of files written.
    """
    n = 0
    for r in records:
        source = r.get("source", "unknown")
        qid = r.get("id", "unknown")
        # Sanitize id for filename (replace problematic chars)
        safe_id = re.sub(r'[/\\:]', '_', str(qid))
        out_dir = os.path.join("data", "questions", source)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{safe_id}.json")
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)
        n += 1
    return n


# ---------------------------------------------------------------------------
# GitHub URL mode: download from a specific commit
# ---------------------------------------------------------------------------

def process_github_url(github_url: str, *, lookup_dataset_ref_values=False):
    """Download resolution + question sets from a specific GitHub blob URL.

    Returns list of preprocessed records (resolved questions only).
    """
    raw_res_url = _blob_to_raw(github_url)
    print(f"Downloading resolution set from:\n  {raw_res_url}")
    resp = requests.get(raw_res_url)
    resp.raise_for_status()
    res_data = resp.json()

    # Extract resolved entries
    resolutions = [
        r for r in res_data["resolutions"]
        if r.get("resolved") is True and not isinstance(r.get("id"), list)
    ]
    print(f"  {len(resolutions)} resolved entries "
          f"(of {len(res_data['resolutions'])} total)")

    # Download corresponding question set
    question_set_name = res_data.get("question_set", "")
    if not question_set_name:
        sys.exit("ERROR: resolution set has no 'question_set' field")
    raw_qs_url = _resolution_url_to_question_url(raw_res_url, question_set_name)
    print(f"Downloading question set from:\n  {raw_qs_url}")
    resp = requests.get(raw_qs_url)
    resp.raise_for_status()
    qs_data = resp.json()

    questions = qs_data["questions"]
    forecast_due_date = qs_data.get("forecast_due_date",
                                    res_data.get("forecast_due_date", ""))
    print(f"  {len(questions)} questions, forecast_due_date={forecast_due_date}")

    # Filter combination questions
    questions = [q for q in questions if not isinstance(q.get("id"), list)]

    # Build resolution lookup
    res_lookup = {}
    for r in resolutions:
        qid = str(r["id"])
        src = r["source"]
        key = (qid, src)
        if src in MARKET_SOURCES:
            res_lookup[key] = (r.get("resolved_to"), r.get("resolution_date"))
        else:
            res_lookup.setdefault(key, {})[r.get("resolution_date", "")] = r.get("resolved_to")

    # Join questions with resolutions
    records = []
    for q in questions:
        qid = str(q.get("id", ""))
        src = q.get("source", "")
        rec = dict(q)
        rec["id"] = qid
        rec["forecast_due_date"] = forecast_due_date

        if src in DATASET_SOURCES:
            rdates = q.get("resolution_dates", [])
            if not isinstance(rdates, list):
                rdates = []
            resolved_vals = []
            local = res_lookup.get((qid, src))
            for rd in rdates:
                val = None
                if isinstance(local, dict):
                    val = local.get(rd)
                resolved_vals.append(val)
            is_resolved = any(v is not None for v in resolved_vals)
            rec["resolved_to"] = resolved_vals
            rec.pop("resolution_date", None)
        else:
            local = res_lookup.get((qid, src))
            if local is not None:
                rec["resolved_to"] = [local[0]]
                rec["resolution_dates"] = [local[1]]
                is_resolved = True
            else:
                rec["resolved_to"] = None
                is_resolved = False
            rec.pop("resolution_date", None)

        rec.pop("resolved", None)
        if not is_resolved:
            continue
        records.append(rec)

    print(f"  {len(records)} resolved records after join")

    # Standard pipeline
    records = strip_unresolved_dates(records)
    records = preprocess_records(records,
                                live_lookup=lookup_dataset_ref_values)
    records = normalize_fields(records)
    return records


# ---------------------------------------------------------------------------
# Exam creation
# ---------------------------------------------------------------------------

def create_exam(records: list, exam_name: str):
    """Create experiments/exams/{name}/indices.json and meta.json from records."""
    indices = defaultdict(list)
    for r in records:
        src = r.get("source", "unknown")
        qid = r.get("id", "unknown")
        safe_id = re.sub(r'[/\\:]', '_', str(qid))
        indices[src].append(safe_id)

    # Sort within each source
    for src in indices:
        indices[src] = sorted(indices[src])

    exam_dir = os.path.join("data", "exams", exam_name)
    os.makedirs(exam_dir, exist_ok=True)

    indices_path = os.path.join(exam_dir, "indices.json")
    with open(indices_path, "w") as f:
        json.dump(dict(sorted(indices.items())), f, indent=2)

    meta = {"nquestions": {src: len(ids) for src, ids in sorted(indices.items())}}
    meta["nquestions"]["total"] = sum(meta["nquestions"].values())

    # Derive date ranges from the records
    fdd_dates = [r.get("forecast_due_date", "") for r in records
                 if r.get("forecast_due_date")]
    res_dates = []
    for r in records:
        rd = r.get("resolution_date", "")
        if rd:
            res_dates.append(rd)
        for rd in r.get("resolution_dates", []):
            if rd:
                res_dates.append(rd)
    if fdd_dates:
        meta["ask_start"] = min(fdd_dates)
        meta["ask_end"] = max(fdd_dates)
    if res_dates:
        meta["resolution_start"] = min(res_dates)
        meta["resolution_end"] = max(res_dates)

    meta_path = os.path.join(exam_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total = meta["nquestions"]["total"]
    print(f"\nExam '{exam_name}': {total} questions")
    for src, ids in sorted(indices.items()):
        print(f"  {src}: {len(ids)}")
    print(f"  Wrote {indices_path}")
    print(f"  Wrote {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download questions+resolutions and write one JSON per question."
    )

    # Date selection
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--date",
                            help="Comma-separated list of dates, e.g. 2026-01-04,2026-02-01")
    date_group.add_argument("--start-date",
                            help="Generate biweekly dates from this date to --end-date (or today)")
    date_group.add_argument("--github-url",
                            help="Download from a specific GitHub blob URL for a resolution set "
                                 "(e.g. .../blob/COMMIT/.../DATE_resolution_set.json)")

    parser.add_argument("--end-date", default=None,
                        help="End date (inclusive) when using --start-date. Default: today.")
    parser.add_argument("--unresolved", action="store_true",
                        help="Include unresolved questions (for live forecasting)")
    parser.add_argument("--lookup-dataset-ref-values", action="store_true",
                        help="Fetch live reference values for dataset questions from "
                             "yfinance/FRED/dbnomics (slow). Default: off.")
    parser.add_argument("--exam", default=None,
                        help="Also create experiments/exams/NAME/ with indices.json and meta.json")
    args = parser.parse_args()

    # --- GitHub URL mode ---
    if args.github_url:
        all_records = process_github_url(
            args.github_url,
            lookup_dataset_ref_values=args.lookup_dataset_ref_values,
        )
        n_written = write_questions(all_records)
        by_source = defaultdict(int)
        for r in all_records:
            by_source[r.get("source", "unknown")] += 1
        print(f"\nWrote {n_written} question files to data/questions/")
        print(f"By source: {dict(sorted(by_source.items()))}")
        if args.exam:
            create_exam(all_records, args.exam)
        return

    # --- Date range mode ---
    end_date = args.end_date or str(date_type.today())

    if args.date:
        dates = [d.strip() for d in args.date.split(",") if d.strip()]
    else:
        all_dates = list_available_dates()
        dates = [d for d in all_dates if d >= args.start_date and d <= end_date]
        print(f"Found {len(dates)} question sets ({args.start_date} .. {end_date}) "
              f"({len(all_dates)} total on GitHub)")

    if not dates:
        sys.exit("No dates found.")

    # Build resolution lookup across all dates
    print("Building resolution lookup across all dates...")
    resolution_lookup = build_resolution_lookup(dates)
    print(f"Resolution lookup: {len(resolution_lookup)} entries")

    # Load all records across dates
    all_records = []
    for date in sorted(dates):
        try:
            records = process_date(date, unresolved=args.unresolved,
                                   resolution_lookup=resolution_lookup)
        except Exception as e:
            print(f"[{date}] Skipping: {e}")
            continue
        all_records.extend(records)

    # Deduplicate
    all_records = deduplicate(all_records)

    # Strip unresolved dates (unless in unresolved mode)
    if not args.unresolved:
        all_records = strip_unresolved_dates(all_records)

    # Preprocess dataset questions
    all_records = preprocess_records(all_records,
                                    live_lookup=args.lookup_dataset_ref_values)

    # Normalize fields
    all_records = normalize_fields(all_records)

    # Write one file per question
    n_written = write_questions(all_records)

    # Summary
    by_source = defaultdict(int)
    for r in all_records:
        by_source[r.get("source", "unknown")] += 1
    print(f"\nWrote {n_written} question files to data/questions/")
    print(f"By source: {dict(sorted(by_source.items()))}")

    if args.exam:
        create_exam(all_records, args.exam)


if __name__ == "__main__":
    main()
