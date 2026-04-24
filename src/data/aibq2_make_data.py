#!/usr/bin/env python3
"""aibq2_make_data.py — Download and convert the AIBQ2 dataset to per-question JSONs.

Downloads the AIBQ2 CSV from GitHub and writes one JSON file per question to
data/questions/aibq2/aibq2_NNNN.json (integer row IDs for readability).

Each JSON includes a title_hash field (SHA-256 of the question title) for
detecting silent changes if the gist is updated. When running without --force,
existing files are validated against the current gist and warnings are printed
if any titles have changed.

Source: https://gist.github.com/enjeeneer/86e24a52e6041a3d78e333bcab16984d

Usage:
    python3 src/aibq2_make_data.py
    python3 src/aibq2_make_data.py --force   # overwrite existing files
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import requests

GIST_RAW_URL = (
    "https://gist.githubusercontent.com/enjeeneer/"
    "86e24a52e6041a3d78e333bcab16984d/raw"
)

OUT_DIR = os.path.join("data", "questions", "aibq2")


def _extract_first_url(markdown_text: str) -> str:
    """Extract the first URL from markdown text (link or bare URL)."""
    # Markdown link: [text](url)
    m = re.search(r'\[.*?\]\((https?://\S+?)\)', markdown_text)
    if m:
        return m.group(1)
    # Bare URL
    m = re.search(r'(https?://\S+)', markdown_text)
    if m:
        return m.group(1)
    return ""


def _to_date(timestamp: str) -> str:
    """Extract YYYY-MM-DD from a timestamp string like '2025-04-21 12:00:00+00:00'."""
    return timestamp[:10] if timestamp else ""


def download_csv() -> str:
    """Download the AIBQ2 CSV from GitHub."""
    print(f"Downloading from {GIST_RAW_URL} ...")
    resp = requests.get(GIST_RAW_URL)
    resp.raise_for_status()
    return resp.text


def _title_hash(title: str) -> str:
    """SHA-256 hash of the question title (for validation against gist changes)."""
    return hashlib.sha256(title.strip().encode()).hexdigest()[:16]


def convert(csv_text: str, force: bool = False) -> tuple[int, int]:
    """Parse CSV and write per-question JSON files. Returns (n_written, n_skipped).

    Uses integer IDs (row numbers) for human readability. Stores a title_hash
    in each JSON for detecting silent changes if the gist is updated.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    reader = csv.DictReader(io.StringIO(csv_text))
    n_written = 0
    n_skipped = 0

    for i, row in enumerate(reader, 1):
        title = row.get("title", "").strip()
        if not title:
            continue
        qid = f"aibq2_{i:04d}"
        out_path = os.path.join(OUT_DIR, f"{qid}.json")

        if not force and os.path.exists(out_path):
            # Validate: check title_hash matches to detect gist changes
            with open(out_path) as f:
                existing = json.load(f)
            existing_hash = existing.get("title_hash", "")
            expected_hash = _title_hash(title)
            if existing_hash and existing_hash != expected_hash:
                print(f"  WARNING: {qid} title changed! "
                      f"Old: {existing.get('question', '')[:60]}... "
                      f"New: {title[:60]}...")
                print(f"    Re-run with --force to update.")
            n_skipped += 1
            continue

        # Build resolution_criteria from resolution_criteria + fine_print
        rc = row.get("resolution_criteria", "").strip()
        fp = row.get("fine_print", "").strip()
        if fp:
            rc = f"{rc}\n\nFine print:\n{fp}"

        # Background = description field (markdown with links)
        description = row.get("description", "").strip()
        url = _extract_first_url(description)

        # Resolution: "True"/"False" -> 1.0/0.0
        resolution_str = row.get("resolution", "").strip()
        if resolution_str.lower() == "true":
            resolved_to = 1.0
        elif resolution_str.lower() == "false":
            resolved_to = 0.0
        else:
            resolved_to = None

        q = {
            "id": qid,
            "source": "aibq2",
            "question": title,
            "title_hash": _title_hash(title),
            "background": description,
            "resolution_criteria": rc,
            "url": url,
            "forecast_due_date": _to_date(row.get("prediction_time", "")),
            "resolution_date": _to_date(row.get("scheduled_resolve_time", "")),
            "resolved_to": resolved_to,
        }

        with open(out_path, "w") as f:
            json.dump(q, f, indent=2)
        n_written += 1

    return n_written, n_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Download AIBQ2 dataset and write per-question JSONs")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing question files")
    args = parser.parse_args()

    csv_text = download_csv()
    n_written, n_skipped = convert(csv_text, force=args.force)

    total = n_written + n_skipped
    print(f"\nDone: {total} questions total")
    print(f"  Written: {n_written}")
    if n_skipped:
        print(f"  Skipped: {n_skipped} (already exist, use --force to overwrite)")
    print(f"  Output:  {OUT_DIR}/")


if __name__ == "__main__":
    main()
