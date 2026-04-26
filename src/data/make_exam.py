"""
make_exam.py — Generate an exam (a view of data/questions/) from a mixture spec.

An exam is a named subset of questions defined by a mixture.json file.
The mixture specifies which sources to include, how to slice them,
date filtering, and an optional random seed for shuffling.

Usage:
    python3 src/make_exam.py --name my_exam

Reads:
    data/exams/{name}/mixture.json

Writes:
    data/exams/{name}/indices.json

mixture.json format:
{
    "ask-start": "2025-10-26",        // filter by forecast_due_date (when asked)
    "ask-end": "2026-03-31",          // (also accepts legacy "start-date"/"end-date")
    "resolution-start": "2025-11-01", // optional: filter by resolution_date (when resolved)
    "resolution-end": "2026-03-31",   // optional: only include questions with ground truth
    "seed": 0,                        // optional, default 0; null = alphabetical sort
    "select": {
        "polymarket": [0, 25],           // skip 0, take 25 (= first 25)
        "manifold": 25,                  // shorthand for [0, 25]
        "metaculus": "all",              // all questions in date range
        "infer": [25, 10],               // skip 25, take 10
        "aibq2": ["aibq2_0001", "aibq2_0042"]  // explicit question IDs
    }
}

Select values:
    N                    → [0, N]  (first N items after shuffling)
    [offset, count]      → skip offset items, take count
    "all"                → all questions in date range
    ["id1", "id2", ...]  → explicit question IDs (date filter still applied)

If "select" is omitted, all available sources are included (all questions).

indices.json format:
{
    "polymarket": ["id1", "id2", ...],
    "metaculus": ["id1", "id2", ...],
    ...
}
Always materialized — no "all" sentinel. This is a frozen snapshot.
"""

import argparse
import json
import os
import random
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))


def load_mixture(exam_name: str) -> dict:
    """Load mixture.json for an exam."""
    path = os.path.join("data", "exams", exam_name, "mixture.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: mixture file not found: {path}")
    with open(path) as f:
        return json.load(f)


def list_question_ids(source: str, start_date: str, end_date: str,
                      resolution_start: str = "", resolution_end: str = "") -> list[str]:
    """List all question IDs for a source, filtered by date ranges.

    Filters:
      - forecast_due_date in [start_date, end_date]  (when was the question asked?)
      - resolution_date in [resolution_start, resolution_end]  (when did it resolve?)
        Only applied if resolution_start or resolution_end is non-empty.

    Returns sorted list of IDs (alphabetical by the filename/id).
    """
    source_dir = os.path.join("data", "questions", source)
    if not os.path.isdir(source_dir):
        print(f"  WARNING: no questions for source '{source}' in {source_dir}")
        return []

    ids = []
    for fname in os.listdir(source_dir):
        if not fname.endswith(".json"):
            continue
        qid = fname.removesuffix(".json")
        fpath = os.path.join(source_dir, fname)
        with open(fpath) as f:
            q = json.load(f)

        # Filter by forecast_due_date
        fdd = q.get("forecast_due_date", "")
        if not (start_date <= fdd <= end_date):
            continue

        # Filter by resolution_date (if specified)
        if resolution_start or resolution_end:
            rd = q.get("resolution_date", "")
            # For dataset questions with resolution_dates list, use the first
            rds = q.get("resolution_dates", [])
            if rds and isinstance(rds, list):
                rd = str(rds[0]) if rds else rd
            r_lo = resolution_start or "1900-01-01"
            r_hi = resolution_end or "2999-12-31"
            if not (r_lo <= rd <= r_hi):
                continue

        ids.append(qid)

    return sorted(ids)


def apply_seed(ids: list[str], seed: int | None) -> list[str]:
    """Shuffle IDs with seed. If seed is None, return alphabetically sorted."""
    if seed is None:
        return sorted(ids)
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return shuffled


def _is_id_list(spec: list) -> bool:
    """Return True if spec is a list of explicit question IDs (not [offset, count])."""
    return any(isinstance(x, str) and x.lower() != "all" for x in spec)


def parse_select(spec) -> tuple[int, int | None] | list[str]:
    """Parse a select spec.

    Returns either (offset, count) or a list of explicit question IDs.

    Slice forms:
      N           → (0, N)
      "all"       → (0, None)
      [offset, N] → (offset, N)
      [offset, "all"] → (offset, None)

    Explicit ID list:
      ["aibq2_0001", "aibq2_0042"]  → those IDs (date filter still applied)
    """
    if isinstance(spec, str) and spec.lower() == "all":
        return (0, None)
    if isinstance(spec, (int, float)):
        return (0, int(spec))
    if isinstance(spec, list):
        if _is_id_list(spec):
            return list(spec)
        if len(spec) == 2:
            offset = int(spec[0])
            count = None if (isinstance(spec[1], str) and spec[1].lower() == "all") else int(spec[1])
            return (offset, count)
    raise ValueError(f"Invalid select spec: {spec!r}. "
                     f"Expected N, \"all\", [offset, count], or [\"id1\", \"id2\", ...].")


def build_indices(mixture: dict) -> dict[str, list[str]]:
    """Build indices from a mixture spec.

    Returns {source: [id1, id2, ...]} with fully materialized ID lists.
    """
    # Accept legacy "start-date"/"end-date" as aliases for ask-start/ask-end.
    # (Older compete.py writes the legacy names; without this the date filter
    # silently passes every question on disk.)
    start_date = mixture.get("ask-start", mixture.get("start-date", "1900-01-01"))
    end_date = mixture.get("ask-end", mixture.get("end-date", "2999-12-31"))
    resolution_start = mixture.get("resolution-start", "")
    resolution_end = mixture.get("resolution-end", "")
    seed = mixture.get("seed", 0)  # default seed=0 for reproducibility
    select = mixture.get("select", {})

    # Collect all sources mentioned in select
    all_sources = set(select.keys())

    # If select is empty, discover all available sources
    if not all_sources:
        questions_dir = os.path.join("data", "questions")
        if os.path.isdir(questions_dir):
            all_sources = {
                d for d in os.listdir(questions_dir)
                if os.path.isdir(os.path.join(questions_dir, d))
            }

    indices = {}
    for source in sorted(all_sources):
        # Get all IDs for this source within date range
        ids = list_question_ids(source, start_date, end_date,
                               resolution_start, resolution_end)
        n_total = len(ids)

        # Apply select slice or explicit ID list
        if source in select:
            sel = parse_select(select[source])
            if isinstance(sel, list):
                # Explicit ID list — keep only those present in the date-filtered pool
                id_set = set(ids)
                selected = [qid for qid in sel if qid in id_set]
                missing = [qid for qid in sel if qid not in id_set]
                if missing:
                    print(f"  {source}: WARNING: IDs not found in date range: {missing}")
                print(f"  {source}: explicit ids → {len(selected)} of {len(sel)} requested")
            else:
                if not ids:
                    print(f"  {source}: 0 questions in date range, skipping")
                    continue
                ids = apply_seed(ids, seed)
                offset, count = sel
                selected = ids[offset:] if count is None else ids[offset:offset + count]
                print(f"  {source}: select [{offset}, {count if count is not None else 'all'}] "
                      f"→ {len(selected)} of {n_total}")
        else:
            if not ids:
                print(f"  {source}: 0 questions in date range, skipping")
                continue
            # Source not in select (auto-discovered): keep all
            selected = apply_seed(ids, seed)
            print(f"  {source}: all {len(selected)}")

        if selected:
            indices[source] = selected

    return indices


def main():
    parser = argparse.ArgumentParser(
        description="Generate an exam (question subset) from a mixture specification."
    )
    parser.add_argument("--name", required=True,
                        help="Exam name. Reads data/exams/{name}/mixture.json, "
                             "writes data/exams/{name}/indices.json.")
    args = parser.parse_args()

    mixture = load_mixture(args.name)
    print(f"Building exam '{args.name}'")
    ask_s = mixture.get("ask-start", "?")
    ask_e = mixture.get("ask-end", "?")
    res_s = mixture.get("resolution-start", "")
    res_e = mixture.get("resolution-end", "")
    print(f"  Asked: {ask_s} to {ask_e}")
    if res_s or res_e:
        print(f"  Resolved: {res_s or '*'} to {res_e or '*'}")
    print(f"  Seed: {mixture.get('seed', 0)}")

    indices = build_indices(mixture)

    # Write indices.json
    out_dir = os.path.join("data", "exams", args.name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "indices.json")
    with open(out_path, "w") as f:
        json.dump(indices, f, indent=2)

    # Write meta.json (mixture settings + computed stats, excluding "select"
    # which is redundant with "nquestions")
    total = sum(len(ids) for ids in indices.values())
    nquestions = {source: len(ids) for source, ids in sorted(indices.items())}
    nquestions["total"] = total
    meta = {k: v for k, v in mixture.items() if k != "select"}
    meta["nquestions"] = nquestions
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    print(f"\nWrote {out_path}")
    print(f"Wrote {meta_path}")
    print(f"  {total} questions across {len(indices)} sources")
    for source, ids in sorted(indices.items()):
        print(f"    {source}: {len(ids)}")

    # Generate data visualizations
    if total > 0:
        from data.plot_exams import (generate_rdate_by_fdate_scatter,
                                generate_horizon_histogram,
                                generate_tag_distribution)
        from config.tags import get_tags_for_exam, discover_classified_spaces

        # Plotting is cosmetic — indices.json is already written, so
        # don't let a plotting bug fail the whole exam build (compete.py
        # treats a non-zero exit as fatal).
        for fn, args_ in [
            (generate_rdate_by_fdate_scatter, (args.name, indices)),
            (generate_horizon_histogram,      (args.name, indices)),
        ]:
            try:
                p = fn(*args_)
                if p:
                    print(f"  {p}")
            except Exception as e:
                print(f"  WARNING: {fn.__name__} failed: {e}")
        for ls in discover_classified_spaces():
            tags = get_tags_for_exam(indices, ls)
            if tags:
                try:
                    p = generate_tag_distribution(args.name, tags, tag_version=ls,
                                                  n_total=total)
                    if p:
                        print(f"  {p}")
                except Exception as e:
                    print(f"  WARNING: tag-distribution plot failed: {e}")


if __name__ == "__main__":
    main()
