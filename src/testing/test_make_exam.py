#!/usr/bin/env python3
"""test_make_exam.py — regression tests for make_exam date filtering.

Catches the field-name drift that, on 2026-04-26, caused compete.py to
build a 9399-question exam (entire on-disk corpus) instead of the
500-question live LLM set: the legacy "start-date"/"end-date" mixture
fields were silently being ignored by make_exam, so the date filter
fell back to (1900-01-01, 2999-12-31) and let everything through.

These tests build small temporary mixtures and assert the exam size
matches expectation under both the canonical and legacy field names.

Usage:
    python3 src/testing/test_make_exam.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

_THIS = os.path.abspath(__file__)
_REPO = os.path.normpath(os.path.join(_THIS, "..", "..", ".."))


def _run_make_exam(name: str) -> int:
    """Invoke make_exam.py for `name`; return its exit code."""
    cmd = [sys.executable, "src/data/make_exam.py", "--name", name]
    return subprocess.run(cmd, cwd=_REPO, capture_output=True).returncode


def _exam_count(name: str) -> int:
    p = os.path.join(_REPO, "data", "exams", name, "indices.json")
    with open(p) as f:
        d = json.load(f)
    return sum(len(v) for v in d.values())


def _smallest_known_date() -> str:
    """Return the smallest forecast_due_date present in data/questions/.
    We need a date that's known to have at least one question on disk."""
    questions_dir = os.path.join(_REPO, "data", "questions")
    dates: set[str] = set()
    for src in os.listdir(questions_dir):
        sd = os.path.join(questions_dir, src)
        if not os.path.isdir(sd):
            continue
        for fn in os.listdir(sd)[:200]:  # cheap sample
            if fn.endswith(".json"):
                # Filename pattern: {id}_{YYYY-MM-DD}.json
                stem = fn.removesuffix(".json")
                if len(stem) >= 10 and stem[-10:].count("-") == 2:
                    dates.add(stem[-10:])
    if not dates:
        raise RuntimeError("no questions found on disk")
    return min(dates)


def _make_temp_exam(mixture: dict, name: str) -> int:
    """Build a temporary exam with the given mixture, return question count.
    Cleans up after."""
    exam_dir = os.path.join(_REPO, "data", "exams", name)
    os.makedirs(exam_dir, exist_ok=True)
    try:
        with open(os.path.join(exam_dir, "mixture.json"), "w") as f:
            json.dump(mixture, f)
        rc = _run_make_exam(name)
        if rc != 0:
            raise RuntimeError(f"make_exam exited {rc}")
        return _exam_count(name)
    finally:
        shutil.rmtree(exam_dir, ignore_errors=True)


# ---------------------------------------------------------------------------

def test_canonical_field_names():
    """ask-start/ask-end should filter strictly to the given date."""
    date = _smallest_known_date()
    n = _make_temp_exam({"ask-start": date, "ask-end": date},
                        name="_test_canonical")
    assert 0 < n < 9000, (
        f"Canonical ask-start/ask-end on a single date returned {n} "
        f"questions; expected a small per-date subset (well under 9000). "
        f"This means the date filter isn't actually filtering."
    )
    print(f"  PASS: canonical (ask-start/ask-end) on {date}: {n} questions")


def test_legacy_field_names():
    """start-date/end-date (legacy) should be honored as aliases."""
    date = _smallest_known_date()
    n = _make_temp_exam({"start-date": date, "end-date": date},
                        name="_test_legacy")
    assert 0 < n < 9000, (
        f"Legacy start-date/end-date returned {n} questions; expected "
        f"a small per-date subset. The fields are silently being ignored "
        f"and the date filter is falling back to its 1900..2999 default. "
        f"This is the bug that caused the 2026-04-26 9399-question exam."
    )
    print(f"  PASS: legacy (start-date/end-date) on {date}: {n} questions")


def test_canonical_and_legacy_agree():
    """Both spellings should yield the same exam size for the same date."""
    date = _smallest_known_date()
    n_canon = _make_temp_exam({"ask-start": date, "ask-end": date},
                              name="_test_agree_canonical")
    n_legacy = _make_temp_exam({"start-date": date, "end-date": date},
                               name="_test_agree_legacy")
    assert n_canon == n_legacy, (
        f"canonical={n_canon} vs legacy={n_legacy} disagree on the same "
        f"date {date} — the legacy alias isn't a perfect drop-in."
    )
    print(f"  PASS: canonical and legacy both return {n_canon} for {date}")


def test_no_dates_means_all_questions():
    """Sanity: when no date fields are given, the filter falls open and
    returns the full on-disk corpus. Documents existing behaviour."""
    n = _make_temp_exam({}, name="_test_no_dates")
    assert n > 5000, (
        f"Empty mixture should fall back to (1900..2999) and return the "
        f"whole corpus, but only got {n}; something else is filtering."
    )
    print(f"  PASS: empty mixture returns full corpus: {n} questions")


def main():
    print("Running make_exam regression tests...")
    test_canonical_field_names()
    test_legacy_field_names()
    test_canonical_and_legacy_agree()
    test_no_dates_means_all_questions()
    print("All tests passed.")


if __name__ == "__main__":
    main()
