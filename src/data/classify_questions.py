#!/usr/bin/env python3
"""classify_questions.py — Classify forecasting questions into categories using an LLM.

Reads all question JSONs from one or more source directories under data/questions/
and classifies each into a category using a flash LLM.  Results are written
per-question to data/tags_{version}/{source}/{id}.json.

Also generates:
- data/tags_{version}/figs/tag_distribution_{source}.png
- data/tags_{version}/figs/tag_examples.html

Usage:
    python3 src/classify_questions.py --source aibq2
    python3 src/classify_questions.py --source metaculus,polymarket --max-workers 100
    python3 src/classify_questions.py --version ben
    python3 src/classify_questions.py --version all   # both xinghua and ben (default)
"""

import argparse
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv
dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

from agent.llm_client import chat

CLASSIFIER_MODEL = "openrouter/google/gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Category sets by version
# ---------------------------------------------------------------------------

# Legacy tag sets (kept for reference, data deleted)
# CATEGORIES_XINGHUA: 15 categories (too fine-grained, many near-empty)
# CATEGORIES_BEN: 9 categories ("Economics & Business" too broad at 121/400)

CATEGORIES_KEVIN: list[str] = [
    "Geopolitics & Conflict",      # wars, international relations, ACLED events
    "Domestic Politics",            # elections, governance, policy decisions
    "Financial Markets",            # stocks, indices, crypto, prediction markets
    "Macroeconomics",               # interest rates, employment, GDP, exchange rates
    "Weather & Climate",            # temperature, precipitation, energy from weather
    "Science & Technology",         # AI, space, research, tech companies
    "Sports & Entertainment",       # athletics, arts, culture, media
    "Health & Biology",             # medicine, diseases, vaccines, public health
    "Business & Industry",          # companies, earnings, trade, supply chain
    "Society & Law",                # demographics, legal, social issues, education
]

VERSION_CATEGORIES = {
    "kevin": CATEGORIES_KEVIN,
}


def _build_system_prompt(categories: list[str]) -> str:
    cat_list = "\n".join(f"- {c}" for c in categories)
    return f"""\
You are a question classifier for a forecasting benchmark.

Your task: given a forecasting question (with its question text, resolution
criteria, and background information), classify it into exactly ONE category
from the list below.

Categories:
{cat_list}

If none of the categories are a good fit, choose "Other" and suggest a better
category name (up to 5 words).

Respond with ONLY a JSON object (no markdown fences) in this exact format:
{{"category": "<label from the list or Other>", "suggested_category": "<new name or null>", "justification": "<one sentence>"}}
"""


def _build_user_prompt(q: dict) -> str:
    parts = [f"Question: {q.get('question', '')}"]
    rc = q.get("resolution_criteria", "")
    if rc:
        parts.append(f"Resolution criteria: {rc}")
    bg = q.get("background", "")
    if bg:
        parts.append(f"Background: {bg}")
    return "\n\n".join(parts)


def _parse_response(text: str, categories: list[str]) -> dict:
    """Extract the JSON object from the LLM response."""
    text = (text or "").strip()
    if not text:
        return {"category": "Other", "suggested_category": "Unclassified",
                "justification": "LLM returned empty response"}
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())
    # Try to extract JSON object if there's surrounding text
    m = re.search(r'\{[^{}]*\}', text)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"category": "Other", "suggested_category": "Unclassified",
                "justification": f"Failed to parse LLM response: {text[:100]}"}
    # Normalise
    cat = obj.get("category", "Other")
    # Case-insensitive match
    cat_lower = {c.lower(): c for c in categories}
    if cat.lower() in cat_lower:
        cat = cat_lower[cat.lower()]
    elif cat.lower() != "other":
        obj["suggested_category"] = cat
        cat = "Other"
    obj["category"] = cat
    if obj.get("suggested_category") in (None, "null", ""):
        obj["suggested_category"] = None
    return {
        "category": obj["category"],
        "suggested_category": obj.get("suggested_category"),
        "justification": obj.get("justification", ""),
    }


class _RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, rpm: int = 250):
        self._lock = threading.Lock()
        self._rpm = rpm
        self._min_rpm = 20
        self._interval = 60.0 / rpm
        self._next_slot = time.monotonic()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next_slot:
                delay = self._next_slot - now
                self._next_slot += self._interval
            else:
                delay = 0
                self._next_slot = now + self._interval
        if delay > 0:
            time.sleep(delay)

    def reduce(self):
        with self._lock:
            new_rpm = max(self._min_rpm, self._rpm // 2)
            if new_rpm < self._rpm:
                self._rpm = new_rpm
                self._interval = 60.0 / new_rpm
                print(f"  Rate limited — reducing to {self._rpm} rpm")


_rate_limiter: _RateLimiter | None = None


def classify_one(q: dict, system_prompt: str, categories: list[str],
                 max_retries: int = 6) -> tuple[str, str, dict]:
    """Classify a single question with retry + backoff on rate limits."""
    source = q.get("source", "unknown")
    qid = str(q.get("id", "unknown"))
    prompt = _build_user_prompt(q)

    for attempt in range(max_retries + 1):
        if _rate_limiter:
            _rate_limiter.wait()
        try:
            text, _, _, _ = chat(prompt, model=CLASSIFIER_MODEL, max_tokens=256,
                                 system=system_prompt)
            result = _parse_response(text, categories)
            return source, qid, result
        except Exception as e:
            if "RateLimitError" in type(e).__name__ or "429" in str(e):
                if attempt == max_retries:
                    raise
                if _rate_limiter:
                    _rate_limiter.reduce()
                wait = min(2 ** attempt * 2, 60)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


QUESTIONS_DIR = os.path.join("data", "questions")


def _load_source_questions(source: str) -> list[dict]:
    source_dir = os.path.join(QUESTIONS_DIR, source)
    if not os.path.isdir(source_dir):
        sys.exit(f"ERROR: source directory not found: {source_dir}")
    questions = []
    for fname in sorted(os.listdir(source_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(source_dir, fname)) as f:
            q = json.load(f)
        questions.append(q)
    return questions


def _discover_sources() -> list[str]:
    if not os.path.isdir(QUESTIONS_DIR):
        sys.exit(f"ERROR: {QUESTIONS_DIR} not found")
    return sorted(d for d in os.listdir(QUESTIONS_DIR)
                  if os.path.isdir(os.path.join(QUESTIONS_DIR, d)))


def _run_version(version: str, all_questions: list[str], sources: list[str],
                 max_workers: int, force: bool):
    """Run classification for one tag version."""
    categories = VERSION_CATEGORIES[version]
    system_prompt = _build_system_prompt(categories)
    out_dir = f"data/tags_{version}"

    os.makedirs(out_dir, exist_ok=True)

    # Determine which questions need classification
    if force:
        to_classify = all_questions
    else:
        to_classify = [
            q for q in all_questions
            if not os.path.exists(
                os.path.join(out_dir, q.get('source', 'unknown'),
                             f"{q.get('id', 'unknown')}.json"))
        ]
        skipped = len(all_questions) - len(to_classify)
        if skipped:
            print(f"  Skipping {skipped} already-classified questions (use --force to redo)")

    # Run classifier if needed
    global _rate_limiter
    if to_classify:
        rpm = min(max_workers * 5, 250)
        print(f"\n  Classifying {len(to_classify)} questions "
              f"({max_workers} threads, rpm={rpm})")
        _rate_limiter = _RateLimiter(rpm=rpm)
        n_ok = 0
        n_fail = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(classify_one, q, system_prompt, categories): q
                       for q in to_classify}
            for fut in as_completed(futures):
                q = futures[fut]
                qid = q.get("id", "unknown")
                source = q.get("source", "unknown")
                try:
                    source, qid, result = fut.result()
                    result["question"] = q.get("question", "")
                    os.makedirs(os.path.join(out_dir, source), exist_ok=True)
                    out_path = os.path.join(out_dir, source, f"{qid}.json")
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)
                    n_ok += 1
                except Exception as e:
                    print(f"  FAIL [{source}/{qid}]: {e}")
                    n_fail += 1
                if (n_ok + n_fail) % 100 == 0:
                    print(f"  Progress: {n_ok + n_fail}/{len(to_classify)} "
                          f"({n_ok} ok, {n_fail} fail)")
        print(f"  Done: {n_ok} classified, {n_fail} failed")
    else:
        print("  All questions already classified.")

    # Generate histograms and examples
    _generate_histograms(sources, out_dir, categories)
    _generate_tag_examples(sources, out_dir, categories)


def main():
    parser = argparse.ArgumentParser(description="Classify questions into categories")
    parser.add_argument("--source", default=None,
                        help="Comma-separated source names (directories under "
                             "data/questions/). Default: all sources.")
    parser.add_argument("--version", default="kevin",
                        choices=["kevin"],
                        help="Tag version: 'kevin' (10 categories)")
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Max parallel LLM calls (default: 50)")
    parser.add_argument("--exam", default=None,
                        help="Exam name — only classify questions in this exam "
                             "(e.g., tranche-ab). Much faster than classifying all.")
    parser.add_argument("--force", action="store_true",
                        help="Re-classify even if tag file already exists")
    args = parser.parse_args()

    # Resolve sources (optionally filtered by exam)
    if args.exam:
        import json as _json
        exam_path = os.path.join("data", "exams", args.exam, "indices.json")
        if not os.path.exists(exam_path):
            sys.exit(f"ERROR: exam not found: {exam_path}")
        with open(exam_path) as f:
            exam_indices = _json.load(f)
        sources = list(exam_indices.keys())
        # Filter _load_source_questions to only return exam questions
        _orig_load = _load_source_questions
        def _filtered_load(source):
            all_qs = _orig_load(source)
            exam_ids = set(exam_indices.get(source, []))
            return [q for q in all_qs if q.get("id") in exam_ids]
        globals()['_load_source_questions'] = _filtered_load
        print(f"Exam mode: {args.exam} ({sum(len(v) for v in exam_indices.values())} questions)")
    elif args.source:
        sources = [s.strip() for s in args.source.split(",") if s.strip()]
    else:
        sources = _discover_sources()

    # Load questions from all sources
    all_questions = []
    for source in sources:
        qs = _load_source_questions(source)
        all_questions.extend(qs)
        print(f"Loaded {len(qs)} questions from {source}")

    # Determine versions to run
    if args.version == "all":
        versions = ["xinghua", "ben"]
    else:
        versions = [args.version]

    for version in versions:
        print(f"\n=== Version: {version} ({len(VERSION_CATEGORIES[version])} categories) ===")
        _run_version(version, all_questions, sources, args.max_workers, args.force)


def _generate_histograms(sources, out_dir, categories):
    """Generate a category histogram for each source, plus an 'all' aggregate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histo_dir = os.path.join(out_dir, "figs")
    os.makedirs(histo_dir, exist_ok=True)

    def _collect_counts(source_list):
        counts = {}
        for source in source_list:
            tag_dir = os.path.join(out_dir, source)
            if not os.path.isdir(tag_dir):
                continue
            for fname in os.listdir(tag_dir):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(tag_dir, fname)) as f:
                    tag = json.load(f)
                cat = tag.get("category", "unknown")
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _plot_histogram(counts, title, out_path):
        if not counts:
            return
        ordered = [(c, counts.get(c, 0)) for c in categories if counts.get(c, 0) > 0]
        extras = [(c, n) for c, n in sorted(counts.items(), key=lambda x: -x[1])
                  if c not in categories]
        ordered.extend(extras)

        labels = [c for c, _ in ordered]
        values = [v for _, v in ordered]

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 5))
        bars = ax.barh(range(len(labels)), values, color="#4a90d9")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Number of questions")
        ax.set_title(title, fontsize=12, pad=10)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_width() + max(values) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Histogram: {out_path}")

    # Per-source histograms
    for source in sources:
        counts = _collect_counts([source])
        _plot_histogram(counts, f"Category distribution: {source}",
                        os.path.join(histo_dir, f"tag_distribution_{source}.png"))

    # Aggregate histogram across all sources
    if len(sources) > 1:
        all_counts = _collect_counts(sources)
        _plot_histogram(all_counts, f"Category distribution: all ({len(sources)} sources)",
                        os.path.join(histo_dir, "tag_distribution_all.png"))


def _generate_tag_examples(sources, out_dir, categories):
    """Generate tag_examples_all.html and tag_examples_{source}.html."""
    import html as html_lib

    def _esc(text):
        return html_lib.escape(str(text)) if text else ""

    def _collect_examples(source_list):
        cat_examples = {}
        for source in source_list:
            tag_dir = os.path.join(out_dir, source)
            if not os.path.isdir(tag_dir):
                continue
            for fname in sorted(os.listdir(tag_dir)):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(tag_dir, fname)) as f:
                    tag = json.load(f)
                cat = tag.get("category", "unknown")
                q_text = tag.get("question", "")
                qid = fname.removesuffix(".json")
                cat_examples.setdefault(cat, []).append((source, qid, q_text))
        return cat_examples

    def _write_examples_html(cat_examples, title, source_label, out_path):
        if not cat_examples:
            return
        ordered_cats = [c for c in categories if c in cat_examples]
        extras = [c for c in sorted(cat_examples) if c not in categories]
        ordered_cats.extend(extras)

        rows = ""
        for cat in ordered_cats:
            examples = cat_examples[cat][:3]
            n_total = len(cat_examples[cat])
            cells = ""
            for source, qid, q_text in examples:
                short = q_text[:300]
                if len(q_text) > 300:
                    cells += (f'<td><details><summary>{_esc(short)}...</summary>'
                              f'{_esc(q_text)}</details>'
                              f'<div style="font-size:0.8em;color:#888">{_esc(source)}/{_esc(qid)}</div></td>')
                else:
                    cells += (f'<td>{_esc(q_text)}'
                              f'<div style="font-size:0.8em;color:#888">{_esc(source)}/{_esc(qid)}</div></td>')
            for _ in range(3 - len(examples)):
                cells += "<td></td>"
            rows += f'<tr><td style="font-weight:bold;white-space:nowrap">{_esc(cat)} ({n_total})</td>{cells}</tr>\n'

        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{_esc(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 20px; }}
table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
details summary {{ cursor: pointer; }}
</style></head><body>
<h1>{_esc(title)}</h1>
<p>Sources: {_esc(source_label)} |
{sum(len(v) for v in cat_examples.values())} questions across {len(ordered_cats)} categories</p>
<table>
<tr><th>Category (count)</th><th>Example 1</th><th>Example 2</th><th>Example 3</th></tr>
{rows}
</table>
</body></html>"""

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(html)
        print(f"  Tag examples: {out_path}")

    figs_dir = os.path.join(out_dir, "figs")

    # Per-source examples
    for source in sources:
        cat_examples = _collect_examples([source])
        _write_examples_html(cat_examples, f"Tag Examples: {source}", source,
                             os.path.join(figs_dir, f"tag_examples_{source}.html"))

    # All-sources aggregate
    if len(sources) > 1:
        cat_examples = _collect_examples(sources)
        _write_examples_html(cat_examples, "Tag Examples: all sources",
                             ", ".join(sources),
                             os.path.join(figs_dir, "tag_examples_all.html"))


if __name__ == "__main__":
    main()
