#!/usr/bin/env python3
"""leak_detective.py — Detect post-cutoff information leakage in search results.

Examines all saved search results for an experiment and uses a lightweight LLM
to check whether any contain information from after the knowledge cutoff date.

This can detect:
  - True positives: results that leaked through the date filter
  - False positives: results flagged by the detective but actually fine
It cannot detect false negatives (results correctly filtered out), unless
rejected results are saved in searches/{qid}/rejected/.

Usage:
    python3 src/leak_detective.py --xid xid-market-both
    python3 src/leak_detective.py --xid xid-market-both --config pro-high-brave-crowd1-tools1
    python3 src/leak_detective.py --xid xid-market-both
"""

import argparse
import glob
import html as html_lib
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import dotenv
dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

from agent.llm_client import chat

_DETECTIVE_MODEL = "openrouter/x-ai/grok-4.1-fast"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    config: str
    trial: str
    source: str
    qid: str
    search_file: str
    url: str
    cutoff: str
    verdict: str  # "LEAK" or "CLEAN"
    reason: str              # detective's reason
    filter_reason: str = ""  # filter's KEEP/DROP reason
    rejected: bool = False   # True if from rejected/ dir

    @property
    def error_type(self) -> str:
        """FN=kept+leak, FP=rejected+clean, TP=rejected+leak, TN=kept+clean."""
        if self.verdict == "LEAK":
            return "TP" if self.rejected else "FN"
        else:
            return "FP" if self.rejected else "TN"


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, rpm: int = 200):
        self._lock = threading.Lock()
        self._rpm = rpm
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
            new_rpm = max(20, self._rpm // 2)
            if new_rpm < self._rpm:
                self._rpm = new_rpm
                self._interval = 60.0 / new_rpm


_rl = _RateLimiter()


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _extract_url(text: str) -> str:
    """Extract first URL from a search result file."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("http"):
            return line
    return ""


def _check_one(filepath: str, cutoff: str, max_retries: int = 3) -> str:
    """Ask LLM whether this search result contains post-cutoff info.

    Returns (verdict, reason) where verdict is LEAK or CLEAN.
    """
    with open(filepath) as f:
        content = f.read()

    # Truncate very long results
    if len(content) > 4000:
        content = content[:4000] + "\n...[truncated]"

    prompt = (
        f"You are auditing search results for information leakage.\n"
        f"The knowledge cutoff date is {cutoff}. Any information about events, "
        f"outcomes, decisions, or data from AFTER {cutoff} is a leak.\n\n"
        f"Search result:\n{content}\n\n"
        f"Does this search result contain any information from after {cutoff}?\n"
        f"Reply with exactly one line: LEAK or CLEAN, followed by a brief reason.\n"
        f"Example: LEAK - mentions election results from December 2025\n"
        f"Example: CLEAN - all content is from before the cutoff"
    )

    for attempt in range(max_retries + 1):
        _rl.wait()
        try:
            text, _, _, _ = chat(prompt, model=_DETECTIVE_MODEL, max_tokens=150)
            text = (text or "").strip()
            if text.upper().startswith("LEAK"):
                return "LEAK", text[4:].strip().lstrip("- ").strip()
            else:
                return "CLEAN", text[5:].strip().lstrip("- ").strip() if text.upper().startswith("CLEAN") else text
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                _rl.reduce()
                time.sleep(2 ** attempt)
                continue
            raise
    return "CLEAN", "max retries"


def _check_reasoning(forecast_path: str, cutoff: str,
                     max_retries: int = 3) -> tuple[str, str]:
    """Check an LLM reasoning trace for post-cutoff knowledge leakage.

    Examines thinking, reasoning_text, and final reasoning from a forecast file.
    Returns (verdict, reason) where verdict is LEAK or CLEAN.
    """
    with open(forecast_path) as f:
        fc = json.load(f)

    # Collect all reasoning text
    parts = []
    for entry in fc.get("tool_log", []):
        if entry.get("thinking"):
            parts.append(entry["thinking"])
        if entry.get("reasoning_text"):
            parts.append(entry["reasoning_text"])
    final_reasoning = fc.get("reasoning", "")
    if final_reasoning:
        parts.append(final_reasoning)

    if not parts:
        return "CLEAN", "no reasoning text"

    combined = "\n---\n".join(parts)
    if len(combined) > 6000:
        combined = combined[:6000] + "\n...[truncated]"

    prompt = (
        f"You are auditing an LLM's reasoning trace for knowledge leakage.\n"
        f"The knowledge cutoff date is {cutoff}. The LLM should only know "
        f"information from BEFORE this date.\n\n"
        f"Check whether the reasoning reveals knowledge of specific events, "
        f"outcomes, data points, or decisions from AFTER {cutoff}. "
        f"General knowledge and reasoning are fine — only flag specific "
        f"post-cutoff FACTS (e.g. 'the election was won by X in March 2026', "
        f"'the stock price fell to Y on date Z').\n\n"
        f"Reasoning trace:\n{combined}\n\n"
        f"Reply with exactly one line: LEAK or CLEAN, followed by a brief reason.\n"
        f"Example: LEAK - mentions specific stock price from January 2026\n"
        f"Example: CLEAN - reasoning uses only general knowledge and pre-cutoff facts"
    )

    for attempt in range(max_retries + 1):
        _rl.wait()
        try:
            text, _, _, _ = chat(prompt, model=_DETECTIVE_MODEL, max_tokens=200)
            text = (text or "").strip()
            if text.upper().startswith("LEAK"):
                return "LEAK", text[4:].strip().lstrip("- ").strip()
            else:
                reason = text[5:].strip().lstrip("- ").strip() if text.upper().startswith("CLEAN") else text
                return "CLEAN", reason
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                _rl.reduce()
                time.sleep(2 ** attempt)
                continue
            raise
    return "CLEAN", "max retries"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def _parse_filter_log(search_dir: str, search_idx: int) -> dict:
    """Parse filter log to get per-result KEEP/DROP reasons."""
    log_path = os.path.join(search_dir, f"search_{search_idx}_filter_log.txt")
    reasons = {}
    if not os.path.exists(log_path):
        return reasons
    with open(log_path) as f:
        for line in f:
            m = re.match(r'\[(\d+)\]\s+(KEEP|DROP)\s*(.*)', line.strip())
            if m:
                num = int(m.group(1))
                reason = m.group(3).strip().lstrip("(").rstrip(")").strip()
                reasons[num] = (m.group(2), reason)
    return reasons


def _collect_search_tasks(config_name, trial_dirs, exam, cutoff_map):
    """Collect all search result files to check.

    Returns list of (filepath, config, trial, source, qid, rejected, cutoff, filter_reason).
    """
    tasks = []
    for trial_dir in trial_dirs:
        trial = os.path.basename(trial_dir)
        if trial == config_name:
            trial = "single"
        for source, ids in exam.items():
            for qid in ids:
                cutoff = cutoff_map.get((source, qid), "")
                if not cutoff:
                    continue
                safe_id = re.sub(r'[/\\:]', '_', str(qid))
                search_dir = os.path.join(trial_dir, source, "searches", safe_id)
                if not os.path.isdir(search_dir):
                    continue

                # Parse filter logs for this question
                filter_reasons = {}
                for log_f in glob.glob(os.path.join(search_dir, "search_*_filter_log.txt")):
                    m = re.search(r'search_(\d+)_filter_log', log_f)
                    if m:
                        sidx = int(m.group(1))
                        for num, val in _parse_filter_log(search_dir, sidx).items():
                            filter_reasons[(sidx, num)] = val

                # Kept results
                for f in sorted(glob.glob(os.path.join(search_dir, "search_*_result_*.md"))):
                    tasks.append((f, config_name, trial, source, qid, False, cutoff, "KEEP"))

                # lookup_url results
                for f in sorted(glob.glob(os.path.join(search_dir, "lookup_*.md"))):
                    tasks.append((f, config_name, trial, source, qid, False, cutoff, "lookup_url"))

                # Rejected results
                rej_dir = os.path.join(search_dir, "rejected")
                if os.path.isdir(rej_dir):
                    for f in sorted(glob.glob(os.path.join(rej_dir, "search_*_result_*.md"))):
                        fname = os.path.basename(f)
                        m = re.match(r'search_(\d+)_result_(\d+)', fname)
                        fr = ""
                        if m:
                            sidx, rnum = int(m.group(1)), int(m.group(2))
                            dec_reason = filter_reasons.get((sidx, rnum + 1))
                            if dec_reason:
                                fr = f"DROP: {dec_reason[1]}"
                        tasks.append((f, config_name, trial, source, qid, True, cutoff, fr))
    return tasks


def _collect_reasoning_tasks(config_name, trial_dirs, exam, cutoff_map):
    """Collect all reasoning trace files to check.

    Returns list of (forecast_path, config, trial, source, qid, cutoff).
    """
    tasks = []
    for trial_dir in trial_dirs:
        trial = os.path.basename(trial_dir)
        if trial == config_name:
            trial = "single"
        for source, ids in exam.items():
            for qid in ids:
                cutoff = cutoff_map.get((source, qid), "")
                if not cutoff:
                    continue
                safe_id = re.sub(r'[/\\:]', '_', str(qid))
                fc_path = os.path.join(trial_dir, source, f"{safe_id}.json")
                if os.path.exists(fc_path):
                    tasks.append((fc_path, config_name, trial, source, qid, cutoff))
    return tasks


def _check_search_with_cache(filepath, q_cutoff, cfg, source, rejected,
                             cache_dir, force):
    """Check one search result file, using cache if available."""
    fname = os.path.basename(filepath)
    sub = "rejected" if rejected else "kept"
    cache_path = os.path.join(cache_dir, cfg, source, sub, f"{fname}.json") if cache_dir else ""

    if cache_path and not force and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["verdict"], cached["reason"]

    verdict, reason = _check_one(filepath, q_cutoff)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"verdict": verdict, "reason": reason,
                       "cutoff": q_cutoff, "file": filepath}, f, indent=2)
    return verdict, reason


def _check_reasoning_with_cache(fc_path, q_cutoff, cfg, source,
                                cache_dir, force):
    """Check one reasoning trace, using cache if available."""
    fname = os.path.basename(fc_path)
    cache_path = os.path.join(cache_dir, cfg, source, "reasoning",
                              f"{fname}.json") if cache_dir else ""

    if cache_path and not force and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return cached["verdict"], cached["reason"]

    verdict, reason = _check_reasoning(fc_path, q_cutoff)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"verdict": verdict, "reason": reason,
                       "cutoff": q_cutoff, "file": fc_path}, f, indent=2)
    return verdict, reason


def _run_search_checks(tasks, cache_dir, force, max_workers):
    """Run LLM leak checks on search result files in parallel.

    Returns (results, counts).
    """
    results = []
    counts = {"kept_clean": 0, "kept_leak": 0, "rej_clean": 0, "rej_leak": 0}

    # Count cached
    n_cached = 0
    if cache_dir and not force:
        for filepath, cfg, trial, source, qid, rejected, q_cutoff, fr in tasks:
            fname = os.path.basename(filepath)
            sub = "rejected" if rejected else "kept"
            cp = os.path.join(cache_dir, cfg, source, sub, f"{fname}.json")
            if os.path.exists(cp):
                n_cached += 1
    print(f"    {len(tasks)} files ({n_cached} cached, {len(tasks) - n_cached} to classify)")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for filepath, cfg, trial, source, qid, rejected, q_cutoff, fr in tasks:
            fut = pool.submit(_check_search_with_cache, filepath, q_cutoff,
                              cfg, source, rejected, cache_dir, force)
            futures[fut] = (filepath, cfg, trial, source, qid, rejected, q_cutoff, fr)

        n_done = 0
        for fut in as_completed(futures):
            filepath, cfg, trial, source, qid, rejected, q_cutoff, fr = futures[fut]
            try:
                verdict, reason = fut.result()
            except Exception as e:
                verdict, reason = "ERROR", str(e)

            n_done += 1
            is_leak = (verdict == "LEAK")
            if rejected:
                counts["rej_leak" if is_leak else "rej_clean"] += 1
            else:
                counts["kept_leak" if is_leak else "kept_clean"] += 1

            with open(filepath) as f:
                content = f.read()
            url = _extract_url(content)
            results.append(CheckResult(
                config=cfg, trial=trial, source=source, qid=qid,
                search_file=os.path.basename(filepath),
                url=url, cutoff=q_cutoff,
                verdict=verdict, reason=reason,
                filter_reason=fr, rejected=rejected,
            ))
            if n_done % 200 == 0:
                print(f"    {n_done}/{len(tasks)} checked, "
                      f"{counts['kept_leak']} kept leaks, {counts['rej_leak']} rej leaks")

    return results, counts


def _run_reasoning_checks(reasoning_tasks, cache_dir, force):
    """Check reasoning traces for parametric knowledge leakage.

    Returns list of CheckResult for leaks only.
    """
    results = []
    n_leaks = 0
    for fc_path, cfg, trial, source, qid, q_cutoff in reasoning_tasks:
        verdict, reason = _check_reasoning_with_cache(
            fc_path, q_cutoff, cfg, source, cache_dir, force)
        if verdict == "LEAK":
            n_leaks += 1
            fname = os.path.basename(fc_path)
            results.append(CheckResult(
                config=cfg, trial=trial, source=source, qid=qid,
                search_file=f"reasoning:{fname}",
                url="", cutoff=q_cutoff,
                verdict="LEAK", reason=f"[reasoning] {reason}",
                filter_reason="reasoning_trace", rejected=False,
            ))
    return results, n_leaks


def scan_config(config_name: str, exam: dict,
                cutoff_map: dict[tuple[str, str], str],
                max_workers: int = 30,
                cache_dir: str = "",
                force: bool = False,
                reasoning_only: bool = False) -> tuple[list[CheckResult], dict]:
    """Scan search results and/or reasoning traces for a config.

    Returns (leaks, counts) where counts = {kept_clean, kept_leak, rej_clean, rej_leak}.
    """
    base = os.path.join("experiments", "forecasts", config_name)
    trial_dirs = sorted(glob.glob(os.path.join(base, "trial_*")))
    trial_dirs.append(base)

    results = []
    counts = {"kept_clean": 0, "kept_leak": 0, "rej_clean": 0, "rej_leak": 0}

    # Search result checks
    if not reasoning_only:
        tasks = _collect_search_tasks(config_name, trial_dirs, exam, cutoff_map)
        if tasks:
            search_results, counts = _run_search_checks(
                tasks, cache_dir, force, max_workers)
            results.extend(search_results)
            print(f"  [{config_name}] {sum(counts.values())} search files: "
                  f"kept={counts['kept_clean']}clean+{counts['kept_leak']}leak, "
                  f"rejected={counts['rej_clean']}clean+{counts['rej_leak']}leak")

    # Reasoning trace checks
    reasoning_tasks = _collect_reasoning_tasks(
        config_name, trial_dirs, exam, cutoff_map)
    if reasoning_tasks:
        print(f"  [{config_name}] Checking {len(reasoning_tasks)} reasoning traces...")
        reasoning_results, n_leaks = _run_reasoning_checks(
            reasoning_tasks, cache_dir, force)
        results.extend(reasoning_results)
        print(f"  [{config_name}] {len(reasoning_tasks)} reasoning traces: "
              f"{n_leaks} leaks")

    if not results and not any(counts.values()):
        print(f"  [{config_name}] No files found to check")

    return results, counts


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _esc(text):
    return html_lib.escape(str(text)) if text else ""


def _confusion_matrix_html(counts: dict) -> str:
    kc = counts.get("kept_clean", 0)
    kl = counts.get("kept_leak", 0)
    rc = counts.get("rej_clean", 0)
    rl = counts.get("rej_leak", 0)
    total = kc + kl + rc + rl
    contamination = kl / (kc + kl) * 100 if (kc + kl) > 0 else 0
    filter_miss = kl / (kl + rl) * 100 if (kl + rl) > 0 else 0
    filter_fp = rc / (rc + kc) * 100 if (rc + kc) > 0 else 0
    return f"""<table style="font-size:14px;border-collapse:collapse;margin:8px 0">
<tr><th></th><th style="padding:6px 16px">Clean = Negative</th><th style="padding:6px 16px">Leak = Positive</th><th>Total</th></tr>
<tr><td style="font-weight:bold;padding:4px 12px">Kept</td>
    <td style="background:#c6efce;text-align:center">{kc} <span style="color:#888;font-size:0.85em">TN</span></td>
    <td style="background:#ffc7ce;text-align:center;font-weight:bold">{kl} <span style="color:#c0392b;font-size:0.85em">FN</span></td>
    <td style="text-align:center">{kc+kl}</td></tr>
<tr><td style="font-weight:bold;padding:4px 12px">Rejected</td>
    <td style="background:#fce4d6;text-align:center">{rc} <span style="color:#e67e22;font-size:0.85em">FP</span></td>
    <td style="background:#c6efce;text-align:center">{rl} <span style="color:#888;font-size:0.85em">TP</span></td>
    <td style="text-align:center">{rc+rl}</td></tr>
<tr><td style="font-weight:bold;padding:4px 12px">Total</td>
    <td style="text-align:center">{kc+rc}</td>
    <td style="text-align:center">{kl+rl}</td>
    <td style="text-align:center;font-weight:bold">{total}</td></tr>
</table>
<p style="font-size:13px;margin:8px 0">
<b>Contamination rate</b> (FN / kept) = {kl}/{kc+kl} = <b>{contamination:.1f}%</b>
of results shown to the agent may contain post-cutoff info.<br>
Filter miss rate (FN / all leaks) = {kl}/{kl+rl} = {filter_miss:.0f}%.
Filter false positive rate (FP / all clean) = {rc}/{rc+kc} = {filter_fp:.0f}%.
</p>"""


def generate_report(all_leaks: list[CheckResult], output_path: str,
                    xid_name: str, configs_checked: list[str],
                    model: str = "",
                    total_counts: dict | None = None):
    """Generate search_analysis.html report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Summary stats
    by_config = defaultdict(list)
    for lr in all_leaks:
        by_config[lr.config].append(lr)

    kept_leaks = [lr for lr in all_leaks if not lr.rejected]
    rejected_leaks = [lr for lr in all_leaks if lr.rejected]

    summary_rows = ""
    for cfg in configs_checked:
        leaks = by_config.get(cfg, [])
        kept = sum(1 for lr in leaks if not lr.rejected)
        rej = sum(1 for lr in leaks if lr.rejected)
        summary_rows += (f'<tr><td>{_esc(cfg)}</td>'
                         f'<td>{kept}</td><td>{rej}</td></tr>\n')

    # Detail rows grouped by (source, qid), showing ALL results
    from itertools import groupby
    sorted_results = sorted(all_leaks,
                            key=lambda x: (x.source, x.qid, not x.rejected, x.search_file))

    _ERROR_STYLE = {
        "FN": 'color:#c0392b;font-weight:bold',  # kept + leak — RED
        "FP": 'color:#e67e22;font-weight:bold',   # rejected + clean — ORANGE
        "TP": 'color:#2980b9',                     # rejected + leak — blue
        "TN": 'color:#888',                        # kept + clean — gray
    }

    detail_html = ""
    if not all_leaks:
        detail_html = "<p>No results to show.</p>"
    else:
        for (source, qid), group in groupby(sorted_results, key=lambda x: (x.source, x.qid)):
            items = list(group)
            error_counts = {}
            for r in items:
                et = r.error_type
                error_counts[et] = error_counts.get(et, 0) + 1
            # Highlight FN count
            fn = error_counts.get("FN", 0)
            fp = error_counts.get("FP", 0)
            summary_parts = []
            if fn:
                summary_parts.append(f'<span style="color:#c0392b;font-weight:bold">{fn} FN</span>')
            if fp:
                summary_parts.append(f'<span style="color:#e67e22">{fp} FP</span>')
            summary_parts.append(f'{error_counts.get("TP", 0)} TP')
            summary_parts.append(f'{error_counts.get("TN", 0)} TN')

            cutoff_str = items[0].cutoff if items else ""
            detail_html += (
                f'<h3 style="margin-top:16px">{_esc(source)}/{_esc(qid)}'
                f' <span style="font-size:0.8em;font-weight:normal;color:#666">'
                f'cutoff={cutoff_str}</span>'
                f' <span style="font-size:0.8em;font-weight:normal">'
                f'({", ".join(summary_parts)})</span></h3>\n'
                f'<table><tr><th>File</th><th>Verdict</th><th>Verdict reason</th>'
                f'<th>Filter</th><th>Filter reason</th><th>Error</th></tr>\n')
            for r in items:
                et = r.error_type
                style = _ERROR_STYLE.get(et, "")
                verdict_style = "color:#c0392b" if r.verdict == "LEAK" else "color:#27ae60"
                # File as clickable link to URL
                if r.url:
                    file_cell = (f'<a href="{_esc(r.url)}" target="_blank" '
                                 f'title="{_esc(r.url)}">{_esc(r.search_file)}</a>')
                else:
                    file_cell = _esc(r.search_file)
                # Truncatable reasons
                def _trunc(text, n=60):
                    if not text or len(text) <= n:
                        return _esc(text or "")
                    short = _esc(text[:n])
                    full = _esc(text)
                    return (f'<details style="display:inline"><summary style="cursor:pointer">'
                            f'{short}...</summary>{full}</details>')
                detail_html += (
                    f'<tr><td>{file_cell}</td>'
                    f'<td style="{verdict_style}">{r.verdict}</td>'
                    f'<td style="font-size:0.85em">{_trunc(r.reason)}</td>'
                    f'<td>{"rejected" if r.rejected else "kept"}</td>'
                    f'<td style="font-size:0.85em">{_trunc(r.filter_reason)}</td>'
                    f'<td style="{style}">{et}</td></tr>\n')
            detail_html += "</table>\n"

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Leak Detective — {_esc(xid_name)}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f5f5f5; }}
h1 {{ font-size: 1.4em; }}
h2 {{ font-size: 1.1em; margin-top: 28px; }}
.note {{ color: #666; font-size: 0.9em; }}
</style></head><body>

<h1>Leak Detective Report — {_esc(xid_name)}</h1>
<p class="note">
Detective model: {_esc(model)} (classifies each search result as LEAK or CLEAN)<br>
Forecast configs: {', '.join(configs_checked)} (the search filter that decided kept/rejected ran during prediction)
</p>

<h2>Filter confusion matrix</h2>
{_confusion_matrix_html(total_counts) if total_counts else "<p>(counts not available)</p>"}
<p class="note">
<b>Kept + Clean</b> = correctly kept pre-cutoff results (true negatives)&nbsp;&nbsp;
<b>Kept + Leak</b> = post-cutoff info shown to agent (false negatives — <b>concerning</b>)<br>
<b>Rejected + Leak</b> = correctly rejected post-cutoff results (true positives)&nbsp;&nbsp;
<b>Rejected + Clean</b> = incorrectly rejected pre-cutoff results (false positives — useful info lost)
</p>

<h2>All results by question</h2>
<p class="note">Sorted: rejected files first, then kept.
<span style="color:#c0392b;font-weight:bold">FN</span> = kept+leak (agent saw post-cutoff info) &nbsp;
<span style="color:#e67e22">FP</span> = rejected+clean (useful info lost) &nbsp;
<span style="color:#2980b9">TP</span> = rejected+leak (filter caught it) &nbsp;
<span style="color:#888">TN</span> = kept+clean (correct)
</p>
{detail_html}


</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect post-cutoff information leakage in search results")
    parser.add_argument("--xid", required=True, help="Experiment ID")
    parser.add_argument("--config", default=None,
                        help="Comma-separated config names (default: all in xid)")
    parser.add_argument("--max-workers", type=int, default=30,
                        help="Max parallel LLM calls (default: 30)")
    parser.add_argument("--model", default="openrouter/x-ai/grok-4.1-fast",
                        help="LLM model for detection")
    parser.add_argument("--force", action="store_true",
                        help="Re-check all files (ignore cached verdicts)")
    parser.add_argument("--reasoning-only", action="store_true",
                        help="Only check reasoning traces (skip search results)")
    args = parser.parse_args()

    global _DETECTIVE_MODEL
    _DETECTIVE_MODEL = args.model

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.eval import load_xid, load_exam, _PSEUDO_CONFIGS

    xid_data = load_xid(args.xid)
    exam_name = xid_data["exam"]
    exam = load_exam(exam_name)

    if args.config:
        config_names = [c.strip() for c in args.config.split(",")]
    else:
        config_names = xid_data.get("config", [])
        if isinstance(config_names, str):
            config_names = [config_names]
    # Resolve delta strings to filesystem names
    from config.config import resolve_config, pprint_path
    resolved = []
    for c in config_names:
        if c in _PSEUDO_CONFIGS or c.endswith(("_calibrated",)):
            continue
        if "/" in c and ":" in c:
            cfg = resolve_config(c)
            resolved.append(pprint_path(cfg))
        else:
            resolved.append(c)
    # Skip configs with no web search (they have no search results to check)
    # Unless --reasoning-only, which checks all configs
    if not args.reasoning_only:
        final = []
        for c in resolved:
            cfg = resolve_config(c)
            if cfg.search_engine != "none":
                final.append(c)
        config_names = final
    else:
        config_names = resolved

    if not config_names:
        sys.exit("No configs with search results to check.")

    # Build per-question cutoff lookup
    cutoff_map = {}  # (source, qid) -> cutoff date
    for source, ids in exam.items():
        for qid in ids:
            safe_id = re.sub(r'[/\\:]', '_', str(qid))
            q_path = os.path.join("data", "questions", source, f"{safe_id}.json")
            if os.path.exists(q_path):
                with open(q_path) as f:
                    q = json.load(f)
                cutoff_map[(source, qid)] = q.get("forecast_due_date", "")[:10]

    print(f"Leak Detective: {args.xid}")
    print(f"  Exam: {exam_name}, {len(cutoff_map)} questions with cutoffs")
    print(f"  Configs: {config_names}")
    print(f"  Model: {_DETECTIVE_MODEL}")

    cache_dir = os.path.join("experiments", "eval", args.xid, "search_analysis")

    all_leaks = []
    total_counts = {"kept_clean": 0, "kept_leak": 0, "rej_clean": 0, "rej_leak": 0}
    for cfg in config_names:
        leaks, counts = scan_config(cfg, exam, cutoff_map,
                                    max_workers=args.max_workers,
                                    cache_dir=cache_dir,
                                    force=args.force,
                                    reasoning_only=args.reasoning_only)
        all_leaks.extend(leaks)
        for k in total_counts:
            total_counts[k] += counts[k]

    # Generate report
    output_dir = os.path.join("experiments", "eval", args.xid)
    output_path = os.path.join(output_dir, "search_analysis.html")
    generate_report(all_leaks, output_path, args.xid, config_names,
                    model=_DETECTIVE_MODEL, total_counts=total_counts)

    tc = total_counts
    n_search = sum(tc.values())
    n_reasoning_leaks = sum(1 for l in all_leaks if l.filter_reason == "reasoning_trace")
    n_reasoning_total = 0
    # Count total reasoning traces checked (leaks + clean)
    for cfg in config_names:
        base = os.path.join("experiments", "forecasts", cfg)
        trial_dirs = sorted(glob.glob(os.path.join(base, "trial_*")))
        trial_dirs.append(base)
        for td in trial_dirs:
            for source, ids in exam.items():
                for qid in ids:
                    safe_id = re.sub(r'[/\\:]', '_', str(qid))
                    if os.path.exists(os.path.join(td, source, f"{safe_id}.json")):
                        n_reasoning_total += 1

    if n_search > 0:
        print(f"\nSearch results: {tc['kept_clean']} TN, {tc['kept_leak']} FN, "
              f"{tc['rej_clean']} FP, {tc['rej_leak']} TP")
        if tc['kept_leak'] > 0:
            print(f"  WARNING: {tc['kept_leak']} leaks in kept results")
    print(f"Reasoning traces: {n_reasoning_leaks}/{n_reasoning_total} leaks "
          f"({n_reasoning_leaks/n_reasoning_total*100:.1f}%)" if n_reasoning_total > 0
          else "Reasoning traces: none checked")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
