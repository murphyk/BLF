#!/usr/bin/env python3
"""collate.py — Collate per-question raw forecasts into one file per (date, config).

Input:  experiments/forecasts_raw/{config}/{source}/{id}_{date}.json (+ optional
        trial_{t}/{source}/... per-trial files)
Output: experiments/forecasts_final/{YYYY-MM-DD}/{config}.json — an
        FB-style per-submission file with one entry per (source, id,
        resolution_date). Multi-resolution dataset questions contribute N
        entries, matching the shape FB publishes in
        data/fb_cache/forecastbench-processed-forecast-sets/.

Top-level shape (same keys as FB's processed forecast sets):
    {
      "organization":       "sirbayes",
      "model":              "pro-high-brave-c1-t1",   // config directory name
      "model_organization": "sirbayes",
      "forecast_due_date":  "2025-10-26",
      "question_set":       "",                       // unused for our runs
      "leaderboard_eligible": true,
      "config": {...},                                // full AgentConfig dump
      "forecasts": [
        {"id": ..., "source": ..., "resolution_date": ..., "forecast": ...,
         "raw_trials": [...],                         // per-trial probs at this res_date
         "resolved_to": ..., "forecast_due_date": ...,
         "market_value": ..., "market_date": ...,
         "reasoning": ..., "n_steps": ..., "submitted": ...,
         "tokens_in": ..., "tokens_out": ..., "elapsed_seconds": ...,
         "tool_counts": {...}, "n_searches": ...},
        ...
      ]
    }

`forecast` is the default aggregation across trials (arithmetic mean for
legacy runs; logit-mean going forward). Further variants added by
aggregate.py live in `forecasts_aggregated: {"mean:1": [...],
"shrink5-loo": [...], ...}`. Calibrated variants added by calibrate.py
live in `forecasts_calibrated: {"global-cal": [...], "hier-cal": [...],
...}`. Both are indexed positionally against `forecasts[]`.

For external ForecastBench methods, `fb_leaderboard.py --import-method`
writes the same shape to experiments/forecasts_final/{date}/fb-{key}.json
by copying the FB tarball file and keeping the native FB schema.

Usage:
    python3 src/core/collate.py --xid my-xid
    python3 src/core/collate.py --config pro-high-brave-c0-t1
    python3 src/core/collate.py --all
    python3 src/core/collate.py --from-raw-root /path/to/v7/experiments/forecasts_raw \\
                                --all                               # migrate from another tree
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

from config.paths import FORECASTS_DIR, FORECASTS_FINAL_DIR, XIDS_DIR


DEFAULT_ORG = "sirbayes"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit_mean(probs: list[float]) -> float:
    xs = [_logit(p) for p in probs if p is not None]
    if not xs:
        return 0.5
    return _sigmoid(sum(xs) / len(xs))


# ---------------------------------------------------------------------------
# Tool-log summarization
# ---------------------------------------------------------------------------

def _summarize_tool_log(fc: dict) -> dict:
    tool_log = fc.get("tool_log", []) or []
    tool_counts: dict[str, int] = {}
    n_searches = 0
    for e in tool_log:
        if e.get("type") != "tool_call":
            continue
        t = e.get("tool", "unknown")
        tool_counts[t] = tool_counts.get(t, 0) + 1
        if t == "web_search":
            n_searches += 1
    return {"tool_counts": tool_counts, "n_searches": n_searches}


# ---------------------------------------------------------------------------
# Per-entry construction
# ---------------------------------------------------------------------------

def _flatten_question(fc: dict, trial_fcs: list[dict]) -> list[dict]:
    """Flatten a raw forecast file into one entry per resolution_date."""
    rdates = fc.get("resolution_dates") or []
    ps_main = fc.get("forecasts") or []
    resolved = fc.get("resolved_to")

    shared_fields = {
        "forecast_due_date": fc.get("forecast_due_date", ""),
        "market_value": fc.get("market_value"),
        "market_date": fc.get("market_date", ""),
        "reasoning": fc.get("reasoning", ""),
        "n_steps": fc.get("n_steps"),
        "submitted": fc.get("submitted"),
        "tokens_in": fc.get("tokens_in"),
        "tokens_out": fc.get("tokens_out"),
        "elapsed_seconds": fc.get("elapsed_seconds"),
        "imputed": fc.get("imputed", False),
    }
    if fc.get("tool_counts") is None and fc.get("n_searches") is None:
        s = _summarize_tool_log(fc)
        shared_fields["tool_counts"] = s["tool_counts"]
        shared_fields["n_searches"] = s["n_searches"]
    else:
        shared_fields["tool_counts"] = fc.get("tool_counts") or {}
        shared_fields["n_searches"] = fc.get("n_searches") or 0

    def _trial_prob(tfc: dict, i: int):
        fs = tfc.get("forecasts")
        if isinstance(fs, list) and i < len(fs):
            return fs[i]
        if i == 0:
            return tfc.get("forecast")
        return None

    entries: list[dict] = []
    if rdates and isinstance(rdates, list):
        n = len(rdates)
        rs = resolved if isinstance(resolved, list) else [resolved] * n
        for i, rd in enumerate(rdates):
            raw_trials = [p for p in (_trial_prob(t, i) for t in trial_fcs)
                          if p is not None]
            main_p = ps_main[i] if i < len(ps_main) else (raw_trials[0] if raw_trials else None)
            entries.append({
                "id": fc.get("id", ""),
                "source": fc.get("source", ""),
                "resolution_date": str(rd),
                "forecast": main_p,
                "raw_trials": raw_trials,
                "resolved_to": rs[i] if i < len(rs) else None,
                **shared_fields,
            })
    else:
        raw_trials = [p for p in (_trial_prob(t, 0) for t in trial_fcs) if p is not None]
        entries.append({
            "id": fc.get("id", ""),
            "source": fc.get("source", ""),
            "resolution_date": fc.get("resolution_date", ""),
            "forecast": fc.get("forecast"),
            "raw_trials": raw_trials,
            "resolved_to": (resolved[0] if isinstance(resolved, list) and resolved
                            else resolved),
            **shared_fields,
        })
    return entries


# ---------------------------------------------------------------------------
# Config-level collation
# ---------------------------------------------------------------------------

def _load_trials(config_name: str, source: str, fn: str,
                 src_root: str) -> list[dict]:
    trials = []
    for tdir in sorted(glob.glob(os.path.join(src_root, config_name, "trial_*"))):
        p = os.path.join(tdir, source, fn)
        if os.path.isfile(p):
            with open(p) as f:
                trials.append(json.load(f))
    return trials


def _load_config_struct(config_name: str, src_root: str) -> dict:
    p = os.path.join(src_root, config_name, "config.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}


def collate_config(config_name: str, src_root: str = FORECASTS_DIR,
                   dst_root: str = FORECASTS_FINAL_DIR,
                   verbose: bool = False) -> tuple[int, int]:
    src_dir = os.path.join(src_root, config_name)
    if not os.path.isdir(src_dir):
        print(f"  SKIP: {src_dir} not a directory")
        return 0, 0

    config_struct = _load_config_struct(config_name, src_root)

    trial_nums = sorted(
        int(m.group(1)) for d in glob.glob(os.path.join(src_dir, "trial_*"))
        if (m := re.match(r".*trial_(\d+)$", d))
    )

    by_date: dict[str, list[dict]] = defaultdict(list)

    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if not d.startswith("trial_")]
        for fn in files:
            if not fn.endswith(".json") or fn == "config.json":
                continue
            with open(os.path.join(root, fn)) as f:
                fc = json.load(f)
            source = fc.get("source") or os.path.basename(root)
            date = fc.get("forecast_due_date", "")
            if not date:
                continue
            trial_fcs = _load_trials(config_name, source, fn, src_root) if trial_nums else [fc]
            for entry in _flatten_question(fc, trial_fcs):
                by_date[date].append(entry)

    n_dates = 0
    n_entries = 0
    for date, entries in sorted(by_date.items()):
        out = {
            "organization": DEFAULT_ORG,
            "model": config_name,
            "model_organization": DEFAULT_ORG,
            "forecast_due_date": date,
            "question_set": "",
            "leaderboard_eligible": True,
            "config": config_struct,
            "forecasts": sorted(entries,
                                key=lambda e: (e["source"], e["id"], e["resolution_date"])),
        }
        dst = os.path.join(dst_root, date, f"{config_name}.json")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        n_dates += 1
        n_entries += len(entries)

    if verbose:
        print(f"  {config_name}: {n_dates} dates, {n_entries} entries")
    return n_dates, n_entries


# ---------------------------------------------------------------------------
# FB-tarball import
# ---------------------------------------------------------------------------

def import_fb_from_tarball(method_key: str, tarball_dir: str,
                           dst_root: str = FORECASTS_FINAL_DIR,
                           model_name: str = "",
                           alt_patterns: list[str] | None = None) -> tuple[int, int]:
    """Copy FB processed-forecast files to forecasts_final/{date}/fb-{safe_key}.json.

    The tarball files are named `{date}.{fb_method_value}.json` (the
    `fb_method` string appears literally in the filename, e.g.
    `2025-10-26.external.Cassi-AI.2.json`). Pass `alt_patterns` if the
    method has variant names across dates (e.g. Lightning-Rod-Labs →
    LightningRodLabs.1).
    """
    safe_key = method_key.replace(".", "-").replace(" ", "-").lower()
    safe_key = re.sub(r"^external-", "", safe_key)
    # The primary filename chunk after the leading "{date}." prefix.
    patterns = [method_key] + (alt_patterns or [])
    n_dates = n_entries = 0
    for date_dir in sorted(glob.glob(os.path.join(tarball_dir, "*"))):
        if not os.path.isdir(date_dir):
            continue
        date = os.path.basename(date_dir)
        matches: list[str] = []
        for pat in patterns:
            matches.extend(glob.glob(os.path.join(date_dir, f"{date}.{pat}.json")))
        if not matches:
            continue
        with open(matches[0]) as f:
            d = json.load(f)
        out = {
            "organization": d.get("organization", "ForecastBench"),
            "model": model_name or d.get("model", ""),
            "model_organization": d.get("model_organization", ""),
            "forecast_due_date": d.get("forecast_due_date", date),
            "question_set": d.get("question_set", ""),
            "leaderboard_eligible": d.get("leaderboard_eligible", True),
            "fb_method": method_key,
            "forecasts": d.get("forecasts", []),
        }
        dst = os.path.join(dst_root, date, f"fb-{safe_key}.json")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        n_dates += 1
        n_entries += len(out["forecasts"])
    return n_dates, n_entries


def import_fb_from_split_tree(src_root: str,
                              dst_root: str = FORECASTS_FINAL_DIR) -> tuple[int, int]:
    """Re-collate v6-style split fb-* dirs (per-question JSONs) into per-date files."""
    n_dates = n_files = 0
    for d in sorted(glob.glob(os.path.join(src_root, "fb-*"))):
        name = os.path.basename(d)
        by_date: dict[str, list[dict]] = defaultdict(list)
        header = {"organization": "ForecastBench",
                  "model_organization": "", "model": name,
                  "fb_method": "", "question_set": "",
                  "leaderboard_eligible": True}
        for src in glob.glob(os.path.join(d, "*", "*.json")):
            with open(src) as f:
                fc = json.load(f)
            # The split-format files encode the date in the filename suffix
            m = re.match(r".*_(\d{4}-\d{2}-\d{2})\.json$", src)
            date = m.group(1) if m else fc.get("forecast_due_date", "")[:10]
            if not date:
                continue
            # Keep metadata from the first entry we see
            header["fb_method"] = fc.get("fb_method", header["fb_method"])
            header["model"] = fc.get("fb_model", header["model"])
            header["model_organization"] = fc.get("fb_organization", header["model_organization"])
            by_date[date].append({
                "id": fc.get("id", ""),
                "source": fc.get("source", ""),
                "resolution_date": date,  # split-format lacks an explicit resolution_date
                "forecast": fc.get("forecast"),
                "resolved_to": fc.get("resolved_to"),
                "imputed": fc.get("imputed", False),
            })
        for date, entries in by_date.items():
            out = {**header,
                   "forecast_due_date": date,
                   "forecasts": sorted(entries, key=lambda e: (e["source"], e["id"]))}
            dst = os.path.join(dst_root, date, f"{name}.json")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                json.dump(out, f, separators=(",", ":"))
            n_dates += 1
            n_files += len(entries)
    return n_dates, n_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configs_in_xid(xid: str) -> list[str]:
    from config.config import resolve_config, pprint as cfg_pprint
    with open(os.path.join(XIDS_DIR, f"{xid}.json")) as f:
        xid_data = json.load(f)
    labels: list[str] = []
    for field in ("config", "eval", "calibrate"):
        for entry in xid_data.get(field, []):
            name = entry.split("[")[0]
            if ":" in name or "/" in name:
                try:
                    cfg = resolve_config(name)
                    labels.append(cfg_pprint(cfg))
                    continue
                except Exception:
                    pass
            labels.append(name)
    return sorted(set(labels))


def main() -> None:
    ap = argparse.ArgumentParser(description="Collate forecasts → forecasts_final/{date}/{config}.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--xid", help="Collate every config referenced by this xid")
    g.add_argument("--config", help="Collate a single config by directory name")
    g.add_argument("--all", action="store_true",
                   help="Collate every top-level config dir under forecasts_raw/")
    ap.add_argument("--from-raw-root", default=FORECASTS_DIR,
                    help=f"Source forecasts_raw tree (default: {FORECASTS_DIR})")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    src_root = args.from_raw_root

    if args.xid:
        configs = _configs_in_xid(args.xid)
    elif args.config:
        configs = [args.config]
    else:
        configs = sorted(
            d for d in os.listdir(src_root)
            if os.path.isdir(os.path.join(src_root, d)) and not d.startswith("fb-")
        )

    total_d = total_e = 0
    for c in configs:
        d, e = collate_config(c, src_root=src_root, verbose=args.verbose)
        total_d += d
        total_e += e
    print(f"collate: {len(configs)} configs → {total_d} date-files, "
          f"{total_e} entries → {FORECASTS_FINAL_DIR}/")


if __name__ == "__main__":
    main()
