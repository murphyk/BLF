#!/usr/bin/env python3
"""finalize.py — Slim forecast files from experiments/forecasts → experiments/forecasts_final.

Strips heavy trace fields (belief_history, tool_log, system_prompt,
question_prompt) while keeping everything eval.py/calibrate.py need for
leaderboard + plots. Precomputes `tool_counts` and `n_searches` from the
tool_log so eval doesn't need to reread it.

Usage:
    python3 src/core/finalize.py --xid my-xid              # all configs in xid
    python3 src/core/finalize.py --config pro-high-brave-c0-t1
    python3 src/core/finalize.py --all                     # every config dir under forecasts/
    python3 src/core/finalize.py --fb-import v7-path       # slim-copy fb-* from another tree
"""

import argparse
import glob
import json
import os
import shutil  # noqa: kept for future use
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

from config.paths import FORECASTS_DIR, FORECASTS_FINAL_DIR, XIDS_DIR

# Fields to drop when slimming. Two groups:
#
# Heavy trace fields — only used by per-question detail pages, which
# continue to read from experiments/forecasts/.
# Question metadata — duplicated in data/questions/{source}/{id}.json.
# eval.py reconstitutes these from the question file when missing.
STRIP_FIELDS = {
    # Trace-only
    "belief_history",
    "tool_log",
    "system_prompt",
    "question_prompt",
    # Question metadata (reconstituted from data/questions/)
    "question",
    "background",
    "resolution_criteria",
    "url",
    "resolution_date",  # singular; resolution_dates is a list we keep
}


def _precompute_tool_stats(fc: dict) -> dict:
    """Derive tool_counts and n_searches from tool_log before dropping it."""
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


def slim(fc: dict) -> dict:
    """Return a slimmed copy of a forecast dict."""
    stats = _precompute_tool_stats(fc)
    out = {k: v for k, v in fc.items() if k not in STRIP_FIELDS}
    # Only add precomputed stats if we actually had a tool_log to summarize;
    # otherwise leave the keys untouched (fb-* files already lack them).
    if "tool_log" in fc:
        out.setdefault("tool_counts", stats["tool_counts"])
        out.setdefault("n_searches", stats["n_searches"])
    return out


def finalize_file(src_path: str, dst_path: str) -> int:
    """Slim one JSON and write it. Returns bytes saved."""
    with open(src_path) as f:
        fc = json.load(f)
    slim_fc = slim(fc)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(slim_fc, f, separators=(",", ":"))
    return os.path.getsize(src_path) - os.path.getsize(dst_path)


def finalize_config(config_name: str, src_root: str = FORECASTS_DIR,
                    dst_root: str = FORECASTS_FINAL_DIR,
                    verbose: bool = False) -> tuple[int, int]:
    """Slim every JSON under {src_root}/{config_name}/{source}/*.json.

    Skips per-trial directories (trial_*) and any file ending in _trace.html.
    Returns (files_written, total_bytes_saved).
    """
    src_dir = os.path.join(src_root, config_name)
    if not os.path.isdir(src_dir):
        print(f"  SKIP: {src_dir} not a directory")
        return 0, 0

    n = 0
    saved = 0
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if not d.startswith("trial_")]
        for fn in files:
            if not fn.endswith(".json"):
                continue
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, src_root)
            dst = os.path.join(dst_root, rel)
            if fn == "config.json" and os.path.dirname(rel) == config_name:
                # Copy the per-config config.json straight through (no slim).
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
                n += 1
                continue
            saved += finalize_file(src, dst)
            n += 1

    if verbose:
        print(f"  {config_name}: {n} files, saved {saved // 1024} KB")
    return n, saved


def _configs_in_xid(xid: str) -> list[str]:
    """Pull the hyphenated config directory names from an xid's {config, eval}."""
    from config.config import resolve_config, pprint as cfg_pprint

    path = os.path.join(XIDS_DIR, f"{xid}.json")
    with open(path) as f:
        xid_data = json.load(f)
    labels: list[str] = []
    for field in ("config", "eval", "calibrate"):
        for entry in xid_data.get(field, []):
            # strip [agg_key] suffix if any
            name = entry.split("[")[0]
            # resolve delta-string (e.g. "pro/thk:high") to directory name
            try:
                cfg = resolve_config(name)
                labels.append(cfg_pprint(cfg))
            except Exception:
                labels.append(name)
    return sorted(set(labels))


def main() -> None:
    ap = argparse.ArgumentParser(description="Slim forecasts → forecasts_final.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--xid", help="Finalize every config referenced by this xid")
    g.add_argument("--config", help="Finalize a single config by directory name")
    g.add_argument("--all", action="store_true",
                   help="Finalize every top-level config dir under forecasts/")
    g.add_argument("--fb-import", metavar="SRC_ROOT",
                   help="Copy fb-*/ directories from another forecasts/ tree "
                        "(e.g. ../forecast-bench-v7/experiments/forecasts)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.fb_import:
        # These files are already in the minimal FB-leaderboard format; just
        # copy JSONs (no traces) straight across.
        src_root = args.fb_import
        n_dirs = 0
        n_files = 0
        for d in sorted(glob.glob(os.path.join(src_root, "fb-*"))):
            name = os.path.basename(d)
            dst = os.path.join(FORECASTS_FINAL_DIR, name)
            n_dirs += 1
            for src in glob.glob(os.path.join(d, "*", "*.json")):
                rel = os.path.relpath(src, d)
                out = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copyfile(src, out)
                n_files += 1
        print(f"fb-import: {n_dirs} directories, {n_files} JSONs → {FORECASTS_FINAL_DIR}/")
        return

    if args.xid:
        configs = _configs_in_xid(args.xid)
    elif args.config:
        configs = [args.config]
    else:
        configs = sorted(
            d for d in os.listdir(FORECASTS_DIR)
            if os.path.isdir(os.path.join(FORECASTS_DIR, d)) and not d.startswith("fb-")
        )

    total_files = 0
    total_saved = 0
    for c in configs:
        n, saved = finalize_config(c, verbose=args.verbose)
        total_files += n
        total_saved += saved
    print(f"finalize: {len(configs)} configs, {total_files} files, "
          f"saved {total_saved // 1024} KB → {FORECASTS_FINAL_DIR}/")


if __name__ == "__main__":
    main()
