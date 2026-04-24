#!/usr/bin/env python3
"""convert_legacy_data.py — Convert v5-era forecasts and exams to v6 format.

v5 used bare question IDs (e.g. "1722") while v6 uses date-stamped IDs
(e.g. "1722_2026-02-15"). This script:

1. Copies v5 forecast files to v6, renaming with date-stamped IDs
2. Rebuilds the exam indices with date-stamped IDs
3. Updates the xid to v6 syntax (reference field, groups format)

Usage:
    python3 src/convert_legacy_data.py --xid xid-market-both --v5-dir ../forecast-bench-v5

    # Dry run (no files written)
    python3 src/convert_legacy_data.py --xid xid-market-both --v5-dir ../forecast-bench-v5 --dry-run
"""

import argparse
import json
import os
import re
import shutil
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))


def _load_xid(xid: str) -> dict:
    path = f"experiments/xids/{xid}.json"
    with open(path) as f:
        return json.load(f)


def _load_exam(exam_name: str) -> dict:
    path = os.path.join("data", "exams", exam_name, "indices.json")
    with open(path) as f:
        return json.load(f)


def _safe_id(qid: str) -> str:
    return re.sub(r'[/\\:]', '_', str(qid))


def convert_forecasts(v5_dir: str, configs: list[str], exam: dict,
                      dry_run: bool = False) -> dict[str, list[str]]:
    """Copy v5 forecasts to v6, renaming with date-stamped IDs.

    Returns {(source, bare_id): date_stamped_id} mapping for exam update.
    """
    v5_fc_dir = os.path.join(v5_dir, "results", "forecasts")
    v6_fc_dir = os.path.join("experiments", "forecasts")

    id_mapping = {}  # (source, bare_id) -> date_stamped_id
    total_copied = 0
    total_skipped = 0

    for config in configs:
        v5_config_dir = os.path.join(v5_fc_dir, config)
        if not os.path.isdir(v5_config_dir):
            print(f"  [{config}] not found in v5, skipping")
            continue

        n_copied = 0
        for source, ids in exam.items():
            for bare_id in ids:
                safe = _safe_id(bare_id)
                v5_path = os.path.join(v5_config_dir, source, f"{safe}.json")
                if not os.path.exists(v5_path):
                    # Try trial dirs
                    continue

                # Read to get forecast_due_date
                with open(v5_path) as f:
                    fc = json.load(f)
                fdd = fc.get("forecast_due_date", "")
                if not fdd:
                    print(f"  WARNING: {v5_path} has no forecast_due_date, skipping")
                    continue

                # Construct date-stamped ID
                date_stamped_id = f"{bare_id}_{fdd}"
                id_mapping[(source, bare_id)] = date_stamped_id

                # Update the forecast's id field
                fc["id"] = date_stamped_id

                # Write to v6
                v6_safe = _safe_id(date_stamped_id)
                v6_dir = os.path.join(v6_fc_dir, config, source)
                v6_path = os.path.join(v6_dir, f"{v6_safe}.json")

                if os.path.exists(v6_path):
                    total_skipped += 1
                    continue

                if not dry_run:
                    os.makedirs(v6_dir, exist_ok=True)
                    with open(v6_path, "w") as f:
                        json.dump(fc, f, indent=2)
                n_copied += 1

                # Also copy trial files if they exist
                for trial_dir_name in os.listdir(v5_config_dir):
                    if not trial_dir_name.startswith("trial_"):
                        continue
                    v5_trial_path = os.path.join(v5_config_dir, trial_dir_name,
                                                  source, f"{safe}.json")
                    if not os.path.exists(v5_trial_path):
                        continue
                    with open(v5_trial_path) as f:
                        trial_fc = json.load(f)
                    trial_fc["id"] = date_stamped_id
                    v6_trial_dir = os.path.join(v6_fc_dir, config, trial_dir_name, source)
                    v6_trial_path = os.path.join(v6_trial_dir, f"{v6_safe}.json")
                    if not os.path.exists(v6_trial_path) and not dry_run:
                        os.makedirs(v6_trial_dir, exist_ok=True)
                        with open(v6_trial_path, "w") as f:
                            json.dump(trial_fc, f, indent=2)

        total_copied += n_copied
        if n_copied:
            print(f"  [{config}] copied {n_copied} forecasts")

    if total_skipped:
        print(f"  ({total_skipped} already existed, skipped)")
    print(f"  Total: {total_copied} forecasts copied")
    return id_mapping


def update_exam_indices(exam_name: str, id_mapping: dict,
                        dry_run: bool = False) -> int:
    """Update exam indices.json to use date-stamped IDs."""
    path = os.path.join("data", "exams", exam_name, "indices.json")
    with open(path) as f:
        exam = json.load(f)

    n_updated = 0
    new_exam = {}
    for source, ids in exam.items():
        new_ids = []
        for bare_id in ids:
            mapped = id_mapping.get((source, bare_id))
            if mapped:
                new_ids.append(mapped)
                n_updated += 1
            else:
                # Check if already date-stamped
                if re.search(r'_\d{4}-\d{2}-\d{2}$', bare_id):
                    new_ids.append(bare_id)
                else:
                    # Try to find matching question file
                    q_dir = os.path.join("data", "questions", source)
                    if os.path.isdir(q_dir):
                        matches = [f.removesuffix(".json") for f in os.listdir(q_dir)
                                   if f.startswith(bare_id + "_") and f.endswith(".json")]
                        if len(matches) == 1:
                            new_ids.append(matches[0])
                            n_updated += 1
                        elif matches:
                            # Multiple dates — keep bare ID (manual resolution needed)
                            print(f"  WARNING: {source}/{bare_id} matches {len(matches)} "
                                  f"question files, keeping bare ID")
                            new_ids.append(bare_id)
                        else:
                            new_ids.append(bare_id)
                    else:
                        new_ids.append(bare_id)
        new_exam[source] = new_ids

    if not dry_run:
        with open(path, "w") as f:
            json.dump(new_exam, f, indent=2)
    print(f"  Updated {n_updated} IDs in {path}")
    return n_updated


def update_xid(xid: str, dry_run: bool = False):
    """Update xid to v6 syntax: move references to 'reference' field,
    convert legacy 'group' dict to 'groups' list, rename old config names."""
    path = f"experiments/xids/{xid}.json"
    with open(path) as f:
        data = json.load(f)

    changed = False

    # Move reference configs from 'config' to 'reference'
    _REF_NAMES = {"sota", "superhuman", "baseline"}
    configs = data.get("config", [])
    if isinstance(configs, str):
        configs = [configs]
    refs = [c for c in configs if c in _REF_NAMES]
    real_configs = [c for c in configs if c not in _REF_NAMES]
    if refs:
        data["config"] = real_configs
        # baseline stays in config (it's a scoring method, not a reference)
        ref_only = [r for r in refs if r != "baseline"]
        if ref_only:
            existing_refs = data.get("reference", [])
            data["reference"] = list(dict.fromkeys(existing_refs + ref_only))
        if "baseline" in refs:
            data["config"] = ["baseline"] + real_configs
        changed = True

    # Convert legacy 'group' dict to 'groups' list
    if "group" in data and "groups" not in data:
        old_group = data.pop("group")
        # Old format: {"polymarket": "polymarket", "overall": "all"}
        # New format: ["Qsource", "overall"]
        # Detect if it's a source-based grouping
        sources = [k for k in old_group if k != "overall"]
        if sources:
            data["groups"] = ["Qsource", "overall"]
        changed = True

    # Rename old-style config names (crowd0/1 -> c0/c1 etc.)
    # v5 used names like "pro-high-brave-crowd1-tools1"
    # v6 uses "pro-high-brave-c1-t1"
    # Keep both as valid — the forecast directories use the old names

    if changed and not dry_run:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Updated {path}")
    elif changed:
        print(f"  Would update {path}")
    else:
        print(f"  {path} already up to date")


def main():
    parser = argparse.ArgumentParser(
        description="Convert v5 forecasts and exams to v6 format")
    parser.add_argument("--xid", required=True, help="Experiment ID")
    parser.add_argument("--v5-dir", default="../forecast-bench-v5",
                        help="Path to v5 directory (default: ../forecast-bench-v5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing files")
    parser.add_argument("--skip-forecasts", action="store_true",
                        help="Skip forecast copying (only update exam/xid)")
    args = parser.parse_args()

    xid_data = _load_xid(args.xid)
    exam_name = xid_data["exam"]
    exam = _load_exam(exam_name)

    # Get real configs (not references)
    _REFS = {"sota", "superhuman", "baseline"}
    configs = xid_data.get("config", [])
    if isinstance(configs, str):
        configs = [configs]
    configs = [c for c in configs if c not in _REFS]

    total_questions = sum(len(ids) for ids in exam.values())
    print(f"Converting {args.xid}")
    print(f"  Exam: {exam_name} ({total_questions} questions)")
    print(f"  Configs: {configs}")
    if args.dry_run:
        print("  [DRY RUN — no files will be written]")

    # Step 1: Copy forecasts from v5 to v6 with date-stamped IDs
    id_mapping = {}
    if not args.skip_forecasts:
        print(f"\nStep 1: Copy forecasts from {args.v5_dir}")
        id_mapping = convert_forecasts(args.v5_dir, configs, exam,
                                        dry_run=args.dry_run)
    else:
        print("\nStep 1: Skipped (--skip-forecasts)")

    # Step 2: Update exam indices
    print(f"\nStep 2: Update exam indices")
    update_exam_indices(exam_name, id_mapping, dry_run=args.dry_run)

    # Step 3: Update xid
    print(f"\nStep 3: Update xid")
    update_xid(args.xid, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
