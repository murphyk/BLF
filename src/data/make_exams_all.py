#!/usr/bin/env python3
"""make_exams_all.py — Build all exams and generate data visualizations.

For each exam in data/exams/:
  1. If mixture.json exists, run make_exam to build/rebuild indices.json + meta.json
  2. Generate rdate_by_fdate_scatter.png (resolution date vs forecast date)
  3. Generate tag distribution heatmaps for all classified tag spaces

Exams without mixture.json (e.g. manually created) are left unchanged but
still get visualizations if they have indices.json.

Usage:
    python3 src/make_exams_all.py
    python3 src/make_exams_all.py --exam market-both,dataset-test  # specific exams
"""

import argparse
import json
import os
import subprocess
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Build all exams and generate visualizations")
    parser.add_argument("--exam", default=None,
                        help="Comma-separated exam names (default: all in data/exams/)")
    args = parser.parse_args()

    exams_dir = os.path.join("data", "exams")
    if not os.path.isdir(exams_dir):
        sys.exit(f"ERROR: {exams_dir} not found")

    if args.exam:
        exam_names = [e.strip() for e in args.exam.split(",")]
    else:
        exam_names = sorted(d for d in os.listdir(exams_dir)
                            if os.path.isdir(os.path.join(exams_dir, d)))

    print(f"Processing {len(exam_names)} exam(s)\n")

    # Step 1: Build indices from mixture.json where available
    for name in exam_names:
        mixture_path = os.path.join(exams_dir, name, "mixture.json")
        if os.path.exists(mixture_path):
            print(f"=== {name} (building from mixture.json) ===")
            r = subprocess.run([sys.executable, "src/make_exam.py", "--name", name])
            if r.returncode != 0:
                print(f"  WARNING: make_exam failed for {name}")
        else:
            print(f"=== {name} (no mixture.json, keeping existing indices) ===")
        print()

    # Step 2: Generate visualizations for all exams with indices.json
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eval.eval_plots import generate_rdate_by_fdate_scatter, generate_tag_distribution
    from config.tags import get_tags_for_exam, discover_classified_spaces

    classified = discover_classified_spaces()
    print(f"Generating visualizations (classified tag spaces: {classified})\n")

    for name in exam_names:
        indices_path = os.path.join(exams_dir, name, "indices.json")
        if not os.path.exists(indices_path):
            continue

        with open(indices_path) as f:
            exam = json.load(f)

        total = sum(len(ids) for ids in exam.values())
        if total == 0:
            continue

        # rdate scatter
        p = generate_rdate_by_fdate_scatter(name, exam)
        if p:
            print(f"  {p}")

        # tag distributions
        for ls in classified:
            tags = get_tags_for_exam(exam, ls)
            if tags:
                p = generate_tag_distribution(name, tags, tag_version=ls,
                                              n_total=total)
                if p:
                    print(f"  {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
