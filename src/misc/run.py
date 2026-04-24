#!/usr/bin/env python3
"""run.py — End-to-end pipeline: predict → calibrate → eval.

Usage:
    python3 src/run.py --xid myxid
    python3 src/run.py --xid myxid --verbose --monitor --max-workers 20

Reads the xid file and runs each applicable step:
    1. predict   (always, unless no "config" field)
    2. calibrate (if "calibrate" field present)
    3. eval      (always)

The xid file (experiments/xids/{xid}.json) should contain:
    "exam":      exam name (required)
    "config":    list of configs to predict with
    "calibrate": list of configs to calibrate (optional, triggers calibrate step)
    "eval":      list of configs to evaluate (defaults to "config")
    "metric":    list of metrics (defaults to ["brier-score", "metaculus-score"])
    "group":     group definitions for leaderboard (optional)

Note: ensembling is a post-hoc step run separately via src/ensemble.py.
"""

import argparse
import json
import os
import subprocess
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))


def load_xid(xid: str) -> dict:
    path = f"experiments/xids/{xid}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"XID not found: {path}")
    with open(path) as f:
        return json.load(f)


def run_step(description: str, cmd: list[str]):
    """Run a subprocess, printing the command and checking for errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"ERROR: {description} failed (exit code {result.returncode})")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end: predict → ensemble → calibrate → eval")
    parser.add_argument("--xid", required=True,
                        help="Experiment ID (experiments/xids/{xid}.json)")
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Max parallel workers per config (default: 50)")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to the first N questions from the exam")
    parser.add_argument("--ntrials", type=int, default=1,
                        help="Run each config N times and average (default: 1)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--monitor", action="store_true",
                        help="Write live progress.html during prediction")
    args = parser.parse_args()

    xid_data = load_xid(args.xid)

    # --- Step 1: Predict ---
    if "config" in xid_data:
        predict_cmd = [sys.executable, "src/predict.py", "--xid", args.xid]
        if args.max_workers != 50:
            predict_cmd += ["--max-workers", str(args.max_workers)]
        if args.n is not None:
            predict_cmd += ["--n", str(args.n)]
        if args.verbose:
            predict_cmd.append("--verbose")
        if args.monitor:
            predict_cmd.append("--monitor")
        if args.ntrials > 1:
            predict_cmd += ["--ntrials", str(args.ntrials)]
        run_step("Predict", predict_cmd)

    # --- Step 2: Calibrate (optional) ---
    if "calibrate" in xid_data:
        calibrate_cmd = [sys.executable, "src/calibrate.py", "--xid", args.xid]
        cv = xid_data.get("calibrate-cv")
        if cv is not None:
            calibrate_cmd += ["--cv", str(cv)]
        run_step("Calibrate", calibrate_cmd)

    # --- Step 3: Eval ---
    eval_cmd = [sys.executable, "src/eval.py", "--xid", args.xid]
    run_step("Evaluate", eval_cmd)

    print("\nDone.")


if __name__ == "__main__":
    main()
