#!/usr/bin/env python3
"""
fb_compete.py — End-to-end ForecastBench competition pipeline.

Pipeline (all operating on one {date}):

    1. fb_make_data.py --date {date} --unresolved
       Fetch the live question set from the ForecastBench GitHub release.

    2. make_exam.py --name {date}
       Build experiments/exams/{date}/indices.json.

    3. predict.py --live --exam {date} --config {config} [--ntrials N]
       Run the agentic forecaster. Raw output: experiments/forecasts_raw/
       {config}/{source}/{id}.json, with per-trial sub-dirs if ntrials > 1.

    4. collate.py --config {config}
       Merge raw per-question files into experiments/forecasts_final/{date}/
       {config}.json (FB-tarball shape). The `forecast` field on each entry
       is the default trial aggregation (mean of trials).

    5. (optional) calibrate.py --apply-model {name}
       Load a pre-fitted Platt model from experiments/calibration_models/
       {name}/{config}.json and populate each entry's
       forecasts_calibrated["global-cal"] in the collated file.

    6. Assemble submission: read forecasts_final/{date}/{config}.json,
       choose the forecast column (calibrated if available, raw
       otherwise), strip metadata, write submissions/{date}.{org}.{N}.json.

    7. Upload to GCS via gsutil.

Usage:
    # Full system with calibration
    caffeinate -s python3 src/compete/fb_compete.py --date 2026-04-12 \\
        --config "pro/thk:high/crowd:1/tools:1" --ntrials 5 \\
        --calibration-model tranche-a1

    # Without calibration
    caffeinate -s python3 src/compete/fb_compete.py --date 2026-04-12 \\
        --config "pro/thk:high/crowd:1/tools:1" --ntrials 5

    # Re-assemble submission from existing collated files (no re-run)
    python3 src/compete/fb_compete.py --date 2026-04-12 \\
        --config pro-high-brave-c1-t1 --skip-predict --calibration-model tranche-a1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

from config.paths import FORECASTS_FINAL_DIR

DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}
DEFAULT_ORG = "sirbayes"
GCP_BUCKET_TEMPLATE = "gs://forecastbench-submissions/{date}/team8"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_step(description: str, cmd: list) -> None:
    print(f"\n{'='*60}\nStep: {description}\n  {' '.join(map(str, cmd))}\n{'='*60}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"ERROR: {description} failed (exit code {r.returncode})")


def load_exam(date: str) -> dict:
    path = os.path.join("data", "exams", date, "indices.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: exam indices not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_question_set(date: str) -> dict:
    path = os.path.join("data", "fb_cache", "question_sets", f"{date}-llm.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: question set not found: {path}  "
                 f"(run fb_make_data.py --date {date} first)")
    with open(path) as f:
        return json.load(f)


def load_collated(date: str, config: str) -> dict:
    path = os.path.join(FORECASTS_FINAL_DIR, date, f"{config}.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: collated file not found: {path}  "
                 f"(run collate.py --config {config} first)")
    with open(path) as f:
        return json.load(f)


_DATE_SUFFIX_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})$")

def strip_date_suffix(qid: str) -> str:
    """Internal qids carry a _{forecast_due_date} suffix; FB wants the base."""
    return _DATE_SUFFIX_RE.sub("", str(qid))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end FB competition pipeline.")
    ap.add_argument("--date", required=True, help="Forecast due date, e.g. 2026-04-12")
    ap.add_argument("--config", default=None,
                    help="Config (directory name or delta string). Comma-separated "
                         "for ensembling. Mutually exclusive with --config-file.")
    ap.add_argument("--config-file", default=None, metavar="PATH",
                    help="Load a full AgentConfig from JSON (e.g. "
                         "experiments/configs/sota.json). Takes its 'name' "
                         "field as the submit config.")
    ap.add_argument("--ensemble", default=None, metavar="NAME",
                    help="Ensemble name; predict every --config, pick members greedily.")
    ap.add_argument("--ensemble-k", type=int, default=5)
    ap.add_argument("--N", type=int, default=1, choices=[1, 2, 3],
                    help="Submission number (1-3)")
    ap.add_argument("--org", default=DEFAULT_ORG)
    ap.add_argument("--model", default="BLF",
                    help="Model identifier (default: BLF)")
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--max-workers", type=int, default=50)
    ap.add_argument("--ntrials", type=int, default=1)
    ap.add_argument("--agg-method", default=None,
                    help="Trial aggregation: 'plain-mean' (default) or 'std-shrinkage'")
    ap.add_argument("--calibration-model", default=None, metavar="NAME",
                    help="Apply experiments/calibration_models/{NAME}/{config}.json "
                         "before submitting.")
    ap.add_argument("--cal-key", default="global-cal",
                    help="Key under forecasts_calibrated to read back (default: global-cal)")
    ap.add_argument("--include-reasoning", action="store_true", default=False)
    ap.add_argument("--skip-predict", action="store_true", default=False,
                    help="Skip fetch/exam/predict/collate — reuse existing collated file")
    ap.add_argument("--skip-upload", action="store_true", default=False)
    ap.add_argument("--notify", action="store_true", default=False)
    args = ap.parse_args()

    date = args.date
    bucket = args.bucket or GCP_BUCKET_TEMPLATE.format(date=date)

    if args.config and args.config_file:
        sys.exit("ERROR: pass only one of --config or --config-file.")
    if not (args.config or args.config_file):
        sys.exit("ERROR: one of --config or --config-file is required.")

    from config.config import resolve_config, pprint as cfg_pprint
    if args.config_file:
        with open(args.config_file) as f:
            cfg_dict = json.load(f)
        if not cfg_dict.get("name"):
            sys.exit(f"ERROR: config file {args.config_file} must have a 'name' field.")
        config_names = [cfg_dict["name"]]
    else:
        raw_configs = [c.strip().removesuffix(".json")
                       for c in args.config.split(",") if c.strip()]
        config_names = []
        for c in raw_configs:
            if "/" in c and ":" in c:
                config_names.append(cfg_pprint(resolve_config(c)))
            else:
                config_names.append(c)

    if args.ensemble and len(config_names) < 2:
        sys.exit("ERROR: --ensemble requires multiple --config values.")

    # -----------------------------------------------------------------------
    # Steps 1-4: fetch, exam, predict, collate (unless --skip-predict)
    # -----------------------------------------------------------------------
    if not args.skip_predict:
        run_step("Fetch live questions",
                 [sys.executable, "src/data/fb_make_data.py",
                  "--date", date, "--unresolved"])

        exam_dir = os.path.join("data", "exams", date)
        os.makedirs(exam_dir, exist_ok=True)
        mixture_path = os.path.join(exam_dir, "mixture.json")
        if not os.path.exists(mixture_path):
            with open(mixture_path, "w") as f:
                json.dump({"ask-start": date, "ask-end": date}, f, indent=2)
        run_step("Build exam indices",
                 [sys.executable, "src/data/make_exam.py", "--name", date])

        predict_cmd = [sys.executable, "src/core/predict.py",
                       "--live", "--exam", date,
                       "--max-workers", str(args.max_workers)]
        if args.config_file:
            predict_cmd += ["--config-file", args.config_file]
        else:
            predict_cmd += ["--config", ",".join(config_names)]
        if args.ntrials > 1:
            predict_cmd += ["--ntrials", str(args.ntrials)]
        if args.agg_method:
            predict_cmd += ["--agg-method", args.agg_method]
        run_step("Predict", predict_cmd)

        for cname in config_names:
            run_step(f"Collate {cname}",
                     [sys.executable, "src/core/collate.py", "--config", cname])
    else:
        print("[Skipping steps 1-4: --skip-predict]")

    # -----------------------------------------------------------------------
    # Step 4b: ensemble (optional)
    # -----------------------------------------------------------------------
    submit_config = config_names[0]
    if args.ensemble and len(config_names) > 1:
        from core.ensemble import load_method_forecasts, greedy_select, write_ensemble
        exam = load_exam(date)
        method_forecasts = {}
        for c in config_names:
            fc = load_method_forecasts(c, exam)
            if fc:
                method_forecasts[c] = fc
        if len(method_forecasts) >= 2:
            selected = greedy_select(method_forecasts, args.ensemble_k)
            if selected:
                write_ensemble(args.ensemble, selected, method_forecasts)
                submit_config = args.ensemble
                run_step(f"Collate ensemble {args.ensemble}",
                         [sys.executable, "src/core/collate.py",
                          "--config", args.ensemble])
        else:
            print("  WARNING: ensemble skipped (fewer than 2 configs with forecasts)")

    # -----------------------------------------------------------------------
    # Step 5: apply pre-fitted calibration model
    # -----------------------------------------------------------------------
    cal_applied = False
    if args.calibration_model:
        from core.calibrate import apply_saved_model
        print(f"\n{'='*60}\nStep: Apply calibration '{args.calibration_model}' "
              f"to {submit_config}\n{'='*60}")
        n = apply_saved_model(submit_config, args.calibration_model, key=args.cal_key)
        cal_applied = n > 0

    # -----------------------------------------------------------------------
    # Step 6: assemble submission
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nStep: Assemble submission\n{'='*60}")
    payload = load_collated(date, submit_config)
    qs_data = load_question_set(date)
    question_set = qs_data["question_set"]
    n_questions = len(qs_data["questions"])

    cal_col = None
    if cal_applied:
        cal_col = (payload.get("forecasts_calibrated") or {}).get(args.cal_key)
        if cal_col is None:
            print(f"  WARNING: forecasts_calibrated[{args.cal_key}] missing; "
                  f"submitting raw forecasts.")

    entries = payload.get("forecasts", [])
    out_forecasts = []
    seen_ids: set[tuple[str, str]] = set()
    for i, e in enumerate(entries):
        source = e.get("source", "")
        base_id = strip_date_suffix(e.get("id", ""))
        p = cal_col[i] if cal_col is not None else e.get("forecast")
        if p is None:
            continue
        rdate = str(e.get("resolution_date", ""))
        row = {"id": base_id, "source": source, "forecast": float(p),
               "resolution_date": rdate}
        if args.include_reasoning and e.get("reasoning"):
            row["reasoning"] = e["reasoning"]
        out_forecasts.append(row)
        seen_ids.add((source, base_id))

    coverage = len(seen_ids) / n_questions if n_questions else 0
    print(f"Coverage: {len(seen_ids)}/{n_questions} questions ({coverage:.1%})")
    if coverage < 0.95:
        print(f"WARNING: coverage {coverage:.1%} is below the required 95% minimum.")

    model = args.model or (args.ensemble or submit_config)
    submission = {
        "organization": args.org,
        "model": model,
        "model_organization": args.org,
        "question_set": question_set,
        "forecasts": out_forecasts,
    }

    os.makedirs("submissions", exist_ok=True)
    out_path = os.path.join("submissions", f"{date}.{args.org}.{args.N}.json")
    with open(out_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Saved → {out_path}")

    # -----------------------------------------------------------------------
    # Step 7: upload to GCS
    # -----------------------------------------------------------------------
    if args.skip_upload:
        print("[Skipping upload: --skip-upload]")
    else:
        print(f"\n{'='*60}\nStep: Upload to GCS\n{'='*60}")
        gcs_dest = f"{bucket}/{os.path.basename(out_path)}"
        print(f"Uploading to {gcs_dest} ...")
        # Try a few known SDK install locations; PATH already includes the
        # active one for most users so this is just belt-and-suspenders.
        env = os.environ.copy()
        for cand in ["~/google-cloud-sdk/bin",
                     "~/Kevin/google-cloud-sdk/bin",
                     "/usr/local/google-cloud-sdk/bin"]:
            cand_p = os.path.expanduser(cand)
            if os.path.isfile(os.path.join(cand_p, "gsutil")):
                env["PATH"] = cand_p + os.pathsep + env.get("PATH", "")
                break
        try:
            r = subprocess.run(["gsutil", "cp", out_path, gcs_dest],
                               capture_output=True, text=True, env=env)
            if r.returncode == 0:
                print("Upload succeeded.")
            else:
                print(f"Upload failed (exit {r.returncode}).")
                print(r.stdout or "")
                print(r.stderr or "")
        except FileNotFoundError:
            print("Upload failed: gsutil not found. "
                  "Install the Google Cloud SDK first.")

    if args.notify:
        try:
            from notify import send_notification
            send_notification(
                f"compete.py done: {date} {submit_config}",
                f"Coverage: {coverage:.1%}\nSubmission: {out_path}",
            )
        except ImportError:
            pass


if __name__ == "__main__":
    main()
