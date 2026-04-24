"""
compete.py — end-to-end ForecastBench competition pipeline.

Usage:
    # Full system with calibration (our best method)
    caffeinate -s python3 src/fb_compete.py --date 2026-04-12 \
        --config "pro/thk:high/crowd:1/tools:1" --ntrials 5 \
        --calibration-model tranche-a1

    # Without calibration
    caffeinate -s python3 src/fb_compete.py --date 2026-04-12 \
        --config "pro/thk:high/crowd:1/tools:1" --ntrials 5

Steps:
    1. fb_make_data.py --date {date} --unresolved
       Fetches the live question set from ForecastBench GitHub.

    2. make_exam.py --name {date}
       Builds experiments/exams/{date}/indices.json.

    3. predict.py --live --exam {date} --config {config} [--ntrials N]
       Runs the agentic forecaster on all questions.
       Results go to experiments/forecasts_raw/{config}/{source}/{id}.json.

    4. Assemble submission JSON
       Expands dataset questions to one entry per resolution_date, strips the
       _{date} suffix from dataset question IDs.
       Writes submissions/{date}.{org}.{N}.json.

    5. Upload to GCS
       gsutil cp submissions/{date}.{org}.{N}.json {bucket}/{filename}
"""

import argparse
import json
import os
import re
import subprocess
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}

DEFAULT_ORG = "sirbayes"
GCP_BUCKET_TEMPLATE = "gs://forecastbench-submissions/{date}/team8"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_step(description, cmd):
    """Run a subprocess, printing the command and checking for errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"ERROR: {description} failed (exit code {result.returncode})")


def load_exam(date):
    """Load exam indices for the given date."""
    path = os.path.join("data", "exams", date, "indices.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: exam indices not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_question_set(date):
    """Load ForecastBench question set for coverage check."""
    path = os.path.join("data", "fb_cache", "question_sets", f"{date}-llm.json")
    if not os.path.exists(path):
        sys.exit(f"ERROR: question set not found: {path}  "
                 f"(run fb_make_data.py --date {date} first)")
    with open(path) as f:
        return json.load(f)


def load_forecasts(config, exam):
    """Load all forecast JSONs for config, keyed by (source, qid)."""
    forecasts = {}
    for source, ids in exam.items():
        for qid in ids:
            safe_id = re.sub(r'[/\\:]', '_', str(qid))
            path = os.path.join("experiments", "forecasts_raw", config, source, f"{safe_id}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                fc = json.load(f)
            if fc.get("forecast") is None:
                continue
            forecasts[(source, qid)] = fc
    return forecasts


def submission_id(source, qid):
    """Return the base question ID for submission.

    Question IDs in v6 have the form "{base_id}_{forecast_due_date}"
    (e.g. "ABBV_2026-03-29" or "0x5c60..._2026-04-12").
    The submission expects just the base ID.
    """
    return re.sub(r'_\d{4}-\d{2}-\d{2}$', '', str(qid))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end ForecastBench competition pipeline."
    )
    parser.add_argument("--date", required=True,
                        help="Forecast due date, e.g. 2026-03-29")
    parser.add_argument("--config", required=True,
                        help="Config name(s), comma-separated for ensemble")
    parser.add_argument("--ensemble", default=None, metavar="NAME",
                        help="Ensemble name: predict with all configs, ensemble them, "
                             "and submit the ensemble. Requires multiple --config values.")
    parser.add_argument("--ensemble-k", type=int, default=5,
                        help="Max ensemble members for greedy selection (default: 5)")
    parser.add_argument("--N", type=int, default=1, choices=[1, 2, 3],
                        help="Submission number for this round (1-3, default 1)")
    parser.add_argument("--org", default=DEFAULT_ORG,
                        help=f"Organization name for submission (default: '{DEFAULT_ORG}')")
    parser.add_argument("--model", default="BLF",
                        help="Model identifier for submission (default: 'BLF', Bayesian Linguistic Forecaster)")
    parser.add_argument("--bucket", default=None,
                        help="GCS bucket path override (default: derived from --date)")
    parser.add_argument("--max-workers", type=int, default=50,
                        help="Max parallel workers for predict.py (default: 50)")
    parser.add_argument("--ntrials", type=int, default=1,
                        help="Number of prediction trials (default: 1)")
    parser.add_argument("--agg-method", default=None,
                        help="Trial aggregation: 'plain-mean' (default) or 'std-shrinkage'")
    parser.add_argument("--calibration-model", default=None, metavar="NAME",
                        help="Apply a pre-fitted calibration model after prediction. "
                             "Loads from experiments/calibration_models/{NAME}/")
    parser.add_argument("--include-reasoning", action="store_true", default=False,
                        help="Include reasoning field in submission (default: omit)")
    parser.add_argument("--skip-predict", action="store_true", default=False,
                        help="Skip prediction step — use existing forecasts "
                             "(useful to re-assemble submission without re-running)")
    parser.add_argument("--notify", action="store_true", default=False,
                        help="Send notification when done.")
    args = parser.parse_args()

    date = args.date
    config_names_raw = [c.strip().removesuffix(".json")
                        for c in args.config.split(",") if c.strip()]
    # Resolve delta strings to directory names
    from config.config import resolve_config, pprint_path
    config_names = []
    for c in config_names_raw:
        if "/" in c and ":" in c:
            cfg = resolve_config(c)
            config_names.append(pprint_path(cfg))
        else:
            config_names.append(c)
    org = args.org
    bucket = args.bucket or GCP_BUCKET_TEMPLATE.format(date=date)

    # The config used for assembly (may change if ensembling or calibrating)
    submit_config = config_names[0] if len(config_names) == 1 else None
    model = args.model or (args.ensemble or config_names[0])

    if args.ensemble and len(config_names) < 2:
        sys.exit("ERROR: --ensemble requires multiple configs (comma-separated)")

    if not args.skip_predict:
        # -------------------------------------------------------------------
        # Step 1: Fetch live questions
        # -------------------------------------------------------------------
        run_step(
            "Fetch live questions from ForecastBench",
            [sys.executable, "src/fb_make_data.py", "--date", date, "--unresolved"],
        )

        # -------------------------------------------------------------------
        # Step 2: Build exam
        # -------------------------------------------------------------------
        exam_dir = os.path.join("data", "exams", date)
        os.makedirs(exam_dir, exist_ok=True)
        mixture_path = os.path.join(exam_dir, "mixture.json")
        if not os.path.exists(mixture_path):
            with open(mixture_path, "w") as f:
                json.dump({"start-date": date, "end-date": date}, f, indent=2)
            print(f"Created {mixture_path}")

        run_step(
            "Build exam indices",
            [sys.executable, "src/make_exam.py", "--name", date],
        )

        # -------------------------------------------------------------------
        # Step 3: Predict (all configs, possibly in parallel)
        # -------------------------------------------------------------------
        predict_cmd = [sys.executable, "src/predict.py",
                       "--live",
                       "--exam", date,
                       "--config", ",".join(config_names),
                       "--max-workers", str(args.max_workers)]
        if args.ntrials > 1:
            predict_cmd += ["--ntrials", str(args.ntrials)]
        if args.agg_method:
            predict_cmd += ["--agg-method", args.agg_method]
        run_step("Generate predictions", predict_cmd)
    else:
        print("\n[Skipping steps 1-3 (--skip-predict)]")

    # -----------------------------------------------------------------------
    # Step 3b: Ensemble (optional, when multiple configs provided)
    # -----------------------------------------------------------------------
    if args.ensemble:
        print(f"\n{'='*60}")
        print(f"Step: Ensemble '{args.ensemble}' from {config_names}")
        print(f"{'='*60}")
        from core.ensemble import load_method_forecasts, greedy_select, write_ensemble
        _exam = load_exam(date)
        method_forecasts = {}
        for cname in config_names:
            fc = load_method_forecasts(cname, _exam)
            if fc:
                method_forecasts[cname] = fc
                print(f"  Loaded {cname}: {len(fc)} forecasts")
            else:
                print(f"  WARNING: {cname} has no forecasts")

        if len(method_forecasts) >= 2:
            print(f"\n  Greedy forward selection (k≤{args.ensemble_k}):")
            selected = greedy_select(method_forecasts, args.ensemble_k)
            if selected:
                print(f"  Selected: {selected}")
                write_ensemble(args.ensemble, selected, method_forecasts)
                submit_config = args.ensemble
            else:
                print("  WARNING: greedy selection failed, using first config")
                submit_config = config_names[0]
        elif len(method_forecasts) == 1:
            print("  WARNING: only 1 config has forecasts, skipping ensemble")
            submit_config = config_names[0]
        else:
            sys.exit("ERROR: no configs have forecasts for ensemble")
    elif len(config_names) == 1:
        submit_config = config_names[0]
    else:
        # Multiple configs but no --ensemble: use first one
        print(f"  WARNING: multiple configs but no --ensemble, using {config_names[0]}")
        submit_config = config_names[0]

    # -----------------------------------------------------------------------
    # Step 3c: Calibrate (optional, using pre-trained model)
    # -----------------------------------------------------------------------
    if args.calibration_model:
        print(f"\n{'='*60}")
        print(f"Step: Apply calibration model '{args.calibration_model}'")
        print(f"{'='*60}")
        from core.calibrate import apply_saved_model
        _exam = load_exam(date)
        n_cal = apply_saved_model(submit_config, _exam, args.calibration_model)
        if n_cal:
            submit_config = f"{submit_config}_calibrated"
            print(f"  Using calibrated config: {submit_config}")
        else:
            print("  WARNING: calibration failed, using uncalibrated forecasts")

    config = submit_config

    # -----------------------------------------------------------------------
    # Step 4: Assemble submission
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step: Assemble submission file")
    print(f"{'='*60}")

    exam = load_exam(date)
    qs_data = load_question_set(date)
    question_set = qs_data["question_set"]
    n_questions = len(qs_data["questions"])

    # Build lookup: (source, base_id) -> question metadata (for resolution_dates)
    qs_lookup = {(q["source"], str(q["id"])): q for q in qs_data["questions"]}

    forecasts_dict = load_forecasts(config, exam)
    print(f"Loaded {len(forecasts_dict)} forecasts")

    # Assemble submission entries.
    # Dataset questions: one entry per resolution_date (same forecast for each).
    # Market questions: single entry with the question's resolution_date.
    forecasts = []
    for (source, qid), fc in sorted(forecasts_dict.items()):
        p = fc.get("forecast")
        base_id = submission_id(source, qid)
        qs_q = qs_lookup.get((source, base_id), {})
        rdates = qs_q.get("resolution_dates")

        if source in DATASET_SOURCES and isinstance(rdates, list) and rdates:
            # Use per-resolution-date forecasts if available
            per_date_ps = fc.get("forecasts", [])
            for i, rdate in enumerate(rdates):
                p_i = per_date_ps[i] if i < len(per_date_ps) else p
                entry = {"id": base_id, "source": source,
                         "forecast": p_i, "resolution_date": rdate}
                if args.include_reasoning:
                    entry["reasoning"] = fc.get("reasoning", "")
                forecasts.append(entry)
        else:
            rdate = fc.get("resolution_date") or qs_q.get("resolution_date", "N/A")
            entry = {"id": base_id, "source": source,
                     "forecast": p, "resolution_date": rdate}
            if args.include_reasoning:
                entry["reasoning"] = fc.get("reasoning", "")
            forecasts.append(entry)

    n_covered = len({(e["source"], e["id"]) for e in forecasts})
    coverage = n_covered / n_questions if n_questions else 0
    print(f"Coverage: {n_covered}/{n_questions} questions ({coverage:.1%})")
    if coverage < 0.95:
        print(f"WARNING: coverage {coverage:.1%} is below the required 95% minimum.")

    forecast_set = {
        "organization": org,
        "model": model,
        "model_organization": org,
        "question_set": question_set,
        "forecasts": forecasts,
    }

    os.makedirs("submissions", exist_ok=True)
    out_path = os.path.join("submissions", f"{date}.{org}.{args.N}.json")
    with open(out_path, "w") as f:
        json.dump(forecast_set, f, indent=2)
    print(f"Saved -> {out_path}")

    # -----------------------------------------------------------------------
    # Step 5: Upload to GCS
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step: Upload to GCS")
    print(f"{'='*60}")

    filename = os.path.basename(out_path)
    gcs_dest = f"{bucket}/{filename}"
    print(f"Uploading to {gcs_dest} ...")

    sdk_bin = os.path.expanduser("~/Kevin/google-cloud-sdk/bin")
    env = os.environ.copy()
    env["PATH"] = sdk_bin + os.pathsep + env.get("PATH", "")
    try:
        result = subprocess.run(
            ["gsutil", "cp", out_path, gcs_dest],
            capture_output=True, text=True, env=env,
        )
        if result.returncode == 0:
            print("Upload succeeded.")
        else:
            print(f"Upload failed (exit code {result.returncode}).")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
    except FileNotFoundError:
        print("Upload failed: 'gsutil' not found. "
              "Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install")

    if args.notify:
        from notify import send_notification
        send_notification(
            f"compete.py done: {date} {config}",
            f"Coverage: {coverage:.1%} ({n_covered}/{n_questions})\n"
            f"Submission: {out_path}",
        )
