#!/usr/bin/env python3
"""ensemble.py — Greedy forward ensemble selection and averaging.

Selects up to K methods that minimize ensemble Brier score on an exam
(stops early if adding the next member would increase the score),
then writes averaged forecasts for each question.

Usage:
    python3 src/ensemble.py --exam market-train --out my-ensemble
    python3 src/ensemble.py --exam market-train --candidates m1,m2,m3 --k 5 --out my-ensemble

Arguments:
    --exam          Exam name (reads experiments/exams/{name}/indices.json)
    --out           Output config name written to results/forecasts/{out}/
    --candidates    Comma-separated config names to consider (default: auto-discover
                    all configs that have forecasts for the exam)
    --k             Max ensemble members (default: 5)
    --calibrate     Also Platt-scale the ensemble via CV (writes {out}_calibrated/)
    --cv            CV folds for calibration (default: 5)

Output:
    results/forecasts/{out}/{source}/{id}.json           Averaged forecasts
    results/forecasts/{out}_calibrated/{source}/{id}.json (if --calibrate)
    experiments/ensembles/{out}.json                      Ensemble definition
                                                          (selected members, exam, n)

Note: ensemble.py reads the aggregated (post-shrinkage) forecasts from each
candidate config. If candidates were run with --ntrials, the per-trial averaging
has already been done by predict.py. Cross-model ensembling then averages these
already-aggregated values. For a joint treatment of cross-trial and cross-model
variance, see aggregate.py --method llm-agg.
"""

import argparse
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def load_exam(exam_name: str) -> dict[str, list[str]]:
    """Load exam indices: {source: [id1, id2, ...]}."""
    path = os.path.join("data", "exams", exam_name, "indices.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Exam not found: {path}")
    with open(path) as f:
        return json.load(f)


def forecast_path(config_name: str, source: str, qid: str) -> str:
    """Build forecast path: results/forecasts/{config}/{source}/{id}.json."""
    safe_id = re.sub(r'[/\\:]', '_', str(qid))
    return os.path.join("experiments", "forecasts", config_name, source, f"{safe_id}.json")


# ---------------------------------------------------------------------------
# Load forecasts
# ---------------------------------------------------------------------------

def load_method_forecasts(config_name: str,
                          exam: dict[str, list[str]]) -> dict[tuple, dict]:
    """Load forecasts for a config across all exam questions.

    Returns {(source, qid): forecast_dict} for questions with both
    a forecast and a resolved_to value.
    """
    results = {}
    for source, ids in exam.items():
        for qid in ids:
            path = forecast_path(config_name, source, qid)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                fc = json.load(f)
            if fc.get("forecast") is not None and fc.get("resolved_to") is not None:
                results[(source, qid)] = fc
    return results


def discover_candidates(exam: dict[str, list[str]]) -> list[str]:
    """Auto-discover all config names that have any forecasts for the exam."""
    forecasts_dir = os.path.join("experiments", "forecasts")
    if not os.path.isdir(forecasts_dir):
        return []
    candidates = []
    for config_name in sorted(os.listdir(forecasts_dir)):
        config_dir = os.path.join(forecasts_dir, config_name)
        if not os.path.isdir(config_dir):
            continue
        # Check if any exam question has a forecast here
        found = False
        for source, ids in exam.items():
            for qid in ids:
                if os.path.exists(forecast_path(config_name, source, qid)):
                    found = True
                    break
            if found:
                break
        if found:
            candidates.append(config_name)
    return candidates


# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------

def greedy_select(method_forecasts: dict[str, dict[tuple, dict]],
                  k: int) -> list[str]:
    """Greedy forward selection: pick up to K methods minimizing ensemble Brier.

    Stops early if adding the next best member would increase the score.
    Returns list of selected method names in selection order.
    """
    # Find common keys (questions all methods have forecasts for)
    all_keys = None
    for mname, forecasts in method_forecasts.items():
        keys = set(forecasts.keys())
        all_keys = keys if all_keys is None else all_keys & keys

    if not all_keys:
        print("  WARNING: no common questions across methods")
        return []

    keys = sorted(all_keys)
    method_names = list(method_forecasts.keys())

    preds = {}
    for mname in method_names:
        preds[mname] = np.array([method_forecasts[mname][k]["forecast"] for k in keys])
    outcomes = np.array([method_forecasts[method_names[0]][k]["resolved_to"] for k in keys])

    def ensemble_brier(selected_names):
        avg = np.mean([preds[m] for m in selected_names], axis=0)
        return float(np.mean((avg - outcomes) ** 2))

    selected = []
    remaining = set(method_names)
    prev_score = float("inf")

    for step in range(min(k, len(method_names))):
        best_name = None
        best_score = float("inf")
        for candidate in remaining:
            trial = selected + [candidate]
            score = ensemble_brier(trial)
            if score < best_score:
                best_score = score
                best_name = candidate

        if best_name is None:
            break

        if selected and best_score >= prev_score:
            print(f"  Step {step + 1}: stopping early — best addition "
                  f"({best_name}) would not improve score "
                  f"({best_score:.4f} >= {prev_score:.4f})")
            break

        selected.append(best_name)
        remaining.remove(best_name)
        prev_score = best_score
        print(f"  Step {step + 1}: +{best_name} → ensemble Brier={best_score:.4f}")

    return selected


# ---------------------------------------------------------------------------
# Write ensemble forecasts
# ---------------------------------------------------------------------------

def write_ensemble(ensemble_name: str,
                   members: list[str],
                   method_forecasts: dict[str, dict[tuple, dict]]) -> int:
    """Average forecasts from selected members, write to results/forecasts/{ensemble_name}/.

    Returns number of questions written.
    """
    # Find common keys across selected members
    common_keys = None
    for mname in members:
        keys = set(method_forecasts[mname].keys())
        common_keys = keys if common_keys is None else common_keys & keys
    common_keys = sorted(common_keys or [])

    if not common_keys:
        print("  WARNING: no common questions across selected members")
        return 0

    n_written = 0
    for source, qid in common_keys:
        ps = [method_forecasts[m][(source, qid)]["forecast"] for m in members]
        avg_p = float(np.mean(ps))

        base_fc = method_forecasts[members[0]][(source, qid)]
        ensemble_fc = {
            "id": base_fc.get("id", "?"),
            "source": base_fc.get("source", source),
            "question": base_fc.get("question", ""),
            "background": base_fc.get("background", ""),
            "resolution_criteria": base_fc.get("resolution_criteria", ""),
            "forecast_due_date": base_fc.get("forecast_due_date", ""),
            "forecast": avg_p,
            "resolution_date": base_fc.get("resolution_date", ""),
            "resolved_to": base_fc.get("resolved_to"),
            "ensemble_name": ensemble_name,
            "ensemble_members": members,
            "member_forecasts": {m: method_forecasts[m][(source, qid)]["forecast"]
                                 for m in members},
        }

        out_path = forecast_path(ensemble_name, source, qid)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(ensemble_fc, f, indent=2)
        n_written += 1

    print(f"  Wrote {n_written} ensemble forecasts → results/forecasts/{ensemble_name}/")
    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Greedy ensemble selection and averaging")
    parser.add_argument("--exam", required=True,
                        help="Exam name (experiments/exams/{name}/indices.json)")
    parser.add_argument("--out", required=True,
                        help="Output config name (results/forecasts/{out}/)")
    parser.add_argument("--candidates", default=None,
                        help="Comma-separated config names to consider "
                             "(default: auto-discover from results/forecasts/)")
    parser.add_argument("--k", type=int, default=5,
                        help="Max ensemble members (default: 5, stops early if score worsens)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Also run Platt scaling CV on the ensemble forecasts")
    parser.add_argument("--cv", default="5",
                        help="CV folds for calibration: integer or 'loo' (default: 5)")
    args = parser.parse_args()

    exam = load_exam(args.exam)

    # Resolve candidate list
    if args.candidates:
        candidate_names = [c.strip() for c in args.candidates.split(",") if c.strip()]
    else:
        candidate_names = discover_candidates(exam)
        if not candidate_names:
            sys.exit("ERROR: no configs found in results/forecasts/ — "
                     "run predict.py first or specify --candidates")

    print(f"Greedy ensemble (k≤{args.k})")
    print(f"  Exam:       {args.exam}")
    print(f"  Output:     {args.out}")
    print(f"  Candidates: {candidate_names}")

    # Load forecasts
    method_forecasts = {}
    for mname in candidate_names:
        fc = load_method_forecasts(mname, exam)
        if fc:
            method_forecasts[mname] = fc
            print(f"  Loaded {mname}: {len(fc)} forecasts")
        else:
            print(f"  Skipping {mname}: no resolved forecasts for this exam")

    if len(method_forecasts) < 2:
        sys.exit("ERROR: need at least 2 methods with resolved forecasts for ensemble")

    # Greedy selection
    print("\nGreedy forward selection:")
    selected = greedy_select(method_forecasts, args.k)

    if not selected:
        sys.exit("ERROR: greedy selection found no members")

    print(f"\nSelected ({len(selected)}): {selected}")

    # Write ensemble
    n = write_ensemble(args.out, selected, method_forecasts)

    # Save ensemble definition for reproducibility
    ensembles_dir = os.path.join("experiments", "ensembles")
    os.makedirs(ensembles_dir, exist_ok=True)
    meta = {
        "exam": args.exam,
        "config": selected,
        "k": len(selected),
        "n_questions": n,
    }
    meta_path = os.path.join(ensembles_dir, f"{args.out}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved ensemble definition → {meta_path}")

    # Optional calibration
    if args.calibrate:
        from core.calibrate import calibrate_config
        cv_k = args.cv if args.cv.lower() == "loo" else int(args.cv)
        cv_label = "LOO" if args.cv.lower() == "loo" else f"{cv_k}-fold"
        print(f"\nCalibrating ensemble ({cv_label} CV):")
        n_cal = calibrate_config(args.out, exam, k=cv_k)
        print(f"  {n_cal} calibrated forecasts → results/forecasts/{args.out}_calibrated/")

    print(f"\nDone: {n} ensemble forecasts written")
    print(f"  Run eval with this config name: {args.out}")
    if args.calibrate:
        print(f"  Calibrated config name: {args.out}_calibrated")


if __name__ == "__main__":
    main()
