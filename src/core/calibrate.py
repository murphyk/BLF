#!/usr/bin/env python3
"""calibrate.py — Platt scaling calibration for forecasts.

Fits logistic regression on log-odds: calibrated = sigmoid(a * logit(p) + b).
Uses K-fold cross-validation for honest calibrated forecasts.

Usage:
    python3 src/calibrate.py --xid myxid
    python3 src/calibrate.py --xid myxid --cv 5
    python3 src/calibrate.py --xid myxid --cv loo

The xid must have "exam" and either "calibrate" or "config" fields.
"calibrate" is a list of config names to calibrate (may include ensemble names).
If "calibrate" is missing, falls back to "config".
Optional "calibrate-cv" field in xid (defaults to 3).

Reads from results/forecasts/{config}/{source}/{id}.json
Writes to results/forecasts/{config}_calibrated/{source}/{id}.json
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
# Platt scaling core
# ---------------------------------------------------------------------------

def calibrate_fit(forecasts, outcomes):
    """Fit Platt scaling model. Returns {a, b, n} or None if too few pairs."""
    from sklearn.linear_model import LogisticRegression

    if len(forecasts) < 10:
        print(f"  WARNING: only {len(forecasts)} pairs, too few for calibration")
        return None

    eps = 1e-4
    clipped = np.clip(forecasts, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))

    lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e6)
    lr.fit(logits.reshape(-1, 1), outcomes.astype(int))

    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return {"a": a, "b": b, "n": len(forecasts)}


def calibrate_apply(p, model, group=None):
    """Apply Platt scaling to a single forecast value.

    If model has 'group_offsets' and group is provided, adds the
    group-specific intercept offset.
    """
    if p is None or model is None:
        return p
    eps = 1e-4
    p_clipped = max(eps, min(1 - eps, p))
    logit_p = np.log(p_clipped / (1 - p_clipped))
    z = model["a"] * logit_p + model["b"]
    if group and "group_offsets" in model:
        z += model["group_offsets"].get(group, 0.0)
    return float(1 / (1 + np.exp(-z)))


# ---------------------------------------------------------------------------
# Hierarchical Platt scaling (shared slope, group-specific intercept offsets)
# ---------------------------------------------------------------------------

def hierarchical_fit(forecasts, outcomes, groups, lam=1.0, verbose=False):
    """Fit hierarchical Platt scaling: logit(p_cal) = a*logit(p) + b + δb_g.

    Parameters:
        forecasts: array of raw forecast probabilities
        outcomes: array of binary outcomes
        groups: array of group labels (one per forecast)
        lam: L2 penalty strength on group offsets δb_g
        verbose: if True, print fitted parameters

    Returns dict {a, b, group_offsets: {g: δb_g}, lambda, n, groups: [g,...]}
    or None if too few data.
    """
    from scipy.optimize import minimize

    n = len(forecasts)
    if n < 10:
        return None

    eps = 1e-4
    clipped = np.clip(forecasts, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))
    y = outcomes.astype(float)

    unique_groups = sorted(set(groups))
    G = len(unique_groups)
    g_idx = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([g_idx[g] for g in groups])

    # Parameter vector: [a, b, δb_0, δb_1, ..., δb_{G-1}]
    def neg_log_lik(params):
        a, b = params[0], params[1]
        deltas = params[2:]
        z = a * logits + b + deltas[group_ids]
        # Numerically stable log-loss
        ll = np.sum(y * z - np.logaddexp(0, z))
        penalty = lam * np.sum(deltas ** 2)
        return -ll + penalty

    def grad(params):
        a, b = params[0], params[1]
        deltas = params[2:]
        z = a * logits + b + deltas[group_ids]
        sig = 1.0 / (1.0 + np.exp(-z))
        resid = y - sig  # n-vector
        g = np.zeros(2 + G)
        g[0] = -np.sum(resid * logits)        # d/da
        g[1] = -np.sum(resid)                  # d/db
        for gi in range(G):
            mask = group_ids == gi
            g[2 + gi] = -np.sum(resid[mask]) + 2 * lam * deltas[gi]
        return g

    x0 = np.zeros(2 + G)
    x0[0] = 1.0  # a ~ 1 (identity mapping)
    res = minimize(neg_log_lik, x0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 1000})

    a = float(res.x[0])
    b = float(res.x[1])
    offsets = {g: float(res.x[2 + i]) for g, i in g_idx.items()}

    if verbose:
        print(f"    Shared: a={a:.4f}, b={b:.4f}")
        for g in unique_groups:
            ng = int(np.sum(group_ids == g_idx[g]))
            print(f"    {g:>15s} (n={ng:3d}): δb={offsets[g]:+.4f}")

    return {"a": a, "b": b, "group_offsets": offsets,
            "lambda": lam, "n": n, "groups": unique_groups}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def load_xid(xid: str) -> dict:
    path = f"experiments/xids/{xid}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"XID not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_exam(exam_name: str) -> dict[str, list[str]]:
    path = os.path.join("data", "exams", exam_name, "indices.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Exam not found: {path}")
    with open(path) as f:
        return json.load(f)


def forecast_path(config_name: str, source: str, qid: str) -> str:
    safe_id = re.sub(r'[/\\:]', '_', str(qid))
    return os.path.join("experiments", "forecasts", config_name, source, f"{safe_id}.json")


# ---------------------------------------------------------------------------
# Loading helpers (multi-resolution-date aware)
# ---------------------------------------------------------------------------

def _load_records(config_name, exam, load_group=False, group_by="Qsource"):
    """Load forecast records, expanding multi-resolution-date questions.

    For single-resdate questions: one record with (forecast, outcome).
    For multi-resdate questions: one record per (forecast_i, outcome_i),
    plus resdate_index so we can write back the calibrated per-resdate values.

    Returns list of dicts with keys: source, qid, forecast, outcome, fc,
    resdate_index (int or None), and optionally 'group'.
    """
    if load_group:
        from config.tags import get_tag

    records = []
    for source, ids in exam.items():
        for qid in ids:
            path = forecast_path(config_name, source, qid)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                fc = json.load(f)

            outcome = fc.get("resolved_to")
            if outcome is None:
                continue

            # Load question for group tag if needed
            group = source
            if load_group:
                q_path = os.path.join("data", "questions", source,
                                      re.sub(r'[/\\:]', '_', str(qid)) + ".json")
                if os.path.exists(q_path):
                    with open(q_path) as f:
                        q = json.load(f)
                    group = get_tag(q, group_by) or source

            if isinstance(outcome, list):
                # Multi-resdate: expand to one record per resdate
                forecasts_list = fc.get("forecasts", [])
                for i, o in enumerate(outcome):
                    if o is None:
                        continue
                    p = forecasts_list[i] if i < len(forecasts_list) else fc.get("forecast")
                    if p is None:
                        continue
                    rec = {"source": source, "qid": qid,
                           "forecast": p, "outcome": float(o),
                           "fc": fc, "resdate_index": i}
                    if load_group:
                        rec["group"] = group
                    records.append(rec)
            else:
                p = fc.get("forecast")
                if p is None:
                    continue
                rec = {"source": source, "qid": qid,
                       "forecast": p, "outcome": float(outcome),
                       "fc": fc, "resdate_index": None}
                if load_group:
                    rec["group"] = group
                records.append(rec)
    return records


def _write_calibrated(records, calibrated, out_config, cal_fn):
    """Write calibrated forecasts, applying cal_fn to per-resdate forecasts too.

    calibrated: dict mapping record index -> calibrated probability.
    cal_fn: function(raw_p) -> calibrated_p (the learned calibration transform).
    """
    # Group records by (source, qid) to handle multi-resdate
    from collections import defaultdict
    by_question = defaultdict(list)  # (source, qid) -> [(index, rec, cal_p)]
    for idx, cal_p in calibrated.items():
        rec = records[idx]
        by_question[(rec["source"], rec["qid"])].append((idx, rec, cal_p))

    n_written = 0
    for (source, qid), entries in by_question.items():
        # Use fc from first entry as base
        fc = dict(entries[0][1]["fc"])
        fc["calibrated"] = True
        if "group" in entries[0][1]:
            fc["calibration_group"] = entries[0][1]["group"]

        if entries[0][1]["resdate_index"] is None:
            # Single resdate
            fc["forecast_raw"] = fc["forecast"]
            fc["forecast"] = entries[0][2]
        else:
            # Multi-resdate: calibrate each per-resdate forecast
            raw_forecasts = fc.get("forecasts", [])
            fc["forecasts_raw"] = list(raw_forecasts)
            cal_forecasts = []
            for i, p in enumerate(raw_forecasts):
                cal_forecasts.append(float(cal_fn(p)))
            fc["forecasts"] = cal_forecasts
            # Also calibrate the top-level forecast
            fc["forecast_raw"] = fc["forecast"]
            fc["forecast"] = float(cal_fn(fc["forecast"]))

        out_path = forecast_path(out_config, source, qid)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f_out:
            json.dump(fc, f_out, indent=2)
        n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# CV calibration over exam questions
# ---------------------------------------------------------------------------

def calibrate_config(config_name: str, exam: dict[str, list[str]],
                     k: int | str = 3,
                     suffix: str = "calibrated") -> int:
    """Run K-fold CV Platt scaling on a config's forecasts over an exam.

    k="loo" means leave-one-out CV (works with small n, e.g. smoke tests).
    Multi-resolution-date questions are expanded: each (forecast_i, outcome_i)
    pair is a separate training/test point.

    Returns number of calibrated forecast files written.
    """
    from sklearn.model_selection import KFold, LeaveOneOut

    out_config = f"{config_name}_{suffix}"
    loo = (str(k).lower() == "loo")
    cv_label = "LOO" if loo else f"k={k}"
    print(f"  Calibrating {config_name} -> {out_config} ({cv_label})")

    records = _load_records(config_name, exam)
    print(f"    Loaded {len(records)} records from {len(exam)} sources")

    min_n = 5 if loo else 10
    if len(records) < min_n:
        print(f"  WARNING: only {len(records)} records, too few for "
              f"{'LOO' if loo else 'CV'} calibration (need >= {min_n})")
        return 0

    # Shuffle and split
    rng = np.random.default_rng(42)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    splitter = LeaveOneOut() if loo else KFold(n_splits=k, shuffle=False)
    calibrated = {}  # index -> calibrated_p
    last_model = None

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(indices)):
        train_forecasts = np.array([records[indices[i]]["forecast"] for i in train_idx])
        train_outcomes = np.array([records[indices[i]]["outcome"] for i in train_idx])

        model = calibrate_fit(train_forecasts, train_outcomes)
        if model is None:
            continue
        last_model = model

        for i in test_idx:
            rec = records[indices[i]]
            calibrated[indices[i]] = calibrate_apply(rec["forecast"], model)

    # Build cal_fn from last model for applying to per-resdate forecasts
    if last_model is not None:
        # Refit on all data for the write transform
        all_f = np.array([r["forecast"] for r in records])
        all_o = np.array([r["outcome"] for r in records])
        full_model = calibrate_fit(all_f, all_o)
        cal_fn = lambda p: calibrate_apply(p, full_model)
    else:
        cal_fn = lambda p: p

    n_written = _write_calibrated(records, calibrated, out_config, cal_fn)
    print(f"  Calibrated {n_written} forecasts -> experiments/forecasts/{out_config}/")
    return n_written


def calibrate_config_hierarchical(config_name: str, exam: dict[str, list[str]],
                                  k: int | str = "loo",
                                  group_by: str = "Qsource",
                                  lam: float = 1.0,
                                  suffix: str = "calibrated") -> int:
    """Run CV hierarchical Platt scaling (shared slope, group-specific intercepts).

    group_by: label space for grouping (e.g. "Qsource", "Qtype").
    lam: L2 penalty on group intercept offsets.

    Returns number of calibrated forecasts written.
    """
    from sklearn.model_selection import KFold, LeaveOneOut

    out_config = f"{config_name}_{suffix}"
    loo = (str(k).lower() == "loo")
    cv_label = "LOO" if loo else f"k={k}"
    print(f"  Hierarchical calibrating {config_name} -> {out_config} "
          f"({cv_label}, group_by={group_by}, λ={lam})")

    records = _load_records(config_name, exam, load_group=True, group_by=group_by)
    print(f"    Loaded {len(records)} records from {len(exam)} sources")

    min_n = 5 if loo else 10
    if len(records) < min_n:
        print(f"  WARNING: only {len(records)} records, too few for calibration")
        return 0

    rng = np.random.default_rng(42)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    all_forecasts = np.array([records[i]["forecast"] for i in indices])
    all_outcomes = np.array([records[i]["outcome"] for i in indices])
    all_groups = np.array([records[i]["group"] for i in indices])

    splitter = LeaveOneOut() if loo else KFold(n_splits=k, shuffle=False)
    calibrated = {}

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(indices)):
        model = hierarchical_fit(
            all_forecasts[train_idx], all_outcomes[train_idx],
            all_groups[train_idx], lam=lam, verbose=False)
        if model is None:
            continue

        for i in test_idx:
            rec = records[indices[i]]
            calibrated[indices[i]] = calibrate_apply(
                rec["forecast"], model, group=rec["group"])

    # Fit on all data for summary display and for cal_fn
    print(f"  Summary (fit on all {len(records)} records):")
    full_model = hierarchical_fit(all_forecasts, all_outcomes, all_groups,
                                  lam=lam, verbose=True)

    # Build cal_fn that uses source-based group lookup
    def cal_fn(p, source=None):
        return calibrate_apply(p, full_model, group=source)

    # For _write_calibrated, we need a cal_fn that takes only p
    # and infers source from the record. Build per-source cal_fns.
    def make_source_cal_fn(source):
        return lambda p: calibrate_apply(p, full_model, group=source)

    # Write: group by question, apply per-source calibration
    from collections import defaultdict
    by_question = defaultdict(list)
    for idx, cal_p in calibrated.items():
        rec = records[idx]
        by_question[(rec["source"], rec["qid"])].append((idx, rec, cal_p))

    n_written = 0
    for (source, qid), entries in by_question.items():
        fc = dict(entries[0][1]["fc"])
        fc["calibrated"] = True
        fc["calibration_group"] = entries[0][1].get("group", source)
        src_cal = make_source_cal_fn(entries[0][1].get("group", source))

        if entries[0][1]["resdate_index"] is None:
            fc["forecast_raw"] = fc["forecast"]
            fc["forecast"] = entries[0][2]
        else:
            raw_forecasts = fc.get("forecasts", [])
            fc["forecasts_raw"] = list(raw_forecasts)
            fc["forecasts"] = [float(src_cal(p)) for p in raw_forecasts]
            fc["forecast_raw"] = fc["forecast"]
            fc["forecast"] = float(src_cal(fc["forecast"]))

        out_path = forecast_path(out_config, source, qid)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f_out:
            json.dump(fc, f_out, indent=2)
        n_written += 1

    print(f"  Calibrated {n_written} forecasts -> experiments/forecasts/{out_config}/")
    return n_written


# ---------------------------------------------------------------------------
# Fit and save / load and apply calibration models
# ---------------------------------------------------------------------------

_MODEL_DIR = os.path.join("experiments", "calibration_models")


def fit_and_save_model(config_name: str, exam: dict[str, list[str]],
                       model_name: str) -> dict | None:
    """Fit Platt scaling on ALL data (no CV) and save model params.

    Reads from results/forecasts/{config_name}/{source}/{id}.json.
    Saves to experiments/calibration_models/{model_name}/{config_name}.json.
    """
    records = []
    for source, ids in exam.items():
        for qid in ids:
            path = forecast_path(config_name, source, qid)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                fc = json.load(f)
            p = fc.get("forecast")
            outcome = fc.get("resolved_to")
            if isinstance(outcome, list):
                outcome = outcome[0] if outcome else None
            if p is not None and outcome is not None:
                records.append({"forecast": p, "outcome": float(outcome)})

    if len(records) < 10:
        print(f"  WARNING: {config_name} has only {len(records)} forecasts, "
              f"too few to fit calibration model")
        return None

    forecasts = np.array([r["forecast"] for r in records])
    outcomes = np.array([r["outcome"] for r in records])
    model = calibrate_fit(forecasts, outcomes)
    if model is None:
        return None

    # Save
    out_dir = os.path.join(_MODEL_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{config_name}.json")
    with open(out_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"  Saved calibration model -> {out_path}")
    return model


def apply_saved_model(config_name: str, exam: dict[str, list[str]],
                      model_name: str, suffix: str = "calibrated") -> int:
    """Apply a previously saved calibration model to forecasts.

    Reads model from experiments/calibration_models/{model_name}/{config_name}.json.
    Reads forecasts from results/forecasts/{config_name}/{source}/{id}.json.
    Writes to results/forecasts/{config_name}_calibrated/{source}/{id}.json.

    Does NOT require resolved_to (works on unlabeled test data).
    """
    model_path = os.path.join(_MODEL_DIR, model_name, f"{config_name}.json")
    if not os.path.exists(model_path):
        print(f"  ERROR: calibration model not found: {model_path}")
        return 0

    with open(model_path) as f:
        model = json.load(f)
    print(f"  Loaded calibration model from {model_path} "
          f"(a={model['a']:.4f}, b={model['b']:.4f}, n={model['n']})")

    out_config = f"{config_name}_{suffix}"
    n_written = 0
    for source, ids in exam.items():
        for qid in ids:
            path = forecast_path(config_name, source, qid)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                fc = json.load(f)
            p = fc.get("forecast")
            if p is None:
                continue

            cal_p = calibrate_apply(p, model)
            fc_out = dict(fc)
            fc_out["forecast_raw"] = p
            fc_out["forecast"] = cal_p
            fc_out["calibrated"] = True
            fc_out["calibration_model"] = model_name

            out_path = forecast_path(out_config, source, qid)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(fc_out, f, indent=2)
            n_written += 1

    print(f"  Applied model to {n_written} forecasts -> "
          f"experiments/forecasts/{out_config}/")
    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Platt scaling calibration (CV, fit+save, or apply)")
    parser.add_argument("--xid", required=True,
                        help="Experiment ID (must have 'exam' and 'calibrate' or 'config')")
    parser.add_argument("--cv", default=None,
                        help="CV folds: integer or 'loo' for leave-one-out "
                             "(overrides xid's calibrate-cv field)")
    parser.add_argument("--save-model", default=None, metavar="NAME",
                        help="Fit on all data and save model to "
                             "experiments/calibration_models/{NAME}/")
    parser.add_argument("--apply-model", default=None, metavar="NAME",
                        help="Apply a saved model (no labels needed). "
                             "Loads from experiments/calibration_models/{NAME}/")
    parser.add_argument("--hierarchical", default=None, metavar="LABEL_SPACE",
                        help="Use hierarchical calibration with group-specific "
                             "intercepts (e.g. --hierarchical Qsource)")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="L2 penalty on group offsets for hierarchical "
                             "calibration (default: 1.0)")
    parser.add_argument("--suffix", default=None,
                        help="Output suffix (default: 'calibrated_global' or "
                             "'calibrated_hier' based on --hierarchical)")
    args = parser.parse_args()

    # Auto-determine suffix if not specified
    if args.suffix is None:
        if args.hierarchical:
            args.suffix = "calibrated_hier"
        else:
            args.suffix = "calibrated_global"

    xid_data = load_xid(args.xid)

    if "exam" not in xid_data:
        sys.exit("ERROR: xid must have an 'exam' field")

    exam_name = xid_data["exam"]
    configs = xid_data.get("calibrate", xid_data.get("config", []))
    if isinstance(configs, str):
        configs = [configs]

    # Resolve delta strings to filesystem names
    from config.config import resolve_config, pprint_path
    resolved = []
    for c in configs:
        if "_calibrated" in c:
            continue
        if "/" in c and ":" in c:
            cfg = resolve_config(c)
            resolved.append(pprint_path(cfg))
        else:
            resolved.append(c)
    configs = resolved

    exam = load_exam(exam_name)

    if args.apply_model:
        # Apply a pre-trained model (no labels needed)
        print(f"Applying calibration model '{args.apply_model}'")
        print(f"  Exam: {exam_name}")
        print(f"  Configs: {configs}")
        total = 0
        for config_name in configs:
            n = apply_saved_model(config_name, exam, args.apply_model,
                                     suffix=args.suffix)
            total += n
        print(f"\nDone: {total} calibrated forecasts")

    else:
        # CV calibration (optionally also save the full-data model)
        cv_raw = args.cv or xid_data.get("calibrate-cv", 3)
        if str(cv_raw).lower() == "loo":
            k = "loo"
        else:
            k = int(cv_raw)

        print(f"CV Calibration (k={k})")
        print(f"  Exam: {exam_name}")
        print(f"  Configs: {configs}")
        total = 0
        for config_name in configs:
            if args.hierarchical:
                n = calibrate_config_hierarchical(
                    config_name, exam, k=k,
                    group_by=args.hierarchical, lam=args.lam,
                    suffix=args.suffix)
            else:
                n = calibrate_config(config_name, exam, k=k,
                                     suffix=args.suffix)
            total += n

        # If --save-model, also fit on all data and save
        if args.save_model:
            print(f"\nFitting full-data models -> '{args.save_model}'")
            for config_name in configs:
                fit_and_save_model(config_name, exam, args.save_model)

        print(f"\nDone: {total} total CV-calibrated forecasts")
        if args.save_model:
            print(f"  Models saved. Apply with: "
                  f"python3 src/calibrate.py --xid ... --apply-model {args.save_model}")


if __name__ == "__main__":
    main()
