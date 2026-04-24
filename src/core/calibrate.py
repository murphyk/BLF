#!/usr/bin/env python3
"""calibrate.py — Platt scaling, applied to forecasts_final/ in place.

Fits logistic regression on log-odds — `calibrated = sigmoid(a·logit(p) + b)` —
either globally (one model per config) or hierarchically (shared slope,
per-group intercepts). Writes results back into the collated files at the
file level:

    "forecasts_calibrated": {
        "global-cal": [0.58, 0.55, 0.60, ...],   // one value per entry, aligned with forecasts[]
        "hier-cal":   [0.61, 0.57, 0.63, ...]
    }

The base `forecast` field is never overwritten — calibration is an opt-in
eval column (`eval.py --add-calibration`).

For the live-submission pipeline (compete.py), `fit_and_save_model` saves
the fitted model to experiments/calibration_models/{exam}/{config}.json,
and `apply_saved_model` reads that model and populates `forecasts_calibrated`
on unlabeled forecasts.

Usage:
    # Fit with LOO CV on a backtesting exam; writes forecasts_calibrated
    # to every forecasts_final/*/{config}.json that has labels.
    python3 src/core/calibrate.py --xid my-xid --cv loo

    # Also save the fitted models for later application.
    python3 src/core/calibrate.py --xid my-xid --cv loo --save-model tranche-a1

    # Apply a saved model to unlabeled forecasts (e.g. a live submission).
    python3 src/core/calibrate.py --xid live-2026-04-12 --apply-model tranche-a1

    # Hierarchical instead of global.
    python3 src/core/calibrate.py --xid my-xid --cv loo --hierarchical Qsource
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import numpy as np

from config.paths import FORECASTS_FINAL_DIR, CALIBRATION_DIR, XIDS_DIR


# ---------------------------------------------------------------------------
# Platt fitting (global)
# ---------------------------------------------------------------------------

def calibrate_fit(forecasts, outcomes):
    """Fit global Platt scaling. Returns {a, b, n} or None if too few pairs."""
    from sklearn.linear_model import LogisticRegression
    if len(forecasts) < 10:
        print(f"  WARNING: only {len(forecasts)} pairs, too few for calibration")
        return None
    eps = 1e-4
    clipped = np.clip(forecasts, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))
    lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e6)
    lr.fit(logits.reshape(-1, 1), outcomes.astype(int))
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]),
            "n": len(forecasts)}


def calibrate_apply(p, model, group=None):
    """Apply Platt transform; adds a group offset if the model is hierarchical."""
    if p is None or model is None:
        return p
    eps = 1e-4
    pc = max(eps, min(1 - eps, float(p)))
    z = model["a"] * math.log(pc / (1 - pc)) + model["b"]
    if group and "group_offsets" in model:
        z += model["group_offsets"].get(group, 0.0)
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Hierarchical Platt (shared slope, per-group intercepts)
# ---------------------------------------------------------------------------

def hierarchical_fit(forecasts, outcomes, groups, lam=1.0, verbose=False):
    """logit(p_cal) = a·logit(p) + b + δb_g, with L2 penalty λ on δb."""
    from scipy.optimize import minimize
    n = len(forecasts)
    if n < 10:
        return None

    eps = 1e-4
    clipped = np.clip(forecasts, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))
    y = outcomes.astype(float)

    unique_groups = sorted(set(groups))
    g_idx = {g: i for i, g in enumerate(unique_groups)}
    group_ids = np.array([g_idx[g] for g in groups])
    G = len(unique_groups)

    def neg_log_lik(params):
        a, b = params[0], params[1]
        deltas = params[2:]
        z = a * logits + b + deltas[group_ids]
        ll = np.sum(y * z - np.logaddexp(0, z))
        return -ll + lam * np.sum(deltas ** 2)

    def grad(params):
        a, b = params[0], params[1]
        deltas = params[2:]
        z = a * logits + b + deltas[group_ids]
        sig = 1.0 / (1.0 + np.exp(-z))
        resid = y - sig
        g = np.zeros(2 + G)
        g[0] = -np.sum(resid * logits)
        g[1] = -np.sum(resid)
        for gi in range(G):
            mask = group_ids == gi
            g[2 + gi] = -np.sum(resid[mask]) + 2 * lam * deltas[gi]
        return g

    x0 = np.zeros(2 + G)
    x0[0] = 1.0
    res = minimize(neg_log_lik, x0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 1000})

    model = {
        "a": float(res.x[0]),
        "b": float(res.x[1]),
        "group_offsets": {g: float(res.x[2 + i]) for g, i in g_idx.items()},
        "lambda": float(lam),
        "n": int(n),
        "groups": unique_groups,
    }
    if verbose:
        print(f"    Shared: a={model['a']:.4f}, b={model['b']:.4f}")
        for g in unique_groups:
            ng = int(np.sum(group_ids == g_idx[g]))
            print(f"    {g:>15s} (n={ng:3d}): δb={model['group_offsets'][g]:+.4f}")
    return model


# ---------------------------------------------------------------------------
# forecasts_final I/O
# ---------------------------------------------------------------------------

def _load_payload_records(config_name: str,
                          root: str = FORECASTS_FINAL_DIR):
    """Yield (path, payload, entry_idx, entry). Yields every entry across every date-file."""
    for path in sorted(glob.glob(os.path.join(root, "*", f"{config_name}.json"))):
        with open(path) as f:
            payload = json.load(f)
        for i, e in enumerate(payload.get("forecasts", [])):
            yield path, payload, i, e


def _group_for_entry(entry: dict, group_by: str) -> str:
    """Pick the group label for an entry. Default is `source`."""
    if group_by in ("Qsource", "source", None, ""):
        return entry.get("source", "unknown")
    # Other tag spaces would need the data/questions/ lookup — defer for now.
    return entry.get("source", "unknown")


def _gather_records(config_name: str, group_by: str,
                    root: str = FORECASTS_FINAL_DIR,
                    exam: dict[str, list[str]] | None = None):
    """Collect (path, idx, entry, p, outcome, group) tuples for labeled entries.

    If `exam` is given, only records whose (source, base_id) appear in the
    exam are returned — this keeps LOO tractable on large configs by
    restricting the fit set to the xid's exam questions.
    """
    allowed: set[tuple[str, str, str]] | None = None
    if exam is not None:
        from core.eval import _strip_date_suffix
        allowed = set()
        for src, ids in exam.items():
            for qid in ids:
                base, date = _strip_date_suffix(qid)
                allowed.add((src, base, date))

    records = []
    from core.eval import _strip_date_suffix
    for path, payload, idx, entry in _load_payload_records(config_name, root=root):
        p = entry.get("forecast")
        o = entry.get("resolved_to")
        if isinstance(o, list):
            o = o[0] if o else None
        if p is None or o is None:
            continue
        if allowed is not None:
            src = entry.get("source", "")
            base_id = _strip_date_suffix(str(entry.get("id", "")))[0]
            fdd = entry.get("forecast_due_date", "")
            if (src, base_id, fdd) not in allowed:
                continue
        records.append({
            "path": path, "payload": payload, "idx": idx, "entry": entry,
            "p": float(p), "outcome": float(o),
            "group": _group_for_entry(entry, group_by),
        })
    return records


def _write_payloads(paths_seen: dict[str, dict]) -> int:
    """Flush every payload dict we mutated."""
    n = 0
    for p, payload in paths_seen.items():
        with open(p, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        n += 1
    return n


# ---------------------------------------------------------------------------
# CV calibration — writes forecasts_calibrated[key] into forecasts_final
# ---------------------------------------------------------------------------

def _cv_splits(n: int, k):
    from sklearn.model_selection import KFold, LeaveOneOut
    loo = str(k).lower() == "loo"
    return LeaveOneOut() if loo else KFold(n_splits=int(k), shuffle=True, random_state=42)


def _paths_seen(records):
    out: dict[str, dict] = {}
    for r in records:
        out.setdefault(r["path"], r["payload"])
    return out


def _fill_full_column(records, cal_p_of_idx, key: str):
    """Populate each payload's forecasts_calibrated[key] with one value per entry."""
    # Group records by path, then by idx for fast placement.
    by_path: dict[str, dict[int, float]] = defaultdict(dict)
    for rec in records:
        by_path[rec["path"]].setdefault(rec["idx"], cal_p_of_idx.get(rec["idx"]))

    # Walk every payload's entries and write the cal column, defaulting to
    # the raw forecast for entries we couldn't calibrate (e.g. no outcome).
    seen_payloads: dict[str, dict] = {}
    for rec in records:
        seen_payloads.setdefault(rec["path"], rec["payload"])

    for path, payload in seen_payloads.items():
        col = []
        for i, e in enumerate(payload["forecasts"]):
            v = by_path.get(path, {}).get(i)
            if v is None:
                v = e.get("forecast")
            col.append(v)
        cal = payload.setdefault("forecasts_calibrated", {})
        cal[key] = col
    return seen_payloads


def calibrate_config_global(config_name: str, k="loo",
                            key: str = "global-cal",
                            root: str = FORECASTS_FINAL_DIR,
                            save_model_name: str | None = None,
                            exam: dict[str, list[str]] | None = None) -> int:
    """Global Platt scaling with CV, written to forecasts_calibrated[key]."""
    records = _gather_records(config_name, group_by="Qsource", root=root, exam=exam)
    print(f"  {config_name}: {len(records)} labeled entries across "
          f"{len({r['path'] for r in records})} date-files")
    if len(records) < (5 if str(k).lower() == "loo" else 10):
        print(f"  SKIP: too few labeled records for {config_name}")
        return 0

    n = len(records)
    forecasts = np.array([r["p"] for r in records])
    outcomes = np.array([r["outcome"] for r in records])

    splitter = _cv_splits(n, k)
    cal_by_rec: dict[int, float] = {}
    indices = np.arange(n)
    # Deterministic per-config shuffle so results are reproducible across runs.
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    for train_idx, test_idx in splitter.split(indices):
        tr = indices[train_idx]
        te = indices[test_idx]
        model = calibrate_fit(forecasts[tr], outcomes[tr])
        if model is None:
            continue
        for i in te:
            cal_by_rec[int(i)] = calibrate_apply(float(forecasts[i]), model)

    # Map rec-index back to entry-index for placement.
    cal_p_of_idx: dict[int, float] = {}
    for rec_i, rec in enumerate(records):
        if rec_i in cal_by_rec:
            # Multiple records can live in the same file — store by file+entry.
            pass
        # For placement we need (path, idx) so just stash on the rec.
        rec["cal_p"] = cal_by_rec.get(rec_i)

    # Write: for each path, build the full column.
    by_path: dict[str, dict[int, float]] = defaultdict(dict)
    for rec in records:
        if rec["cal_p"] is not None:
            by_path[rec["path"]][rec["idx"]] = rec["cal_p"]

    seen_payloads: dict[str, dict] = {}
    for rec in records:
        seen_payloads.setdefault(rec["path"], rec["payload"])

    for path, payload in seen_payloads.items():
        col = []
        for i, e in enumerate(payload["forecasts"]):
            v = by_path[path].get(i, e.get("forecast"))
            col.append(v)
        cal = payload.setdefault("forecasts_calibrated", {})
        cal[key] = col

    if save_model_name:
        full_model = calibrate_fit(forecasts, outcomes)
        if full_model is not None:
            out_dir = os.path.join(CALIBRATION_DIR, save_model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{config_name}.json")
            with open(out_path, "w") as f:
                json.dump(full_model, f, indent=2)
            print(f"  Saved {config_name} model -> {out_path}")

    n_files = _write_payloads(seen_payloads)
    print(f"  {config_name} → {key}: {n_files} date-files written")
    return n_files


def calibrate_config_hier(config_name: str, k="loo", group_by="Qsource",
                          lam: float = 1.0,
                          key: str = "hier-cal",
                          root: str = FORECASTS_FINAL_DIR,
                          save_model_name: str | None = None,
                          exam: dict[str, list[str]] | None = None) -> int:
    """Hierarchical Platt scaling with CV, written to forecasts_calibrated[key]."""
    records = _gather_records(config_name, group_by=group_by, root=root, exam=exam)
    print(f"  {config_name} (hier, group_by={group_by}, λ={lam}): "
          f"{len(records)} labeled entries")
    if len(records) < (5 if str(k).lower() == "loo" else 10):
        print(f"  SKIP: too few labeled records for {config_name}")
        return 0

    forecasts = np.array([r["p"] for r in records])
    outcomes = np.array([r["outcome"] for r in records])
    groups = np.array([r["group"] for r in records])

    splitter = _cv_splits(len(records), k)
    indices = np.arange(len(records))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    cal_by_rec: dict[int, float] = {}
    for train_idx, test_idx in splitter.split(indices):
        tr = indices[train_idx]
        te = indices[test_idx]
        model = hierarchical_fit(forecasts[tr], outcomes[tr], groups[tr], lam=lam)
        if model is None:
            continue
        for i in te:
            cal_by_rec[int(i)] = calibrate_apply(
                float(forecasts[i]), model, group=str(groups[i]))

    for rec_i, rec in enumerate(records):
        rec["cal_p"] = cal_by_rec.get(rec_i)

    by_path: dict[str, dict[int, float]] = defaultdict(dict)
    for rec in records:
        if rec["cal_p"] is not None:
            by_path[rec["path"]][rec["idx"]] = rec["cal_p"]

    seen_payloads: dict[str, dict] = {}
    for rec in records:
        seen_payloads.setdefault(rec["path"], rec["payload"])

    for path, payload in seen_payloads.items():
        col = []
        for i, e in enumerate(payload["forecasts"]):
            v = by_path[path].get(i, e.get("forecast"))
            col.append(v)
        cal = payload.setdefault("forecasts_calibrated", {})
        cal[key] = col

    if save_model_name:
        full_model = hierarchical_fit(forecasts, outcomes, groups, lam=lam, verbose=True)
        if full_model is not None:
            out_dir = os.path.join(CALIBRATION_DIR, save_model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{config_name}.json")
            with open(out_path, "w") as f:
                json.dump(full_model, f, indent=2)
            print(f"  Saved hier {config_name} model -> {out_path}")

    n_files = _write_payloads(seen_payloads)
    print(f"  {config_name} → {key}: {n_files} date-files written")
    return n_files


# ---------------------------------------------------------------------------
# Model save / apply (for live submission)
# ---------------------------------------------------------------------------

def fit_and_save_model(config_name: str, model_name: str,
                       hierarchical_group: str | None = None,
                       lam: float = 1.0,
                       root: str = FORECASTS_FINAL_DIR) -> dict | None:
    """Fit on ALL labeled records for a config and save the model."""
    records = _gather_records(config_name,
                              group_by=hierarchical_group or "Qsource",
                              root=root)
    if len(records) < 10:
        print(f"  WARNING: {config_name} has only {len(records)} labeled forecasts")
        return None
    forecasts = np.array([r["p"] for r in records])
    outcomes = np.array([r["outcome"] for r in records])
    if hierarchical_group:
        groups = np.array([r["group"] for r in records])
        model = hierarchical_fit(forecasts, outcomes, groups, lam=lam, verbose=True)
    else:
        model = calibrate_fit(forecasts, outcomes)
    if model is None:
        return None
    out_dir = os.path.join(CALIBRATION_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{config_name}.json")
    with open(out_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"  Saved {config_name} -> {out_path}")
    return model


def apply_saved_model(config_name: str, model_name: str,
                      key: str = "global-cal",
                      root: str = FORECASTS_FINAL_DIR) -> int:
    """Load a saved model and write forecasts_calibrated[key] into every
    date-file for config_name. Does NOT require resolved outcomes."""
    model_path = os.path.join(CALIBRATION_DIR, model_name, f"{config_name}.json")
    if not os.path.exists(model_path):
        print(f"  ERROR: calibration model not found at {model_path}")
        return 0
    with open(model_path) as f:
        model = json.load(f)
    is_hier = "group_offsets" in model
    print(f"  Loaded {model_name}/{config_name}.json "
          f"(a={model['a']:.3f}, b={model['b']:.3f}, n={model.get('n')}"
          + (", hierarchical" if is_hier else "") + ")")

    n_files = 0
    for path in sorted(glob.glob(os.path.join(root, "*", f"{config_name}.json"))):
        with open(path) as f:
            payload = json.load(f)
        col = []
        for e in payload.get("forecasts", []):
            g = e.get("source") if is_hier else None
            col.append(calibrate_apply(e.get("forecast"), model, group=g))
        cal = payload.setdefault("forecasts_calibrated", {})
        cal[key] = col
        with open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        n_files += 1
    print(f"  {config_name} → {key}: {n_files} date-files updated")
    return n_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configs_in_xid(xid: str) -> list[str]:
    from config.config import resolve_config, pprint as cfg_pprint
    with open(os.path.join(XIDS_DIR, f"{xid}.json")) as f:
        xid_data = json.load(f)
    labels: list[str] = []
    for field in ("calibrate", "config", "eval"):
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
    ap = argparse.ArgumentParser(
        description="Platt scaling — writes forecasts_calibrated into forecasts_final files.")
    ap.add_argument("--xid", required=True)
    ap.add_argument("--cv", default="loo",
                    help="CV scheme: integer (k-fold) or 'loo' (default)")
    ap.add_argument("--hierarchical", default=None, metavar="LABEL_SPACE",
                    help="Hierarchical calibration with per-group intercepts "
                         "(e.g. Qsource). If omitted, runs global calibration.")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="L2 penalty on group offsets for hierarchical (default 1.0)")
    ap.add_argument("--key", default=None,
                    help="Dict key inside forecasts_calibrated (default: "
                         "'global-cal' or 'hier-cal' depending on --hierarchical)")
    ap.add_argument("--save-model", default=None, metavar="NAME",
                    help="Also fit on all data and save the model under "
                         "experiments/calibration_models/{NAME}/{config}.json")
    ap.add_argument("--apply-model", default=None, metavar="NAME",
                    help="Apply a pre-trained model instead of running CV. "
                         "No labels needed; writes forecasts_calibrated[key].")
    args = ap.parse_args()

    key = args.key or ("hier-cal" if args.hierarchical else "global-cal")
    configs = [c for c in _configs_in_xid(args.xid) if not c.startswith("fb-")]

    # Load the xid's exam so CV fitting is restricted to its questions.
    # This keeps LOO tractable when a config has labels across many dates.
    exam = None
    with open(os.path.join(XIDS_DIR, f"{args.xid}.json")) as f:
        xid_data = json.load(f)
    exam_name = xid_data.get("exam")
    if exam_name:
        exam_path = os.path.join("data", "exams", exam_name, "indices.json")
        if os.path.exists(exam_path):
            with open(exam_path) as f:
                exam = json.load(f)

    if args.apply_model:
        total = 0
        print(f"apply: model='{args.apply_model}', key='{key}'")
        for c in configs:
            total += apply_saved_model(c, args.apply_model, key=key)
        print(f"\napply done: {total} date-files updated")
        return

    total = 0
    print(f"cv-calibrate: cv='{args.cv}', hierarchical={args.hierarchical!r}, "
          f"key='{key}', exam={exam_name or '(all)'}")
    for c in configs:
        if args.hierarchical:
            total += calibrate_config_hier(c, k=args.cv,
                                           group_by=args.hierarchical,
                                           lam=args.lam, key=key,
                                           save_model_name=args.save_model,
                                           exam=exam)
        else:
            total += calibrate_config_global(c, k=args.cv, key=key,
                                             save_model_name=args.save_model,
                                             exam=exam)
    print(f"\ncv-calibrate done: {total} date-files updated")


if __name__ == "__main__":
    main()
