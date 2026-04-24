"""
adjusted_brier_score.py — Difficulty-adjusted Brier score, approximating ForecastBench.
Based on https://www.forecastbench.org/assets/pdfs/forecastbench_updated_methodology.pdf
and https://github.com/forecastingresearch/forecastbench.

Dataset question difficulty
---------------------------
Primary path (use_precomputed_fe=True, default):
    Downloads ForecastBench's official question fixed effects from:
        https://www.forecastbench.org/assets/data/question-fixed-effects/
    These are computed nightly using ~269 LLM + Naive Forecaster submissions via
    pyfixest feols("brier_score ~ 1 | question_pk + model_pk").  Files are cached
    locally in datasets/question_fixed_effects/.

Fallback path (use_precomputed_fe=False, or when download fails):
    Estimates question fixed effects from the provided methods via alternating
    projections (Gauss-Seidel), equivalent to pyfixest without external dependencies.
    This uses only the methods passed to compute_adjusted_brier_table, so estimates
    are noisier when the method pool is small.  Optional synthetic always_0 / always_1
    baselines are added when real methods are below synthetic_baseline_threshold to
    stabilise estimates.  ForecastBench excludes these because its pool is always ≥ 269.

Market question difficulty
--------------------------
Always uses the community Brier score per question:
    community_brier_q = (freeze_datetime_value_q - resolved_to_q)^2
This approximates ForecastBench's Imputed Forecaster (crowd forecast at t-1).

Adjusted score (paper notation)
--------------------------------
The paper defines the per-(model, question) adjusted Brier score as:

    b_adj(i, j) = b(i, j) − gamma_j

i.e., the raw Brier score minus the question's fixed effect.  The reported score for
model i is the average over all questions it answered:

    score_i = mean_j [ b(i,j) − gamma_j ]

Rescaling so that Always-0.5 → 0.25
-------------------------------------
ForecastBench shifts every score by a constant so that an Always-0.5 forecaster (which
predicts 0.5 on every question) yields exactly 0.25 (the chance-level Brier score).

For Always-0.5 the raw Brier on question j is (0.5 − resolved_to_j)^2, so its
unadjusted score is:

    score_{always0.5} = mean_j [ (0.5 − resolved_to_j)^2 − gamma_j ]
                      = 0.25 − mean_j(gamma_j)           [for binary resolutions]

To shift this to exactly 0.25 we add  mean_j(gamma_j)  to every model's score:

    reported_i = mean_j [ b(i,j) − gamma_j ] + mean_j(gamma_j)
               = mean_j [ b(i,j) − gamma_j + global_mean_fe ]

where  global_mean_fe = mean_j(gamma_j)  averaged over ALL questions in the pool.

Normalisation-constant invariance (fallback path only)
------------------------------------------------------
The alternating-projections algorithm converges to a solution (alpha, gamma) that
satisfies the OLS normal equations.  The solution is not unique — any constant c can
be added to all alpha_i and subtracted from all gamma_j without changing the fit.
However, the term  global_mean_fe = mean_j(gamma_j)  absorbs this constant: adding c
to all gamma_j shifts global_mean_fe by c, and the shift cancels in the rescaled score
because b(i,j) − gamma_j + mean_j(gamma_j) is invariant to this additive constant.
Therefore the final reported scores are identical regardless of normalisation, matching
pyfixest (which applies a different normalisation convention internally).

Overall score
-------------
Unweighted average of the dataset and market adjusted scores, matching ForecastBench.
"""

import datetime
import json
import logging
import math
import os

import numpy as np
import pandas as pd

try:
    import requests as _requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)

MARKET_SOURCES = {"infer", "manifold", "metaculus", "polymarket"}
DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}

_FE_BASE_URL = "https://www.forecastbench.org/assets/data/question-fixed-effects"
_FE_DEFAULT_CACHE_DIR = "datasets/question_fixed_effects"

# Synthetic baselines added to the dataset FE pool (fallback path only) when real methods
# are below synthetic_baseline_threshold.
# always_0 and always_1 both vary by question (brier = resolved_to and 1-resolved_to respectively)
# so they supply identifying variation, unlike always_0.5 which gives the same Brier score (0.25)
# on every question and provides no discriminating power.
# They act as anchors that reduce variance in the gamma_j estimates when the method pool is thin,
# at the cost of pulling question-difficulty estimates slightly toward base-rate effects.
_SYNTHETIC_BASELINES = {"__synthetic_always_0__": 0.0, "__synthetic_always_1__": 1.0}


# ---------------------------------------------------------------------------
# Precomputed FE helpers
# ---------------------------------------------------------------------------


def _load_or_download_fe_file(date_str: str, cache_dir: str):
    """Load question fixed effects for date_str from cache or download from ForecastBench.

    Args:
        date_str (str): Date in YYYY-MM-DD format.
        cache_dir (str): Local directory for caching downloaded files.

    Returns:
        list of dicts, or None if unavailable.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"question_fixed_effects.{date_str}.json")

    if os.path.exists(local_path):
        with open(local_path) as f:
            return json.load(f)

    if not _HAS_REQUESTS:
        return None

    url = f"{_FE_BASE_URL}/question_fixed_effects.{date_str}.json"
    try:
        resp = _requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        records = resp.json()
    except Exception:
        return None

    with open(local_path, "w") as f:
        json.dump(records, f)
    return records


def _fetch_latest_precomputed_fe(cache_dir: str, max_lookback_days: int = 14):
    """Return a DataFrame of precomputed question fixed effects, trying recent dates.

    Tries today and up to max_lookback_days days back until a file is found.

    Args:
        cache_dir (str): Local directory for caching.
        max_lookback_days (int): How many days back to look for a valid file.

    Returns:
        pd.DataFrame with columns [date, source, id, horizon, question_fixed_effect]
        where date is a YYYY-MM-DD string and horizon is an int (NaN for market questions).
        Returns None if no file could be fetched.
    """
    today = datetime.date.today()
    for delta in range(max_lookback_days + 1):
        date_str = (today - datetime.timedelta(days=delta)).strftime("%Y-%m-%d")
        records = _load_or_download_fe_file(date_str, cache_dir)
        if records is None:
            continue

        rows = []
        for r in records:
            horizon = r.get("horizon")
            rows.append(
                {
                    "date": pd.to_datetime(r["forecast_due_date"], unit="ms").strftime("%Y-%m-%d"),
                    "source": r["source"],
                    "id": str(r["id"]),
                    "horizon": int(horizon) if horizon is not None else pd.NA,
                    "question_fixed_effect": r["question_fixed_effect"],
                }
            )
        fe_df = pd.DataFrame(rows)
        fe_df["horizon"] = fe_df["horizon"].astype("Int64")
        logger.info(f"Loaded precomputed FE from {date_str} ({len(fe_df)} records).")
        return fe_df

    return None


# ---------------------------------------------------------------------------
# Fallback: alternating-projections FE estimator
# ---------------------------------------------------------------------------


def _estimate_question_fe(long_df, tol=1e-8, max_iter=500):
    """
    Estimate question fixed effects via alternating projections (Gauss-Seidel).

    OLS objective
    -------------
    Minimise  L(a, b) = sum_{i,j} (brier_{ij} - a_i - b_j)^2
    over model effects a = {alpha_i} and question effects b = {gamma_j}.

    The first-order (normal) equations are coupled:
        gamma_j = mean_i (brier_{ij} − alpha_i)    for each question j
        alpha_i = mean_j (brier_{ij} − gamma_j)    for each model i

    Alternating projections (block coordinate descent / Gauss-Seidel) solves
    these by updating one set of effects at a time while holding the other fixed:

        gamma_j ← mean_i (brier_{ij} − alpha_i)
        alpha_i ← mean_j (brier_{ij} − gamma_j)

    Each step exactly minimises L over the updated block given the other block,
    so L decreases (or stays the same) monotonically.  Because L is strongly
    convex jointly in (a, b) (when both sets of effects are identified, i.e.
    ≥ 2 methods), the iteration converges to the unique OLS solution up to an
    additive constant (see "Normalisation-constant invariance" in the module
    docstring).  This is mathematically equivalent to pyfixest's
    feols("brier ~ 1 | question + model").

    Args:
        long_df:  DataFrame with columns 'question_pk', 'method', 'brier'.
        tol:      Convergence tolerance on question effects.
        max_iter: Maximum iterations.

    Returns:
        dict {question_pk: question_fixed_effect}, or None if < 2 distinct methods.
    """
    df = long_df[["question_pk", "method", "brier"]].dropna(subset=["brier"]).copy()
    if df["method"].nunique() < 2:
        return None

    questions = df["question_pk"].unique()
    models = df["method"].unique()
    q_idx = pd.Series(np.arange(len(questions)), index=questions)
    m_idx = pd.Series(np.arange(len(models)), index=models)

    qi = df["question_pk"].map(q_idx).values.astype(int)
    mi = df["method"].map(m_idx).values.astype(int)
    brier = df["brier"].values.astype(float)

    n_q = len(questions)
    n_m = len(models)
    q_count = np.bincount(qi, minlength=n_q).astype(float)
    m_count = np.bincount(mi, minlength=n_m).astype(float)

    a = np.zeros(n_m)
    b = np.zeros(n_q)

    for _ in range(max_iter):
        b_prev = b.copy()

        # b_j = mean_i(brier_ij − a_i)
        resid_q = brier - a[mi]
        b = np.where(
            q_count > 0,
            np.bincount(qi, weights=resid_q, minlength=n_q) / np.maximum(q_count, 1),
            0.0,
        )

        # a_i = mean_j(brier_ij − b_j)
        resid_m = brier - b[qi]
        a = np.where(
            m_count > 0,
            np.bincount(mi, weights=resid_m, minlength=n_m) / np.maximum(m_count, 1),
            0.0,
        )

        if np.max(np.abs(b - b_prev)) < tol:
            break

    return dict(zip(questions, b))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _bootstrap_ci(values, n_bootstrap=2000, rng=None):
    """Return (mean, ci_low, ci_high, n) for an array of per-question adjusted scores."""
    rng = rng or np.random.default_rng(0)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(values))
    boot = rng.choice(values, size=(n_bootstrap, n), replace=True).mean(axis=1)
    return mean, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), n


def compute_adjusted_brier_table(
    df,
    methods,
    n_bootstrap=2000,
    use_precomputed_fe=True,
    fe_cache_dir=_FE_DEFAULT_CACHE_DIR,
    include_synthetic_baselines=True,
    synthetic_baseline_threshold=5,
):
    """
    Compute difficulty-adjusted Brier scores for each method.

    Args:
        df:                          Combined DataFrame produced by eval.py, containing columns:
                                       source, id, resolution_date, resolved_to,
                                       freeze_datetime_value, date (optional), and
                                       forecast_{method} for each method in methods.
        methods:                     List of method name strings (must match forecast_{m} columns).
        n_bootstrap:                 Bootstrap replicates for 95% CI.
        use_precomputed_fe:          When True (default), download and use ForecastBench's official
                                       question fixed effects for dataset questions.  Falls back to
                                       alternating-projections estimation if the download fails or
                                       the df has no 'date' column.
        fe_cache_dir:                Local directory for caching downloaded FE files.
        include_synthetic_baselines: Fallback path only.  When True (default), add synthetic
                                       always_0 and always_1 to the FE pool when the number of
                                       real methods is below synthetic_baseline_threshold.
        synthetic_baseline_threshold: Fallback path only.  Threshold below which synthetic
                                       baselines are added (default 5).

    Returns:
        List of dicts, one per method, each containing:
          adjusted_{dataset,market,overall}_brier
          adjusted_{dataset,market,overall}_ci_low
          adjusted_{dataset,market,overall}_ci_high
          adjusted_{dataset,market,overall}_n
          dataset_fe_source         ("precomputed", "estimated", or "unavailable")
          dataset_fe_n_methods      (real methods used; 0 when precomputed)
          dataset_fe_synthetic_used (True if synthetic baselines were added; False when precomputed)
    """
    rng = np.random.default_rng(0)
    df = df.copy()

    # Unique question key: date + source + id + resolution_date
    date_col = (
        df["date"].astype(str) if "date" in df.columns else pd.Series("", index=df.index)
    )
    df["_qpk"] = (
        date_col
        + "|"
        + df["source"].astype(str)
        + "|"
        + df["id"].astype(str)
        + "|"
        + df["resolution_date"].astype(str)
    )

    mkt_mask = df["source"].isin(MARKET_SOURCES)
    ds_mask = df["source"].isin(DATASET_SOURCES)

    # ------------------------------------------------------------------ market
    # question FE = (community_forecast − resolved_to)^2
    df_mkt = df[mkt_mask].copy()
    community = pd.to_numeric(df_mkt["freeze_datetime_value"], errors="coerce")
    df_mkt["_fe"] = (community - df_mkt["resolved_to"]) ** 2

    mkt_fe_valid = df_mkt["_fe"].dropna()
    # global_mean_fe = mean_j(gamma_j): the rescaling shift that makes Always-0.5 → 0.25.
    # Derivation: Always-0.5 unadjusted score = mean_j(0.25 − gamma_j) = 0.25 − mean(gamma_j),
    # so the shift needed to reach 0.25 is exactly mean(gamma_j).  See module docstring.
    global_mean_fe_mkt = float(mkt_fe_valid.mean()) if len(mkt_fe_valid) > 0 else 0.0

    # ----------------------------------------------------------------- dataset
    df_ds = df[ds_mask].copy()
    global_mean_fe_ds = 0.0
    n_methods_fe = 0
    synthetic_used = False
    dataset_fe_source = "unavailable"

    # --- Primary path: precomputed FE from forecastbench.org ---
    if use_precomputed_fe and "date" in df.columns and len(df_ds) > 0:
        fe_df = _fetch_latest_precomputed_fe(fe_cache_dir)
        if fe_df is not None:
            # Keep only dataset sources and rows with a valid horizon for the merge
            fe_ds = fe_df[
                fe_df["source"].isin(DATASET_SOURCES) & fe_df["horizon"].notna()
            ].copy()

            # Compute horizon (days from forecast_due_date to resolution_date) for each row
            df_ds["_horizon"] = (
                pd.to_datetime(df_ds["resolution_date"].astype(str))
                - pd.to_datetime(df_ds["date"].astype(str))
            ).dt.days.astype("Int64")

            # Merge on (date, source, id, horizon) to get official gamma_j per question
            df_ds = df_ds.merge(
                fe_ds[["date", "source", "id", "horizon", "question_fixed_effect"]],
                left_on=["date", "source", "id", "_horizon"],
                right_on=["date", "source", "id", "horizon"],
                how="left",
            ).drop(columns=["horizon"])
            df_ds = df_ds.rename(columns={"question_fixed_effect": "_fe"})

            n_matched = int(df_ds["_fe"].notna().sum())
            n_total = len(df_ds)
            logger.info(f"Precomputed FE: matched {n_matched}/{n_total} dataset rows.")

            if n_matched > 0:
                dataset_fe_source = "precomputed"
                # Average FE per question first, then take the global mean
                _fe_valid = df_ds.loc[df_ds["_fe"].notna(), ["id", "source", "_fe"]]
                global_mean_fe_ds = float(_fe_valid.groupby(["id", "source"])["_fe"].mean().mean())

    # --- Fallback path: estimate FE from provided methods via alternating projections ---
    if dataset_fe_source != "precomputed":
        long_rows = []
        for method in methods:
            col = f"forecast_{method}"
            if col not in df_ds.columns:
                continue
            valid = df_ds[col].notna() & df_ds["resolved_to"].notna() & df_ds[col].between(0, 1)
            if valid.sum() == 0:
                continue
            sub = df_ds.loc[valid, ["_qpk", "resolved_to", col]].copy()
            sub["brier"] = (sub[col] - sub["resolved_to"]) ** 2
            sub["method"] = method
            long_rows.append(
                sub[["_qpk", "method", "brier"]].rename(columns={"_qpk": "question_pk"})
            )

        if long_rows:
            long_ds = pd.concat(long_rows, ignore_index=True)
            n_methods_fe = long_ds["method"].nunique()

            # When real methods are below synthetic_baseline_threshold, add always_0 and always_1
            # to stabilise the FE estimates.  Below 2 methods the model is unidentified; between 2
            # and the threshold the estimates are noisy.  The synthetics reduce variance at the cost
            # of a slight pull toward base-rate difficulty (see _SYNTHETIC_BASELINES comment above).
            if n_methods_fe < synthetic_baseline_threshold and include_synthetic_baselines:
                ds_resolved = df_ds[df_ds["resolved_to"].notna()][["_qpk", "resolved_to"]].copy()
                for name, p in _SYNTHETIC_BASELINES.items():
                    syn_df = ds_resolved.copy()
                    syn_df["brier"] = (p - syn_df["resolved_to"]) ** 2
                    syn_df["method"] = name
                    long_rows.append(
                        syn_df[["_qpk", "method", "brier"]].rename(
                            columns={"_qpk": "question_pk"}
                        )
                    )
                long_ds = pd.concat(long_rows, ignore_index=True)
                synthetic_used = True

            q_fe_ds = _estimate_question_fe(long_ds)
            if q_fe_ds is not None:
                dataset_fe_source = "estimated"
                df_ds["_fe"] = df_ds["_qpk"].map(q_fe_ds)
                # Average FE per question first, then take the global mean
                _fe_valid = df_ds.loc[df_ds["_fe"].notna(), ["id", "source", "_fe"]]
                global_mean_fe_ds = float(_fe_valid.groupby(["id", "source"])["_fe"].mean().mean())
            else:
                df_ds["_fe"] = np.nan
        else:
            df_ds["_fe"] = np.nan

    # ----------------------------------------------- adjusted scores per method
    rows = []
    for method in methods:
        col = f"forecast_{method}"
        row = {
            "method": method,
            "dataset_fe_source": dataset_fe_source,
            "dataset_fe_n_methods": n_methods_fe,
            "dataset_fe_synthetic_used": synthetic_used,
        }

        for label, sub, global_mean_fe in (
            ("market", df_mkt, global_mean_fe_mkt),
            ("dataset", df_ds, global_mean_fe_ds),
        ):

            def _nan_result(lbl=label):
                for sfx in ("brier", "ci_low", "ci_high"):
                    row[f"adjusted_{lbl}_{sfx}"] = float("nan")
                row[f"adjusted_{lbl}_n"] = 0

            if col not in sub.columns:
                _nan_result()
                continue

            has_fe = sub["_fe"].notna() & sub["resolved_to"].notna()
            has_valid_fc = sub[col].notna() & sub[col].between(0, 1)
            # Impute 0.5 for missing/invalid forecasts on resolved questions with FE
            fc_imp = sub[col].copy()
            fc_imp.loc[has_fe & ~has_valid_fc] = 0.5
            valid = has_fe  # all resolved questions with FE
            if valid.sum() == 0:
                _nan_result()
                continue

            brier_q = (fc_imp.loc[valid] - sub.loc[valid, "resolved_to"]) ** 2
            # Paper: b_adj(i,j) = b(i,j) − gamma_j.
            # Rescaled adds global_mean_fe = mean_j(gamma_j) so that Always-0.5 scores 0.25:
            #   reported = mean_j [ b(i,j) − gamma_j + mean_j(gamma_j) ]
            # The global_mean_fe term is invariant to the FE normalisation constant (see
            # module docstring), so the final score matches pyfixest regardless of how the
            # alternating-projections solver distributes the constant between alpha and gamma.
            adj_frame = sub.loc[valid, ["id", "source"]].copy()
            adj_frame["rescaled"] = (brier_q - sub.loc[valid, "_fe"] + global_mean_fe).to_numpy()

            # Aggregate to question level: average adjusted BS across resolution dates
            # within each (id, source).  This gives each question equal weight regardless
            # of how many resolution dates it has (relevant for dataset questions).
            rescaled = adj_frame.groupby(["id", "source"])["rescaled"].mean().to_numpy()

            mean_adj, lo, hi, n = _bootstrap_ci(rescaled, n_bootstrap=n_bootstrap, rng=rng)
            row[f"adjusted_{label}_brier"] = mean_adj
            row[f"adjusted_{label}_ci_low"] = lo
            row[f"adjusted_{label}_ci_high"] = hi
            row[f"adjusted_{label}_n"] = n

        # overall = unweighted average of dataset and market (ForecastBench convention)
        d = row["adjusted_dataset_brier"]
        m = row["adjusted_market_brier"]
        if not (math.isnan(d) or math.isnan(m)):
            row["adjusted_overall_brier"] = (d + m) / 2
            row["adjusted_overall_ci_low"] = (
                row["adjusted_dataset_ci_low"] + row["adjusted_market_ci_low"]
            ) / 2
            row["adjusted_overall_ci_high"] = (
                row["adjusted_dataset_ci_high"] + row["adjusted_market_ci_high"]
            ) / 2
            row["adjusted_overall_n"] = row["adjusted_dataset_n"] + row["adjusted_market_n"]
        else:
            for sfx in ("brier", "ci_low", "ci_high"):
                row[f"adjusted_overall_{sfx}"] = float("nan")
            row["adjusted_overall_n"] = 0

        rows.append(row)

    return rows
