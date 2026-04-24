#!/usr/bin/env python3
"""eval.py — Evaluate forecasts and generate HTML reports.

Usage:
    python3 src/eval.py --xid myxid

The xid must have "eval" (or fallback to "config"), "exam", and "metric" fields.
Optional "group" field defines source groupings for the leaderboard.

Reads from experiments/forecasts_raw/{config}/{source}/{id}.json
Generates in experiments/eval/{xid}/:
    leaderboard.html              (grouped leaderboard)
    dashboard.html                (per-question table)
    {metric}_vs_methods.png
    figs/metric_by_question_heatmaps/{metric}_heatmap_{source}.png
    figs/metric_by_question_heatmaps/steps_heatmap_{source}.png
    figs/metric_by_category_boxplot/{metric}_vs_category_{config}.png
    figs/tool_histograms/tool_histo_{source|overall}.png
    figs/tool_histograms/steps_histo_{source|overall}.png
    figs/tag_distribution_heatmap.png
"""

import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import argparse
import html as html_lib
import json
import math
import os

from config.config_display import config_struct, load_results_config, model_short_name
import re


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

SCORING_FUNCTIONS = {
    "brier-score": lambda p, o: (p - o) ** 2,
    "metaculus-score": lambda p, o: 100 * (1 + math.log2(max(0.001, min(0.999,
        p if o >= 0.5 else (1 - p))))),
    "brier-index": lambda p, o: 1 - math.sqrt((p - o) ** 2),
}

# Adjusted metrics are computed post-hoc (require FE data across all methods).
# They are injected into all_scores after initial scoring, not via SCORING_FUNCTIONS.
ADJUSTED_METRICS = {"adjusted-brier-score", "adjusted-brier-index"}

METRIC_LABELS = {
    "brier-score": ("Brier Score", "lower = better; baseline = 0.25"),
    "metaculus-score": ("Metaculus Score", "higher = better; 0 = chance"),
    "brier-index": ("Brier Index", "higher = better; 0.5 = chance"),
    "adjusted-brier-score": ("Adj. Brier Score", "lower = better; difficulty-adjusted"),
    "adjusted-brier-index": ("Adj. Brier Index", "higher = better; difficulty-adjusted"),
}

# Higher-is-better metrics (for sorting)
HIGHER_IS_BETTER = {"metaculus-score", "brier-index", "adjusted-brier-index"}

# Reference scores: {(ref_name, exam_pattern, metric): (value, description[, {group: value}])}.
# exam_pattern is matched as a prefix (e.g. "market" matches "market-train", "market-both").
# To use in eval, add "manual_reference": ["sota", ...] to the xid (NOT to "config").
# Optional third element: dict of per-group values for composite plots.
REFERENCE_SCORES = {
    # ForecastBench leaderboard (0-100 scale -> 0-1)
    # https://www.forecastbench.org/leaderboards/#tournament
    # SOTA = best LLM system
    ("sota", "market", "adjusted-brier-index"): (0.758, "Cassi ensemble_2_crowdadj (FB market=75.8)"),
    ("sota", "dataset", "adjusted-brier-index"): (0.615, "Cassi ensemble_2_crowdadj (FB dataset=61.5)"),
    # Superhuman = Superforecaster median forecast
    ("superhuman", "market", "adjusted-brier-index"): (0.802, "Superforecaster median (FB market=80.2)"),
    ("superhuman", "dataset", "adjusted-brier-index"): (0.638, "Superforecaster median (FB dataset=63.8)"),
    # AIBQ2
    ("sota", "aibq2", "metaculus-score"): (45.8, "thinkingmachines.ai AIBQ2 results"),
    # Ben-rich exam (Gemini 3.0 RiftRunner, 1P tools, from FB leaderboard)
    ("benrich", "ben-rich", "brier-index"): (0.6137, "Gemini 3.0 RiftRunner 1P tools (BI=61.37)", {
        "manifold": 0.985, "dbnomics": 0.670, "infer": 0.990,
        "wikipedia": 0.865, "polymarket": 0.755, "yfinance": 0.495, "fred": 0.545,
    }),
}

# All known reference names (derived from REFERENCE_SCORES keys)
_REFERENCE_NAMES = {k[0] for k in REFERENCE_SCORES}
# Pseudo-configs that should not be predicted
_PSEUDO_CONFIGS = _REFERENCE_NAMES | {"baseline"}


def _lookup_reference(ref_name: str, exam_name: str, metric: str):
    """Look up a reference score.

    Returns (value, description) or (value, description, group_values) or None.
    group_values is a dict {group_name: value} for per-group composite plots.
    """
    for (rn, exam_pat, m), entry in REFERENCE_SCORES.items():
        if rn == ref_name and exam_name.startswith(exam_pat) and m == metric:
            return entry  # (val, desc) or (val, desc, {group: val})
    return None


def _lookup_sota(exam_name: str, metric: str):
    """Look up SOTA score for an exam/metric pair. Returns (value, description) or None."""
    return _lookup_reference("sota", exam_name, metric)


def _resolve_outcome(val):
    """Normalize resolved_to: extract scalar from list if needed."""
    if isinstance(val, list):
        return val[0] if val else None
    return val


def compute_score(p: float, outcome: float, metric: str) -> float:
    return SCORING_FUNCTIONS[metric](p, outcome)


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
    """Path to the raw per-question forecast — used only by trace/dashboard code.

    Leaderboard and plots read from experiments/forecasts_final/{date}/
    {config}.json via core.eval.load_final_index.
    """
    from config.paths import forecast_path as _raw_path
    return _raw_path(config_name, source, qid)


def _esc(text):
    return html_lib.escape(str(text)) if text else ""


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

# (input $/M, output $/M) — output rate applies to reasoning tokens too.
_COST_PER_M_TOKENS = {
    # Direct Anthropic
    "anthropic/claude-sonnet-4-6":          (3.00,  15.00),
    "anthropic/claude-opus-4-6":            (5.00,  25.00),
    "anthropic/claude-haiku-4-5-20251001":  (0.80,   4.00),
    # OpenRouter — Anthropic
    "openrouter/anthropic/claude-opus-4-6":            (5.00,  25.00),
    "openrouter/anthropic/claude-sonnet-4-6":          (3.00,  15.00),
    "openrouter/anthropic/claude-haiku-4-5-20251001":  (1.00,   5.00),
    "openrouter/anthropic/claude-haiku-4-5":           (1.00,   5.00),
    # OpenRouter — DeepSeek
    "openrouter/deepseek/deepseek-v3.2":           (0.25,  0.40),
    "openrouter/deepseek/deepseek-r2":             (0.50,  2.00),
    # Direct Google
    "gemini/gemini-3.1-pro-preview":               (2.00, 12.00),
    "gemini/gemini-3-flash-preview":               (0.50,  3.00),
    # OpenRouter — Google
    "openrouter/google/gemini-3.1-pro-preview":    (2.00, 12.00),
    "openrouter/google/gemini-3-flash-preview":    (0.50,  3.00),
    # OpenRouter — xAI
    "openrouter/x-ai/grok-4":                     (3.00, 15.00),
    "openrouter/x-ai/grok-4-fast":                (0.20,  0.50),
    "openrouter/x-ai/grok-4.1-fast":              (0.20,  0.50),
    "openrouter/x-ai/grok-4.20-beta-20260309":    (2.00,  6.00),
    "openrouter/x-ai/grok-3":                     (3.00, 15.00),
    "openrouter/x-ai/grok-3-mini":                (0.30,  0.50),
    # OpenRouter — Moonshot
    "openrouter/moonshotai/kimi-k2":              (0.57,  2.30),
    "openrouter/moonshotai/kimi-k2-thinking":     (0.47,  2.00),
    "openrouter/moonshotai/kimi-k2.5":            (0.45,  2.20),
    # OpenRouter — Qwen
    "openrouter/qwen/qwen3-max-thinking":         (0.78,  3.90),
    "openrouter/qwen/qwen3-coder":                (0.30,  1.20),
}

# Cost per search query by engine
_SEARCH_API_COST = {
    "serper":      0.001,
    "brave":       0.003,
    "perplexity":  0.005,
    "asknews":     0.03,
}


def compute_cost(model: str, in_tok: int, out_tok: int) -> float | None:
    """Return dollar cost for token counts, or None if model not in pricing table."""
    rates = _COST_PER_M_TOKENS.get(model)
    if not rates:
        return None
    return (in_tok * rates[0] + out_tok * rates[1]) / 1_000_000


# ---------------------------------------------------------------------------
# Load and score forecasts
# ---------------------------------------------------------------------------

def parse_config_ref(config_ref: str) -> tuple[str, str | None]:
    """Parse a config reference like 'pro-high[mean:1]' into (base_config, agg_key).

    Returns (config_name, agg_key) where agg_key is e.g. "mean:1" or None.
    """
    m = re.match(r'^(.+?)\[(.+)\]$', config_ref)
    if m:
        return m.group(1), m.group(2)
    return config_ref, None


_DATE_SUFFIX_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})$")

def _strip_date_suffix(qid: str) -> tuple[str, str]:
    """Given an exam qid like 'abc...f790a_2025-10-26', return (base_id, date)."""
    m = _DATE_SUFFIX_RE.search(qid)
    if m:
        return qid[:m.start()], m.group(1)
    return qid, ""


# ---------------------------------------------------------------------------
# Final (collated) forecast loading
# ---------------------------------------------------------------------------

from config.paths import FORECASTS_FINAL_DIR as _FINAL_DIR  # noqa: E402


_index_cache: dict[str, dict[tuple[str, str], list[dict]]] = {}


def load_final_index(config_name: str) -> dict[tuple[str, str], list[dict]]:
    """Load every forecasts_final/{date}/{config_name}.json and index by (source, base_id).

    Each map value is a list of entries (multi-resolution questions have one
    entry per resolution_date). Projects `forecasts_aggregated[k]` and
    `forecasts_calibrated[k]` at the file level into per-entry `_agg` and
    `_cal` dicts so downstream lookups are entry-local.

    Cached per-config for the life of the process.
    """
    if config_name in _index_cache:
        return _index_cache[config_name]

    idx: dict[tuple[str, str], list[dict]] = {}
    if os.path.isdir(_FINAL_DIR):
        for date_dir in sorted(os.listdir(_FINAL_DIR)):
            p = os.path.join(_FINAL_DIR, date_dir, f"{config_name}.json")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                payload = json.load(f)
            entries = payload.get("forecasts", [])
            aggs = payload.get("forecasts_aggregated") or {}
            cals = payload.get("forecasts_calibrated") or {}
            for i, e in enumerate(entries):
                e["_agg"] = {k: (v[i] if i < len(v) else None)
                             for k, v in aggs.items()}
                e["_cal"] = {k: (v[i] if i < len(v) else None)
                             for k, v in cals.items()}
                raw_id = str(e.get("id", ""))
                base_id, _ = _strip_date_suffix(raw_id)
                idx.setdefault((e.get("source", ""), base_id), []).append(e)
    _index_cache[config_name] = idx
    return idx


def load_and_score(config_name: str, exam: dict[str, list[str]],
                   metrics: list[str]) -> dict[str, dict]:
    """Load forecasts for a config across all exam questions and compute scores.

    Reads from experiments/forecasts_final/{date}/{config_name}.json (one file
    per forecast_due_date, FB-style). Multi-resolution dataset questions
    contribute one entry per resolution_date and are averaged into a single
    score at the (source, qid) level.

    Returns {(source, qid): {id, source, forecast, outcome, metric_scores, ...}}
    """
    base_config, agg_key = parse_config_ref(config_name)
    index = load_final_index(base_config)

    results: dict[tuple[str, str], dict] = {}
    for source, ids in exam.items():
        for qid in ids:
            base_id, _ = _strip_date_suffix(qid)
            entries = index.get((source, base_id), [])
            if not entries:
                continue

            # Pull per-entry probabilities and outcomes.
            if agg_key:
                # Look in _agg first, then _cal. Both are per-entry dicts built
                # by load_final_index from file-level forecasts_aggregated /
                # forecasts_calibrated.
                ps = []
                for e in entries:
                    v = (e.get("_agg") or {}).get(agg_key)
                    if v is None:
                        v = (e.get("_cal") or {}).get(agg_key)
                    ps.append(v)
                if any(p is None for p in ps):
                    continue
            else:
                ps = [e.get("forecast") for e in entries]
            os_list = [_resolve_outcome(e.get("resolved_to")) for e in entries]

            usable = [(float(p), float(o)) for p, o in zip(ps, os_list)
                      if p is not None and o is not None]
            if not usable:
                continue

            # Score each (p, o) pair and average across resolution dates.
            scores: dict[str, float] = {}
            for m in metrics:
                if m not in SCORING_FUNCTIONS:
                    continue
                scores[m] = sum(compute_score(p, o, m) for p, o in usable) / len(usable)

            # Representative entry for metadata (first resolution date).
            e0 = entries[0]
            display_p = usable[0][0]
            outcome = usable[0][1]

            # Reconstitute question metadata from data/questions/ (collated
            # files strip question/background/resolution_criteria/url).
            q = {}
            qpath = os.path.join("data", "questions", source,
                                 re.sub(r'[/\\:]', '_', str(qid)) + ".json")
            if os.path.exists(qpath):
                with open(qpath) as qf:
                    q = json.load(qf)

            # Per-trial probabilities at each resolution-date entry.
            raw_trials_per_rdate = [e.get("raw_trials", []) for e in entries]

            cfg = e0.get("config") or {}

            results[(source, qid)] = {
                "id": base_id,
                "source": source,
                "qid": qid,
                "question": q.get("question", ""),
                "background": q.get("background", ""),
                "resolution_criteria": q.get("resolution_criteria", ""),
                "resolution_date": e0.get("resolution_date") or q.get("resolution_date", ""),
                "resolution_dates": [e.get("resolution_date") for e in entries],
                "forecast_due_date": e0.get("forecast_due_date")
                                     or q.get("forecast_due_date", ""),
                "url": q.get("url", ""),
                "forecast": display_p,
                "forecasts": ps,             # one value per resolution_date
                "resolved_to": [o for _, o in usable] if len(usable) > 1 else outcome,
                "outcome": outcome,
                "n_steps": e0.get("n_steps") or 0,
                "submitted": e0.get("submitted", True),
                "tokens_in": e0.get("tokens_in") or 0,
                "tokens_out": e0.get("tokens_out") or 0,
                "elapsed_seconds": e0.get("elapsed_seconds") or 0,
                "market_value": e0.get("market_value") or q.get("market_value"),
                "n_searches": e0.get("n_searches") or 0,
                "tool_counts": e0.get("tool_counts") or {},
                "llm": cfg.get("llm", ""),
                "search_engine": cfg.get("search_engine", ""),
                "raw_trials": raw_trials_per_rdate,  # list[list[float]], outer=res_dates
                **scores,
            }
    return results


# ---------------------------------------------------------------------------
# Synthetic baseline (crowd/market estimate for markets, 0.5 for others)
# ---------------------------------------------------------------------------

_MARKET_SOURCES_BASELINE = {"infer", "manifold", "metaculus", "polymarket"}


def load_baseline_scores(exam: dict[str, list[str]],
                         metrics: list[str]) -> dict[str, dict]:
    """Generate synthetic baseline forecasts: market_value for market questions, 0.5 otherwise.

    Reads question files from data/questions/{source}/{id}.json (not forecast files).
    """
    results = {}
    for source, ids in exam.items():
        for qid in ids:
            safe_id = re.sub(r'[/\\:]', '_', str(qid))
            path = os.path.join("data", "questions", source, f"{safe_id}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                q = json.load(f)
            outcome = _resolve_outcome(q.get("resolved_to"))
            if outcome is None:
                continue

            # Baseline forecast: market value for markets, 0.5 for everything else
            if source in _MARKET_SOURCES_BASELINE and q.get("market_value") is not None:
                try:
                    p = float(q["market_value"])
                except (ValueError, TypeError):
                    p = 0.5
            else:
                p = 0.5
            p = max(0.02, min(0.98, p))

            scores = {m: compute_score(p, outcome, m) for m in metrics
                      if m in SCORING_FUNCTIONS}
            results[(source, qid)] = {
                "id": q.get("id", "?"),
                "source": source,
                "qid": qid,
                "question": q.get("question", ""),
                "background": q.get("background", ""),
                "resolution_criteria": q.get("resolution_criteria", ""),
                "resolution_date": q.get("resolution_date", ""),
                "forecast_due_date": q.get("forecast_due_date", ""),
                "url": q.get("url", ""),
                "forecast": p,
                "outcome": outcome,
                "market_value": q.get("market_value"),
                "n_steps": 0,
                "submitted": True,
                "tokens_in": 0,
                "tokens_out": 0,
                "elapsed_seconds": 0,
                "n_searches": 0,
                "tool_counts": {},
                "llm": "baseline",
                "search_engine": "",
                **scores,
            }
    return results


# ---------------------------------------------------------------------------
# Adjusted Brier scores (difficulty-adjusted via question fixed effects)
# ---------------------------------------------------------------------------

_MARKET_SOURCES = {"infer", "manifold", "metaculus", "polymarket"}
_DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}


def inject_adjusted_scores(all_scores: dict[str, dict], metrics: list[str]):
    """Compute and inject per-question adjusted Brier scores into all_scores.

    For market questions: FE = (market_value - outcome)^2 (always available).
    For dataset questions: uses precomputed FE from forecastbench.org if available,
    otherwise estimates FE via alternating projections across methods.

    Injects 'adjusted-brier-score' and/or 'adjusted-brier-index' into each
    question record in all_scores.
    """
    adjusted_requested = [m for m in metrics if m in ADJUSTED_METRICS]
    if not adjusted_requested:
        return

    # Collect all question keys and their metadata
    all_keys = set()
    for scores in all_scores.values():
        all_keys.update(scores.keys())
    if not all_keys:
        return

    # --- Compute question fixed effects ---
    # Market FE: (market_value - outcome)^2
    market_fe = {}  # (source, qid) -> fe
    for key in all_keys:
        # Get metadata from first config that has this question
        for scores in all_scores.values():
            rec = scores.get(key)
            if rec:
                break
        if not rec:
            continue
        source = key[0]
        if source in _MARKET_SOURCES:
            mv = rec.get("market_value")
            outcome = rec.get("outcome")
            if mv is not None and outcome is not None:
                try:
                    market_fe[key] = (float(mv) - float(outcome)) ** 2
                except (ValueError, TypeError):
                    pass

    # Dataset FE: try precomputed from forecastbench.org
    dataset_fe = {}  # (source, qid) -> fe
    dataset_keys = {k for k in all_keys if k[0] in _DATASET_SOURCES}

    if dataset_keys:
        try:
            from eval.adjusted_brier import _fetch_latest_precomputed_fe
            fe_df = _fetch_latest_precomputed_fe("data/fb_cache/question_fixed_effects")
            if fe_df is not None:
                # Build lookup: (date, source, id, horizon) -> fe
                import pandas as pd
                fe_lookup = {}
                for _, row in fe_df.iterrows():
                    fe_lookup[(row["date"], row["source"], str(row["id"]),
                               row["horizon"])] = row["question_fixed_effect"]

                for key in dataset_keys:
                    rec = None
                    for scores in all_scores.values():
                        rec = scores.get(key)
                        if rec:
                            break
                    if not rec:
                        continue
                    fdd = rec.get("forecast_due_date", "")[:10]
                    rdate = rec.get("resolution_date", "")[:10]
                    source = key[0]
                    qid_base = str(rec.get("id", key[1]))
                    # Strip date suffix from dataset IDs for FE lookup
                    qid_base = re.sub(r'_\d{4}-\d{2}-\d{2}$', '', qid_base)
                    try:
                        horizon = (pd.Timestamp(rdate) - pd.Timestamp(fdd)).days
                    except Exception:
                        continue
                    fe_val = fe_lookup.get((fdd, source, qid_base, horizon))
                    if fe_val is not None:
                        dataset_fe[key] = fe_val

                if dataset_fe:
                    print(f"  Adjusted scores: {len(dataset_fe)} dataset questions "
                          f"matched precomputed FE")
        except Exception as e:
            print(f"  WARNING: could not load precomputed FE: {e}")

    # Fallback: estimate dataset FE from methods via alternating projections
    if dataset_keys and not dataset_fe:
        try:
            from eval.adjusted_brier import _estimate_question_fe
            import pandas as pd

            long_rows = []
            for config, scores in all_scores.items():
                for key in dataset_keys:
                    rec = scores.get(key)
                    if rec and rec.get("forecast") is not None:
                        brier = (rec["forecast"] - rec["outcome"]) ** 2
                        long_rows.append({
                            "question_pk": f"{key[0]}|{key[1]}",
                            "method": config,
                            "brier": brier,
                        })
            if long_rows:
                long_df = pd.DataFrame(long_rows)
                fe_dict = _estimate_question_fe(long_df)
                if fe_dict:
                    for key in dataset_keys:
                        pk = f"{key[0]}|{key[1]}"
                        if pk in fe_dict:
                            dataset_fe[key] = fe_dict[pk]
                    print(f"  Adjusted scores: estimated FE for {len(dataset_fe)} "
                          f"dataset questions from {long_df['method'].nunique()} methods")
        except Exception as e:
            print(f"  WARNING: could not estimate dataset FE: {e}")

    # Combine all FE
    all_fe = {**market_fe, **dataset_fe}
    if not all_fe:
        print("  WARNING: no fixed effects available, skipping adjusted metrics")
        return

    # Compute global mean FE (for rescaling so Always-0.5 → 0.25)
    global_mean_fe = sum(all_fe.values()) / len(all_fe)

    # Inject per-question adjusted scores
    n_injected = 0
    for config, scores in all_scores.items():
        for key, rec in scores.items():
            fe = all_fe.get(key)
            if fe is None:
                # No FE for this question: set adjusted metrics to NaN
                for m in adjusted_requested:
                    rec[m] = float("nan")
                continue
            brier = (rec["forecast"] - rec["outcome"]) ** 2
            adj_brier = brier - fe + global_mean_fe
            if "adjusted-brier-score" in adjusted_requested:
                rec["adjusted-brier-score"] = adj_brier
            if "adjusted-brier-index" in adjusted_requested:
                rec["adjusted-brier-index"] = 1 - math.sqrt(max(0, adj_brier))
            n_injected += 1

    n_questions = len(all_fe)
    n_configs = len(all_scores)
    print(f"  Adjusted scores: injected for {n_questions} questions × {n_configs} configs "
          f"(FE: {len(market_fe)} market, {len(dataset_fe)} dataset)")


# ---------------------------------------------------------------------------
# Group logic
# ---------------------------------------------------------------------------

def resolve_groups(group_spec: dict, available_sources: set[str]) -> dict[str, list[str]]:
    """Resolve group spec into {group_name: [source1, source2, ...]}.

    group_spec example:
        {"market": ["polymarket", "manifold"],
         "dataset": ["acled", "yfinance"],
         "overall": "all"}

    "all" means all available sources.
    """
    groups = {}
    for name, members in group_spec.items():
        if isinstance(members, str) and members.lower() == "all":
            groups[name] = sorted(available_sources)
        elif isinstance(members, list):
            groups[name] = [s for s in members if s in available_sources]
        elif isinstance(members, str):
            # Single source name (e.g. "polymarket": "polymarket")
            groups[name] = [members] if members in available_sources else []
        else:
            groups[name] = sorted(available_sources)
    return groups


def compute_group_means(scores: dict[str, dict], groups: dict[str, list[str]],
                        metric: str) -> dict[str, float | None]:
    """Compute mean metric for each group.

    Special handling for "overall" groups that use "all" sources:
    - Computed as equal-weighted mean of the other named groups (not a flat
      average over all sources). This ensures "overall" summarizes what's
      shown in the table, not hidden sources.

    For multi-source groups with adjusted metrics, uses equal-weighted source
    means (matching ForecastBench methodology).

    Returns {group_name: mean_score} or None if no data for that group.
    """
    # First pass: compute non-overall groups
    result = {}
    overall_groups = []  # groups that span multiple sources (candidates for "overall")
    for group_name, sources in groups.items():
        if not sources:
            result[group_name] = None
            continue
        # Detect "overall"-style groups: multi-source groups where sources
        # span more than just the sources in other single-source groups
        other_sources = set()
        for gn, gs in groups.items():
            if gn != group_name and len(gs) == 1:
                other_sources.update(gs)
        is_overall = (len(sources) > 1 and other_sources
                      and set(sources) > other_sources)

        if is_overall:
            overall_groups.append(group_name)
            continue  # defer to second pass

        if len(sources) > 1 and metric in ADJUSTED_METRICS:
            # Equal-weighted average of per-source means
            source_means = []
            for src in sources:
                vals = [s[metric] for key, s in scores.items()
                        if key[0] == src and metric in s
                        and not (isinstance(s[metric], float) and math.isnan(s[metric]))]
                if vals:
                    source_means.append(sum(vals) / len(vals))
            result[group_name] = (sum(source_means) / len(source_means)
                                  if source_means else None)
        else:
            vals = [s[metric] for key, s in scores.items()
                    if key[0] in sources and metric in s
                    and not (isinstance(s[metric], float) and math.isnan(s[metric]))]
            result[group_name] = sum(vals) / len(vals) if vals else None

    # Second pass: compute overall groups as equal-weighted mean of other groups
    for group_name in overall_groups:
        other_means = [v for gn, v in result.items()
                       if gn not in overall_groups and v is not None]
        result[group_name] = (sum(other_means) / len(other_means)
                              if other_means else None)

    return result


# ---------------------------------------------------------------------------
# Config stats
# ---------------------------------------------------------------------------

def _compute_config_stats(scores: dict) -> dict:
    """Compute avg steps, cost, elapsed, and search queries for a config's scored questions."""
    vals = list(scores.values())
    n = len(vals)
    if not n:
        return {}

    # n_steps and elapsed_seconds in aggregated files are summed across trials.
    # Divide by n_trials to get per-trial averages.
    n_trials = max(1, vals[0].get("trial_stats", {}).get("n_trials", 1)
                   if vals else 1)
    avg_steps = sum(v.get("n_steps", 0) for v in vals) / n / n_trials
    avg_elapsed = sum(v.get("elapsed_seconds", 0) for v in vals) / n / n_trials

    # Count timeouts (check tool_log for timeout entries)
    n_timeouts = sum(1 for v in vals
                     if any(e.get("type") == "timeout"
                            for e in v.get("tool_log", [])))

    # Determine model and search engine (from first record that has them)
    llm = next((v["llm"] for v in vals if v.get("llm")), "")
    search_engine = next((v["search_engine"] for v in vals if v.get("search_engine")), "")

    total_in = sum(v.get("tokens_in", 0) for v in vals)
    total_out = sum(v.get("tokens_out", 0) for v in vals)
    total_searches = sum(v.get("n_searches", 0) for v in vals)

    llm_cost = compute_cost(llm, total_in, total_out)
    search_cost = (total_searches * _SEARCH_API_COST[search_engine]
                   if search_engine in _SEARCH_API_COST else None)
    if llm_cost is not None or search_cost is not None:
        total_cost = (llm_cost or 0) + (search_cost or 0)
        cost_per_q = total_cost / n
    else:
        total_cost = None
        cost_per_q = None

    return {
        "avg_steps": avg_steps,
        "avg_elapsed": avg_elapsed,
        "n_timeouts": n_timeouts,
        "n_trials": n_trials,
        "cost_per_q": cost_per_q,
        "total_cost": total_cost,
        "n": n,
        "llm": llm,
        "search_engine": search_engine,
        "total_in": total_in,
        "total_out": total_out,
        "total_searches": total_searches,
        "llm_cost": llm_cost,
        "search_cost": search_cost,
    }


# ---------------------------------------------------------------------------
# Tag loading
# ---------------------------------------------------------------------------

def _get_plot_groups(xid_data: dict) -> list[str]:
    """Resolve which tag spaces to plot. Checks plot_groups, then groups, then default."""
    groups = xid_data.get("plot_groups", xid_data.get("groups"))
    if groups:
        return groups if isinstance(groups, list) else [groups]
    return ["FBQtype"]  # default


def _get_leaderboard_groups(xid_data: dict) -> list[str]:
    """Resolve which tag spaces to use for leaderboard columns.

    Returns list of label space names. Legacy "group" dicts are ignored here
    (handled separately in main).
    """
    groups = xid_data.get("leaderboard_groups", xid_data.get("groups"))
    if groups and isinstance(groups, list):
        return groups
    if groups and isinstance(groups, str):
        return [groups]
    return ["FBQtype"]  # default


def _get_metrics(xid_data: dict) -> list[str]:
    """Resolve metrics list (shared or per-use)."""
    metrics = xid_data.get("metrics", xid_data.get("metric",
              ["brier-index", "adjusted-brier-index", "metaculus-score"]))
    if isinstance(metrics, str):
        metrics = [metrics]
    return metrics

    # (Removed: on-the-fly aggregation variants. Use aggregate.py instead,
    # which materializes variants into trial_stats.aggregations, then reference
    # them in the xid's eval field as "config[method:n]".)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate forecasts")
    parser.add_argument("--xid", required=True,
                        help="Experiment ID (must have 'eval'/'config', 'exam', 'metric')")
    parser.add_argument("--add-calibration", nargs="?", const="*", default=None,
                        metavar="KEYS",
                        help="Also evaluate calibration variants. With no value, "
                             "auto-discovers every key in forecasts_calibrated. "
                             "Pass a comma-separated list to restrict "
                             "(e.g. --add-calibration global-cal,hier-cal). "
                             "Overrides the xid's 'eval_calibration' field.")
    parser.add_argument("--add-aggregation", nargs="?", const="*", default=None,
                        metavar="KEYS",
                        help="Same as --add-calibration but for forecasts_aggregated "
                             "(e.g. --add-aggregation mean:5,shrink5-loo).")
    parser.add_argument("--add-ensemble", default=None,
                        help="Also evaluate this ensemble config name "
                             "(ignored if xid has an explicit 'eval' field)")
    parser.add_argument("--plot-variants", default=None, metavar="NAMES",
                        help="Comma-separated subset of eval_names that get "
                             "per-method plots (scatter/ntrials/heatmap). The "
                             "leaderboard still includes every variant. "
                             "Defaults to the xid's 'plot_variants' field, "
                             "or all eval_names if neither is set.")
    parser.add_argument("--fast", action="store_true",
                        help="Skip plots and trace pages (leaderboard + scores only)")
    parser.add_argument("--fig-dir", default=None,
                        help="Subdirectory for figures (default: 'figs', "
                             "e.g. 'figs_agg' to avoid overwriting)")
    args = parser.parse_args()

    xid_data = load_xid(args.xid)

    if "exam" not in xid_data:
        sys.exit("ERROR: xid must have an 'exam' field")

    exam_name = xid_data["exam"]
    if "eval" in xid_data:
        eval_names = xid_data["eval"]
        if isinstance(eval_names, str):
            eval_names = [eval_names]
    else:
        base_configs = xid_data.get("config", [])
        if isinstance(base_configs, str):
            base_configs = [base_configs]
        eval_names = list(base_configs)
        if args.add_ensemble:
            for name in args.add_ensemble.split(","):
                name = name.strip()
                if name and name not in eval_names:
                    eval_names.append(name)
    # Strip any legacy pseudo-configs from eval_names (old xids may have
    # "sota" etc. in "config"; these now belong in "manual_reference")
    eval_names = [n for n in eval_names if n not in _REFERENCE_NAMES]
    # Resolve delta strings to filesystem names BEFORE adding suffixes
    from config.config import resolve_config, pprint_path
    resolved_names = []
    for name in eval_names:
        if "/" in name and ":" in name:
            cfg = resolve_config(name)
            resolved_names.append(pprint_path(cfg))
        else:
            resolved_names.append(name)
    eval_names = resolved_names
    # Add post-processing variants. Precedence: CLI > xid fields > none.
    # CLI "*" means auto-discover every available key; a comma list restricts
    # to those keys. Xid fields `eval_calibration` / `eval_aggregation`
    # provide persistent lists; the literal string "*" there also means
    # auto-discover.
    def _discover_keys(config: str, field: str) -> list[str]:
        base = _FINAL_DIR
        if not os.path.isdir(base):
            return []
        for date_dir in sorted(os.listdir(base)):
            p = os.path.join(base, date_dir, f"{config}.json")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                payload = json.load(f)
            return list((payload.get(field) or {}).keys())
        return []

    def _resolve_keys(cli_value, xid_value):
        """Return the list of variant keys to add (or None to skip).

        cli_value: None | "*" | "k1,k2,..."
        xid_value: None | "*" | ["k1", ...] | "k1,k2,..."
        """
        if cli_value is not None:
            raw = cli_value
        elif xid_value is not None:
            raw = xid_value
        else:
            return None
        if raw == "*" or (isinstance(raw, list) and raw == ["*"]):
            return "*"
        if isinstance(raw, list):
            return [str(k).strip() for k in raw if str(k).strip()]
        return [k.strip() for k in str(raw).split(",") if k.strip()]

    cal_keys = _resolve_keys(args.add_calibration,
                             xid_data.get("eval_calibration"))
    agg_keys = _resolve_keys(args.add_aggregation,
                             xid_data.get("eval_aggregation"))

    extra_names: list[str] = []
    for field, keys in (("forecasts_calibrated", cal_keys),
                        ("forecasts_aggregated", agg_keys)):
        if keys is None:
            continue
        for c in list(eval_names):
            if c in _PSEUDO_CONFIGS:
                continue
            if keys == "*":
                ks = _discover_keys(c, field)
            else:
                available = set(_discover_keys(c, field))
                ks = [k for k in keys if k in available]
                missing = [k for k in keys if k not in available]
                if missing:
                    print(f"  NOTE: {c} is missing {field} keys: {missing}")
            for k in ks:
                name = f"{c}[{k}]"
                if name not in eval_names and name not in extra_names:
                    extra_names.append(name)
    if (cal_keys is not None or agg_keys is not None) and not extra_names:
        print("  NOTE: --add-calibration/--add-aggregation requested but no "
              "matching keys found. Run aggregate.py or calibrate.py first.")
    eval_names.extend(extra_names)

    # --plot-variants gates which eval_names get per-method plots.
    plot_variants_spec = args.plot_variants or xid_data.get("plot_variants")
    if plot_variants_spec is None:
        plot_variants = None  # no restriction — plot everything
    elif isinstance(plot_variants_spec, list):
        plot_variants = [str(x) for x in plot_variants_spec]
    else:
        plot_variants = [s.strip() for s in str(plot_variants_spec).split(",")
                         if s.strip()]

    # Reference methods (display-only, from xid "manual_reference" field)
    ref_names = xid_data.get("manual_reference", [])
    if isinstance(ref_names, str):
        ref_names = [ref_names]

    # FB reference methods (auto-import from ForecastBench processed forecasts)
    fb_refs = xid_data.get("fb_reference", [])
    if isinstance(fb_refs, str):
        fb_refs = [fb_refs]
    if fb_refs:
        from data.fb_leaderboard import import_method
        for method_key in fb_refs:
            cfg = "fb-" + re.sub(r'[^a-zA-Z0-9]+', '-',
                                  method_key.removeprefix("external.")).strip('-').lower()
            if cfg not in eval_names:
                eval_names.append(cfg)
            # Import if not already present
            import_method(exam_name, method_key, config_name=cfg)

    metrics = _get_metrics(xid_data)

    # Load exam
    exam = load_exam(exam_name)
    available_sources = set(exam.keys())

    # Resolve leaderboard groups from tag space
    # New style: "groups": ["FBQtype"] -> columns = tag values + overall
    # Legacy: "group": {"polymarket": "polymarket", ...} -> old source-based
    lb_group_spaces = _get_leaderboard_groups(xid_data)
    if "group" in xid_data and "groups" not in xid_data:
        # Legacy format — use old resolve_groups
        group_spec = xid_data["group"]
        groups = resolve_groups(group_spec, available_sources)
    else:
        # New tag-based format: build groups from first leaderboard group space
        from config.tags import get_tags_for_exam
        lb_space = lb_group_spaces[0] if lb_group_spaces else "FBQtype"
        lb_tags = get_tags_for_exam(exam, lb_space)
        # Build source-based groups from tag values
        from collections import defaultdict as _ddict
        tag_to_sources = _ddict(set)
        for (src, _), tag_val in lb_tags.items():
            tag_to_sources[tag_val].add(src)
        groups = {tv: sorted(srcs) for tv, srcs in sorted(tag_to_sources.items())}
        groups["overall"] = sorted(available_sources)

    output_dir = os.path.join("experiments", "eval", args.xid)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Evaluation: {args.xid}")
    print(f"  Exam: {exam_name}")
    print(f"  Methods: {eval_names}")
    if ref_names:
        print(f"  References: {ref_names}")
    print(f"  Metrics: {metrics}")
    print(f"  Groups: {list(groups.keys())}")

    # Score all configs
    all_scores = {}
    for config in eval_names:
        if config == "baseline":
            scores = load_baseline_scores(exam, metrics)
        else:
            scores = load_and_score(config, exam, metrics)
        all_scores[config] = scores
        n = len(scores)
        if n:
            sample_metric = metrics[0]
            mean_val = sum(s[sample_metric] for s in scores.values()) / n
            print(f"  [{config}] n={n}, mean {sample_metric}={mean_val:.4f}")
        else:
            print(f"  [{config}] no scored forecasts")

    # Resolve reference scores
    for rn in ref_names:
        found = [m for m in metrics if _lookup_reference(rn, exam_name, m)]
        if found:
            print(f"  [{rn}] reference scores for: {found}")
        else:
            print(f"  [{rn}] no reference scores for this exam/metric combination")

    # Inject adjusted metrics if requested
    inject_adjusted_scores(all_scores, metrics)

    # Compute aggregation variants on the fly
    # If --fig-dir is set, plots go to output_dir/{fig_dir}/ instead of output_dir/figs/
    # We achieve this by temporarily symlinking figs -> fig_dir
    _figs_symlink = None
    if args.fig_dir and args.fig_dir != "figs":
        figs_path = os.path.join(output_dir, "figs")
        target_path = os.path.join(output_dir, args.fig_dir)
        os.makedirs(target_path, exist_ok=True)
        # Remove existing figs if it's a symlink (from previous run)
        if os.path.islink(figs_path):
            os.unlink(figs_path)
        if not os.path.exists(figs_path):
            os.symlink(args.fig_dir, figs_path)
            _figs_symlink = figs_path
            print(f"  Figures -> {target_path}/")

    # --- Import sub-modules for report generation ---
    from eval.eval_html import generate_leaderboard, generate_details
    from eval.eval_plots import (
        generate_metric_plot, generate_relative_metric_plot,
        generate_metric_vs_std_scatter, generate_metric_vs_ntrials,
        generate_metric_vs_questions,
        generate_heatmap, generate_tool_histogram, generate_steps_histogram,
        generate_steps_heatmap, generate_metric_vs_category_composite,
        generate_metric_vs_horizon,
        generate_calibration_curves, generate_ece_histogram,
        generate_metric_by_time, generate_metric_time_histos,
    )
    from eval.eval_trace import generate_detail_pages



    # Generate detail pages for base configs (not calibrated)
    if not args.fast:
        for config in eval_names:
            if not config.endswith("_calibrated"):
                generate_detail_pages(config, exam)

    # Generate leaderboard
    lb_path = generate_leaderboard(output_dir, eval_names, all_scores,
                                   metrics, groups, args.xid,
                                   exam_name=exam_name,
                                   ref_names=ref_names)
    print(f"\n  Leaderboard: {lb_path}")

    # Generate details
    det_path = generate_details(output_dir, eval_names, all_scores, args.xid)
    print(f"  Details: {det_path}")

    if args.fast:
        print("\n  --fast: skipping plots and traces")
        print("\nDone.")
        return

    # Generate metric plots
    for metric in metrics:
        p = generate_metric_plot(output_dir, eval_names, all_scores, metric, args.xid,
                                 exam_name=exam_name,
                                 agg_mode=any("[" in c for c in eval_names),
                                 ref_names=ref_names)
        if p:
            print(f"  Plot: {p}")
        # Relative metric plot (score - best per question)
        p = generate_relative_metric_plot(output_dir, eval_names, all_scores,
                                          metric, args.xid, exam_name=exam_name)
        if p:
            print(f"  Relative plot: {p}")
        # Per-method plots. `plot_variants` (CLI or xid field) restricts
        # which eval_names get these potentially expensive plots.
        if plot_variants is None:
            plot_names = eval_names
        else:
            plot_names = [n for n in eval_names if n in plot_variants]
        # Metric vs std scatter (multi-trial configs only)
        for config in plot_names:
            if config in _PSEUDO_CONFIGS:
                continue
            scores = all_scores.get(config, {})
            if scores:
                p = generate_metric_vs_std_scatter(
                    output_dir, config, scores, metric, args.xid)
                if p:
                    print(f"  Scatter: {p}")
        # Metric vs ntrials (base configs with trials only, skip calibrated/bon)
        # Skip adjusted metrics — they require FE data that can't be recomputed per-trial
        if metric not in ADJUSTED_METRICS:
            for config in plot_names:
                if config in _PSEUDO_CONFIGS:
                    continue
                if config.endswith(("_calibrated", "_bon")):
                    continue
                scores = all_scores.get(config, {})
                if scores:
                    p = generate_metric_vs_ntrials(
                        output_dir, config, scores, metric, xid_name=args.xid)
                    if p:
                        print(f"  Ntrials plot: {p}")
            # Per-question metric with cross-trial CI (multi-trial configs only)
            for config in plot_names:
                if config in _PSEUDO_CONFIGS:
                    continue
                if config.endswith(("_calibrated", "_bon")):
                    continue
                scores = all_scores.get(config, {})
                if scores:
                    paths = generate_metric_vs_questions(
                        output_dir, config, scores, metric, xid_name=args.xid)
                    for p in paths:
                        print(f"  Per-question: {p}")

    # Generate heatmaps (one per metric × source + overall)
    for metric in metrics:
        for source in sorted(available_sources):
            p = generate_heatmap(output_dir, eval_names, all_scores,
                                 metric, source, args.xid)
            if p:
                print(f"  Heatmap: {p}")
        # Overall heatmap (all sources combined)
        if len(available_sources) > 1:
            p = generate_heatmap(output_dir, eval_names, all_scores,
                                 metric, None, args.xid)
            if p:
                print(f"  Heatmap: {p}")

    # Generate tool-call histograms (per source + overall)
    for source in sorted(available_sources):
        p = generate_tool_histogram(output_dir, eval_names, all_scores,
                                    source_filter=source, xid_name=args.xid)
        if p:
            print(f"  Tool histo: {p}")
    p = generate_tool_histogram(output_dir, eval_names, all_scores,
                                source_filter=None, xid_name=args.xid)
    if p:
        print(f"  Tool histo: {p}")

    # Generate steps histograms (per source + overall)
    for source in sorted(available_sources):
        p = generate_steps_histogram(output_dir, eval_names, all_scores,
                                     source_filter=source, xid_name=args.xid)
        if p:
            print(f"  Steps histo: {p}")
    p = generate_steps_histogram(output_dir, eval_names, all_scores,
                                 source_filter=None, xid_name=args.xid)
    if p:
        print(f"  Steps histo: {p}")

    # Generate steps heatmaps (per source)
    for source in sorted(available_sources):
        p = generate_steps_heatmap(output_dir, eval_names, all_scores,
                                   source, xid_name=args.xid)
        if p:
            print(f"  Steps heatmap: {p}")

    # Generate metric-by-tag composite plots for each plot_group
    from config.tags import get_tags_for_exam, discover_classified_spaces
    plot_groups = _get_plot_groups(xid_data)
    # Auto-add classified spaces (xinghua, ben) if available
    classified = discover_classified_spaces()
    all_groups = list(dict.fromkeys(plot_groups + classified))  # dedupe, preserve order

    for label_space in all_groups:
        tags = get_tags_for_exam(exam, label_space)
        if not tags:
            continue
        # Metric composite: grouped bars per tag value, one per config
        for metric in metrics:
            p = generate_metric_vs_category_composite(
                output_dir, eval_names, all_scores, tags, metric, args.xid,
                tag_version=label_space,
                ref_names=ref_names, exam_name=exam_name)
            if p:
                print(f"  Composite ({label_space}): {p}")

    # Generate horizon plots
    for metric in metrics:
        p = generate_metric_vs_horizon(output_dir, eval_names, all_scores,
                                       metric, args.xid)
        if p:
            print(f"  Horizon plot: {p}")

    # Generate metric-by-time plots (metric vs forecast_due_date)
    for metric in metrics:
        p = generate_metric_by_time(output_dir, eval_names, all_scores,
                                    metric, args.xid)
        if p:
            print(f"  Time plot: {p}")
        # Per-config time histograms
        for config in eval_names:
            if config in _PSEUDO_CONFIGS:
                continue
            scores = all_scores.get(config, {})
            if scores:
                p = generate_metric_time_histos(
                    output_dir, config, scores, metric, args.xid)
                if p:
                    print(f"  Time histo: {p}")

    # Generate calibration plots
    for p in generate_calibration_curves(output_dir, eval_names, all_scores, args.xid):
        print(f"  Calibration: {p}")
    p = generate_ece_histogram(output_dir, eval_names, all_scores, args.xid)
    if p:
        print(f"  ECE histo: {p}")

    # Clean up fig-dir symlink
    if _figs_symlink and os.path.islink(_figs_symlink):
        os.unlink(_figs_symlink)

    print("\nDone.")


if __name__ == "__main__":
    main()
