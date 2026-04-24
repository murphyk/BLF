"""eval_plots.py — Matplotlib plots for eval reports."""

import json
import math
import os
import warnings

warnings.filterwarnings("ignore", message="Tight layout not applied")

from config.config_display import config_struct, load_results_config
from core.eval import (
    SCORING_FUNCTIONS, METRIC_LABELS, HIGHER_IS_BETTER,
    _REFERENCE_NAMES, _lookup_reference, forecast_path,
)

# Display names for pseudo-configs
_DISPLAY_NAMES = {
    "baseline": "market/uniform",
    "sota": "sota*",
    "superhuman": "superhuman*",
}


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def _bootstrap_ci(vals, n_bootstrap=1000, seed=42):
    import numpy as np
    n = len(vals)
    mean = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    boot = [float(np.mean(rng.choice(vals, size=n, replace=True)))
            for _ in range(n_bootstrap)]
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# Metric dot-plot
# ---------------------------------------------------------------------------

def generate_metric_plot(output_dir: str, eval_names: list[str],
                         all_scores: dict[str, dict[str, dict]],
                         metric: str, xid_name: str = "",
                         exam_name: str = "",
                         agg_mode: bool = False,
                         ref_names: list[str] | None = None) -> str | None:
    """Dot plot with bootstrap CI comparing methods.

    agg_mode: if True, use simple config name + [agg,n] labels instead of
    structured config_display labels (designed for comparing aggregation methods).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plots")
        return None

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))

    rows = []      # (name, mean, lo, hi, n, is_ref)
    ref_rows = []  # reference entries
    for rn in (ref_names or []):
        ref = _lookup_reference(rn, exam_name, metric)
        if ref:
            ref_rows.append((rn, ref[0], ref[0], ref[0], 0, True))
    for name in eval_names:
        scores = all_scores.get(name, {})
        if not scores:
            continue
        raw = [v[metric] for v in scores.values()
               if metric in v and not (isinstance(v[metric], float) and np.isnan(v[metric]))]
        if not raw:
            continue
        # For adjusted metrics, use equal-weighted source means (matching
        # ForecastBench methodology) so the adjustment doesn't cancel out.
        if metric in ("adjusted-brier-score", "adjusted-brier-index"):
            source_vals = {}  # source -> [values]
            for key, v in scores.items():
                if metric in v and not (isinstance(v[metric], float) and np.isnan(v[metric])):
                    source_vals.setdefault(key[0], []).append(v[metric])
            if len(source_vals) > 1:
                # Bootstrap over equal-weighted source means
                source_arrays = {s: np.array(vs) for s, vs in source_vals.items()}
                grand_mean = np.mean([np.mean(vs) for vs in source_arrays.values()])
                rng = np.random.default_rng(42)
                boots = []
                for _ in range(1000):
                    src_means = []
                    for vs in source_arrays.values():
                        idx = rng.choice(len(vs), size=len(vs), replace=True)
                        src_means.append(float(np.mean(vs[idx])))
                    boots.append(np.mean(src_means))
                mean = float(grand_mean)
                lo = float(np.percentile(boots, 2.5))
                hi = float(np.percentile(boots, 97.5))
                rows.append((name, mean, lo, hi, len(raw), False))
                continue
        vals = np.array(raw)
        mean, lo, hi = _bootstrap_ci(vals)
        rows.append((name, mean, lo, hi, len(vals), False))

    if not rows and not ref_rows:
        return None

    reverse = metric in HIGHER_IS_BETTER
    all_rows = rows + ref_rows
    all_rows.sort(key=lambda r: r[1], reverse=reverse)

    raw_labels = [r[0] for r in all_rows]
    means = [r[1] for r in all_rows]
    lo = [r[2] for r in all_rows]
    hi = [r[3] for r in all_rows]
    ns = [r[4] for r in all_rows]
    is_ref = [r[5] for r in all_rows]
    labels = [_DISPLAY_NAMES.get(l, l) for l in raw_labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    x = list(range(len(labels)))

    # Plot regular methods as circles with CI
    reg_x = [xi for xi, ref in zip(x, is_ref) if not ref]
    reg_m = [m for m, ref in zip(means, is_ref) if not ref]
    reg_lo = [l for l, ref in zip(lo, is_ref) if not ref]
    reg_hi = [h for h, ref in zip(hi, is_ref) if not ref]
    if reg_x:
        ax.errorbar(reg_x, reg_m,
                    yerr=[np.array(reg_m) - np.array(reg_lo),
                          np.array(reg_hi) - np.array(reg_m)],
                    fmt='o', color='#2980b9', capsize=4, markersize=8)

    # Plot SOTA as diamonds
    ref_x = [xi for xi, ref in zip(x, is_ref) if ref]
    ref_m = [m for m, ref in zip(means, is_ref) if ref]
    if ref_x:
        ax.scatter(ref_x, ref_m, marker='D', color='#e74c3c', s=80, zorder=5)

    for i, (m, ref) in enumerate(zip(means, is_ref)):
        fmt_str = ".1f" if metric in ("metaculus-score",) else ".4f" if metric == "brier-score" else ".3f"
        color = '#e74c3c' if ref else '#2980b9'
        ax.annotate(f"{m:{fmt_str}}",
                    (i, m), textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=9, color=color)

    # Build x-tick labels
    if agg_mode:
        # Aggregation mode: show config name with aggregation method
        tick_labels = []
        for name in labels:
            if "[" in name:
                # Has aggregation bracket: "pro-high[mean:1]"
                base_part, agg_part = name.split("[", 1)
                base_part = base_part.rstrip()
                suffix = ""
                if base_part.endswith("_calibrated"):
                    base_part = base_part.removesuffix("_calibrated")
                    suffix = "+cal"
                label = f"{base_part}\n[{agg_part}"
                if suffix:
                    label = label.rstrip("]") + f",{suffix}]"
                tick_labels.append(label)
            elif name.endswith("_calibrated"):
                base = name.removesuffix("_calibrated")
                tick_labels.append(f"{base}\n[shrink,+cal]")
            elif name.endswith("_aggregated"):
                base = name.removesuffix("_aggregated")
                tick_labels.append(f"{base}\n[llm-agg]")
            else:
                tick_labels.append(f"{name}\n[shrink]")
        ax.set_xticks(list(x))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    else:
        # Structured config_display mode
        config_structs = []
        for name in raw_labels:
            cfg = load_results_config(name)
            if cfg:
                config_structs.append(config_struct(cfg))
            else:
                config_structs.append(None)

        all_fields = ["model", "think", "search", "crowd", "tools", "ntrials"]
        field_short = {"model": "", "think": "thk", "search": "srch",
                       "crowd": "crowd", "tools": "tools", "ntrials": "n"}
        invariant = {}
        varying_fields = []
        for field in all_fields:
            vals = set()
            for cs in config_structs:
                if cs:
                    vals.add(str(cs.get(field, "")))
            if len(vals) == 1:
                invariant[field] = vals.pop()
            else:
                varying_fields.append(field)

        def _suffix(name):
            if name.endswith("_aggregated_calibrated"):
                return "+agg+cal"
                return "+bon+cal"
            if name.endswith("_aggregated"):
                return "+agg"
            if name.endswith("_calibrated"):
                return "+cal"
                return "+bon"
            return ""

        suffixes = set(_suffix(n) for n in raw_labels)
        suffix_varies = len(suffixes) > 1

        tick_labels = []
        for raw_name, disp_name, cs in zip(raw_labels, labels, config_structs):
            if cs is None:
                tick_labels.append(disp_name)
                continue
            parts = []
            for field in varying_fields:
                v = cs.get(field, "")
                short = field_short[field]
                parts.append(f"{short}={v}" if short else str(v))
            s = _suffix(raw_name)
            if suffix_varies and s:
                parts.append(s)
            tick_labels.append(", ".join(parts) if parts else disp_name)

        ax.set_xticks(list(x))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)

    # Append (n=X) to tick labels when sample sizes differ across methods
    all_ns_set = set(n for n, ref in zip(ns, is_ref) if not ref)
    if len(all_ns_set) > 1:
        current = [t.get_text() for t in ax.get_xticklabels()]
        new_labels = []
        for txt, n, ref in zip(current, ns, is_ref):
            if ref:
                new_labels.append(txt)
            else:
                new_labels.append(f"{txt}\n(n={n})")
        ax.set_xticklabels(new_labels, rotation=45, ha='right', fontsize=9)

    ax.set_ylabel(label)

    # Build subtitle
    real_ns = [n for n, ref in zip(ns, is_ref) if not ref]
    n_display = real_ns[0] if real_ns and len(set(real_ns)) == 1 else (
        f"{min(real_ns)}-{max(real_ns)}" if real_ns else "?")
    exam_str = f" \u2014 {exam_name}" if exam_name else ""

    if agg_mode:
        subtitle_line2 = f"(nquestions={n_display})"
    else:
        # Invariant config summary (e.g. "gemini-3.1-pro / brave / c=1 / tools=1 / ntrials=5")
        inv_parts = []
        for field in all_fields:
            if field in invariant:
                v = invariant[field]
                if field == "ntrials" and v == "1":
                    continue
                short = field_short[field]
                if field == "ntrials":
                    inv_parts.append(f"ntrials={v}")
                elif short:
                    inv_parts.append(f"{short}={v}")
                else:
                    inv_parts.append(v)
        inv_str = " / ".join(inv_parts) if inv_parts else ""
        subtitle_line2 = f"({inv_str}; nquestions={n_display})" if inv_str else f"(nquestions={n_display})"

    ax.set_title(f"{label}{exam_str} ({subtitle})\n{subtitle_line2}", fontsize=10)

    # Chance reference line (skip if explicit baseline method is present)
    has_baseline = "baseline" in eval_names
    if not has_baseline:
        if metric in ("brier-score", "adjusted-brier-score"):
            ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
        elif metric in ("metaculus-score",):
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        elif metric in ("brier-index", "adjusted-brier-index"):
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Reference footnote
    if ref_rows:
        ref_names_present = [r[0] for r in ref_rows]
        parts = []
        if "sota" in ref_names_present:
            parts.append("sota (Cassi ensemble_2_crowdadj)")
        if "superhuman" in ref_names_present:
            parts.append("superhuman (Superforecaster median)")
        if parts:
            footnote = (f"*{' and '.join(parts)} from "
                        f"forecastbench.org/leaderboards/#tournament (different dataset)")
            fig.text(0.01, 0.01, footnote, fontsize=7, color='gray',
                     ha='left', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 1])  # leave room for footnote
    safe_metric = metric.replace("-", "_")
    figs_dir = os.path.join(output_dir, "figs")
    methods_dir = os.path.join(figs_dir, "metric_by_method")
    out_path = os.path.join(methods_dir, f"{safe_metric}_vs_methods.png")
    os.makedirs(methods_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Relative metric plot
# ---------------------------------------------------------------------------

def generate_relative_metric_plot(output_dir: str, eval_names: list[str],
                                  all_scores: dict[str, dict[str, dict]],
                                  metric: str, xid_name: str = "",
                                  exam_name: str = "") -> str | None:
    """Dot plot of metric relative to the best method per question."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    higher = metric in HIGHER_IS_BETTER

    # Collect per-question scores for all methods (excluding SOTA/reference)
    real_names = [n for n in eval_names if n in all_scores]
    if len(real_names) < 2:
        return None

    # Find common questions
    all_keys = set()
    for name in real_names:
        all_keys.update(all_scores[name].keys())

    # For each question, compute best score and each method's relative score
    method_diffs = {name: [] for name in real_names}
    for key in sorted(all_keys):
        q_scores = {}
        for name in real_names:
            rec = all_scores[name].get(key)
            if rec and metric in rec:
                v = rec[metric]
                if not (isinstance(v, float) and np.isnan(v)):
                    q_scores[name] = v
        if len(q_scores) < 2:
            continue
        best = max(q_scores.values()) if higher else min(q_scores.values())
        for name, val in q_scores.items():
            if higher:
                method_diffs[name].append(val - best)
            else:
                method_diffs[name].append(best - val)

    rows = []
    for name in real_names:
        diffs = method_diffs[name]
        if not diffs:
            continue
        vals = np.array(diffs)
        mean, lo, hi = _bootstrap_ci(vals)
        rows.append((name, mean, lo, hi, len(vals)))

    if not rows:
        return None

    # Sort: closest to 0 (best) first
    rows.sort(key=lambda r: r[1], reverse=True)

    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    lo = [r[2] for r in rows]
    hi = [r[3] for r in rows]
    ns = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    x = list(range(len(labels)))
    ax.errorbar(x, means,
                yerr=[np.array(means) - np.array(lo),
                      np.array(hi) - np.array(means)],
                fmt='o', color='#2980b9', capsize=4, markersize=8)
    for i, m in enumerate(means):
        fmt_str = ".1f" if metric in ("metaculus-score",) else ".4f" if metric == "brier-score" else ".3f"
        ax.annotate(f"{m:{fmt_str}}",
                    (i, m), textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=9, color='#2980b9')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel(f"Relative {label}")
    n_display = ns[0] if len(set(ns)) == 1 else f"{min(ns)}-{max(ns)}"
    exam_str = f" \u2014 {exam_name}" if exam_name else ""
    ax.set_title(f"Relative {label}{exam_str}\n"
                 f"(score \u2212 best per question; 0 = best; nquestions={n_display})", fontsize=10)

    plt.tight_layout()
    safe_metric = metric.replace("-", "_")
    methods_dir = os.path.join(output_dir, "figs", "metric_by_method")
    out_path = os.path.join(methods_dir, f"rel_{safe_metric}_vs_methods.png")
    os.makedirs(methods_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Metric vs std scatter
# ---------------------------------------------------------------------------

def generate_metric_vs_std_scatter(output_dir: str, config: str,
                                   scores: dict[tuple, dict],
                                   metric: str,
                                   xid_name: str = "") -> str | None:
    """Scatter plot of per-question metric vs forecast std across trials."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    label, _ = METRIC_LABELS.get(metric, (metric, ""))

    # Collect (std, metric_score, correct) for each question
    xs, ys, correct = [], [], []
    for key, rec in scores.items():
        if metric not in rec:
            continue
        v = rec[metric]
        if isinstance(v, float) and math.isnan(v):
            continue
        # raw_trials on the score record is list[list[float]] (outer = res_dates).
        # Flatten and compute cross-trial std for the first resolution.
        rt_outer = rec.get("raw_trials") or []
        rt = rt_outer[0] if rt_outer and isinstance(rt_outer[0], list) else rt_outer
        if len(rt) <= 1:
            continue
        mean = sum(rt) / len(rt)
        std = (sum((p - mean) ** 2 for p in rt) / len(rt)) ** 0.5
        xs.append(std)
        ys.append(v)
        p = rec.get("forecast", 0.5)
        o = rec.get("outcome")
        correct.append((o == 1 and p > 0.5) or (o == 0 and p < 0.5))

    if len(xs) < 5:
        return None

    xs = np.array(xs)
    ys = np.array(ys)
    correct = np.array(correct)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot correct vs incorrect
    ax.scatter(xs[correct], ys[correct], c='#27ae60', alpha=0.6, s=40,
               label=f'Correct ({correct.sum()})', edgecolors='white', linewidth=0.5)
    ax.scatter(xs[~correct], ys[~correct], c='#c0392b', alpha=0.6, s=40,
               label=f'Incorrect ({(~correct).sum()})', edgecolors='white', linewidth=0.5)

    # Trend line (skip if zero variance in x)
    if len(xs) > 2 and xs.std() > 1e-10:
        try:
            z = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), '--', color='#555', alpha=0.5)
            corr = np.corrcoef(xs, ys)[0, 1]
            ax.set_title(f"{label} vs Forecast Std \u2014 {config}\n"
                         f"(r={corr:.3f}, n={len(xs)}; {xid_name})", fontsize=10)
        except (np.linalg.LinAlgError, ValueError):
            ax.set_title(f"{label} vs Forecast Std \u2014 {config}\n({xid_name})", fontsize=10)
    else:
        ax.set_title(f"{label} vs Forecast Std \u2014 {config}\n({xid_name})", fontsize=10)

    ax.set_xlabel("Forecast Std (across trials)")
    ax.set_ylabel(label)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_metric = metric.replace("-", "_")
    figs_dir = os.path.join(output_dir, "figs")
    scatter_dir = os.path.join(figs_dir, "std_scatter")
    out_path = os.path.join(scatter_dir, f"{safe_metric}_vs_std_scatter_{config}.png")
    os.makedirs(scatter_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Metric vs ntrials
# ---------------------------------------------------------------------------

def generate_metric_vs_ntrials(output_dir: str, config: str,
                               scores: dict[tuple, dict],
                               metric: str,
                               shrinkage: float | None = None,
                               xid_name: str = "") -> str | None:
    """Plot metric as a function of number of trials averaged."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from itertools import combinations
    except ImportError:
        return None

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    higher = metric in HIGHER_IS_BETTER

    # Collect per-question trial forecasts and outcomes
    questions = []  # list of (outcome, {trial_num: forecast})
    for key, rec in scores.items():
        rt_outer = rec.get("raw_trials") or []
        rt = rt_outer[0] if rt_outer and isinstance(rt_outer[0], list) else rt_outer
        if len(rt) < 2:
            continue
        outcome = rec.get("outcome")
        if outcome is None:
            continue
        trial_forecasts = {i + 1: p for i, p in enumerate(rt)}
        questions.append((outcome, trial_forecasts))

    if not questions:
        return None

    max_trials = max(len(tf) for _, tf in questions)
    if max_trials < 2:
        return None

    def score_fn(p, o):
        return SCORING_FUNCTIONS[metric](p, o)

    def aggregate(ps_list, shrink):
        """Aggregate a list of forecasts with shrinkage in logit space."""
        arr = np.array(ps_list)
        if shrink is None and len(arr) > 1:
            logits = np.log(np.clip(arr, 0.001, 0.999) / (1 - np.clip(arr, 0.001, 0.999)))
            logit_bar = float(np.mean(logits))
            std_logit = float(np.std(logits))
            a = max(0.3, 1.0 - 0.7 * std_logit)
            return float(1 / (1 + np.exp(-a * logit_bar)))
        elif shrink is not None and shrink < 1.0:
            logits = np.log(np.clip(arr, 0.001, 0.999) / (1 - np.clip(arr, 0.001, 0.999)))
            logit_bar = float(np.mean(logits))
            return float(1 / (1 + np.exp(-shrink * logit_bar)))
        return float(np.mean(arr))

    # Get common trial nums across all questions
    common_trials = None
    for _, tf in questions:
        if common_trials is None:
            common_trials = set(tf.keys())
        else:
            common_trials &= set(tf.keys())
    common_trials = sorted(common_trials)
    n_common = len(common_trials)
    if n_common < 2:
        return None

    ks = list(range(1, n_common + 1))

    def _compute_curve(shrink_val):
        """Compute mean + CI for each k with a given shrinkage setting."""
        curve_means, curve_los, curve_his = [], [], []
        for k in ks:
            all_subsets = list(combinations(common_trials, k))
            if len(all_subsets) > 100:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(all_subsets), 100, replace=False)
                all_subsets = [all_subsets[i] for i in idx]

            subset_scores = []
            for subset in all_subsets:
                q_scores = []
                for outcome, tf in questions:
                    ps = [tf[t] for t in subset]
                    agg_p = aggregate(ps, shrink_val)
                    q_scores.append(score_fn(agg_p, outcome))
                subset_scores.append(np.mean(q_scores))

            arr = np.array(subset_scores)
            curve_means.append(float(np.mean(arr)))
            curve_los.append(float(np.percentile(arr, 2.5)))
            curve_his.append(float(np.percentile(arr, 97.5)))
        return curve_means, curve_los, curve_his

    # Compute both curves
    shrink_means, shrink_los, shrink_his = _compute_curve(None)   # std-based
    plain_means, plain_los, plain_his = _compute_curve(1.0)       # plain mean

    # Plot both on same figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plain mean (gray, thinner)
    _plain_lo = np.maximum(0, np.array(plain_means) - np.array(plain_los))
    _plain_hi = np.maximum(0, np.array(plain_his) - np.array(plain_means))
    ax.errorbar(ks, plain_means,
                yerr=[_plain_lo, _plain_hi],
                fmt='s--', color='#888', capsize=3, markersize=5,
                ecolor='#aaa', elinewidth=0.8, label='plain mean')

    # Std-based shrinkage (blue, primary)
    _shrink_lo = np.maximum(0, np.array(shrink_means) - np.array(shrink_los))
    _shrink_hi = np.maximum(0, np.array(shrink_his) - np.array(shrink_means))
    ax.errorbar(ks, shrink_means,
                yerr=[_shrink_lo, _shrink_hi],
                fmt='o-', color='#2980b9', capsize=3, markersize=6,
                ecolor='#2980b9', elinewidth=1, label='std-shrinkage')

    # Annotate all points
    fmt_str = ".1f" if metric == "metaculus-score" else ".4f"
    for i in range(len(ks)):
        ax.annotate(f"{shrink_means[i]:{fmt_str}}",
                    (ks[i], shrink_means[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color='#2980b9')
        ax.annotate(f"{plain_means[i]:{fmt_str}}",
                    (ks[i], plain_means[i]), textcoords="offset points",
                    xytext=(8, -12), fontsize=8, color='#888')

    # Reference line: best single trial (max/min of individual trial scores)
    # Each k=1 subset is a single trial; compute mean score for each trial
    trial_mean_scores = []
    for t in common_trials:
        t_scores = [score_fn(tf[t], o) for o, tf in questions if t in tf]
        if t_scores:
            trial_mean_scores.append(float(np.mean(t_scores)))
    if trial_mean_scores:
        best_trial = max(trial_mean_scores) if higher else min(trial_mean_scores)
        fmt_ref = ".1f" if metric == "metaculus-score" else ".4f"
        ax.axhline(y=best_trial, color='#e74c3c', linestyle='--', alpha=0.6,
                   label=f'Best single trial: {best_trial:{fmt_ref}}')

    ax.set_xlabel("Number of Forecasts Averaged")
    ax.set_ylabel(label)
    ax.set_xticks(ks)
    ax.set_title(f"{label} vs Number of Trials \u2014 {config}\n"
                 f"({subtitle}; {xid_name})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_metric = metric.replace("-", "_")
    ntrials_dir = os.path.join(output_dir, "figs", "ntrials")
    out_path = os.path.join(ntrials_dir, f"{safe_metric}_vs_ntrials_{config}.png")
    os.makedirs(ntrials_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_metric_vs_questions(output_dir: str, config: str,
                                 scores: dict[tuple, dict],
                                 metric: str,
                                 xid_name: str = "") -> list[str]:
    """Per-question metric with cross-trial CI, sorted by mean score.

    Generates two versions:
      {metric}_vs_que_num_{config}.png  — numeric x-axis (no labels)
      {metric}_vs_que_str_{config}.png  — question text as x-axis labels

    Shows two horizontal reference lines:
      - E[single trial] = mean_q mean_trial score(q, trial)
      - E[averaged predictor] = mean_q score(q, mean_trial forecast(q, trial))

    Only works for multi-trial configs (trial_stats present).
    Returns list of output paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    higher = metric in HIGHER_IS_BETTER
    score_fn = SCORING_FUNCTIONS.get(metric)
    if not score_fn:
        return []

    # Collect per-question trial data
    q_data = []  # (q_text, mean_trial_score, [trial_scores], avg_predictor_score)
    for key, rec in scores.items():
        rt_outer = rec.get("raw_trials") or []
        trial_ps = rt_outer[0] if rt_outer and isinstance(rt_outer[0], list) else rt_outer
        if len(trial_ps) < 2:
            continue
        outcome = rec.get("outcome")
        if outcome is None:
            continue
        trial_scores = [score_fn(p, outcome) for p in trial_ps]
        if not trial_scores:
            continue
        # Score of the averaged predictor (mean forecast → single score)
        avg_p = float(np.mean(trial_ps))
        avg_score = score_fn(avg_p, outcome)
        q_text = rec.get("question", f"{key[0]}/{key[1]}")[:80]
        q_data.append((q_text, np.mean(trial_scores), trial_scores, avg_score, outcome))

    if len(q_data) < 2:
        return []

    # Sort by mean trial score (best first)
    q_data.sort(key=lambda x: x[1], reverse=higher)

    labels_q = [d[0] for d in q_data]
    means = [d[1] for d in q_data]
    avg_pred_scores = [d[3] for d in q_data]
    outcomes = [d[4] for d in q_data]

    # Bootstrap CI for each question
    rng = np.random.default_rng(42)
    los, his = [], []
    for _, _, t_scores, _, _ in q_data:
        arr = np.array(t_scores)
        n = len(arr)
        if n > 1:
            boot = rng.choice(arr, size=(500, n), replace=True).mean(axis=1)
            los.append(float(np.percentile(boot, 2.5)))
            his.append(float(np.percentile(boot, 97.5)))
        else:
            los.append(arr[0])
            his.append(arr[0])

    # Two reference lines
    K = len(q_data[0][2])
    single_trial_mean = float(np.mean(means))         # E[single trial]
    avg_pred_mean = float(np.mean(avg_pred_scores))    # E[averaged predictor]
    fmt_str = ".1f" if metric == "metaculus-score" else ".4f"

    n_q = len(q_data)
    safe_metric = metric.replace("-", "_")
    box_dir = os.path.join(output_dir, "figs", "metric_by_question_boxplots")
    os.makedirs(box_dir, exist_ok=True)
    out_paths = []

    # For errors version: filter to at-chance or below, plus bottom 5% of above-chance
    chance_threshold = 0 if higher else 0.25
    if higher:
        above_chance = [m for m in means if m > chance_threshold]
    else:
        above_chance = [m for m in means if m < chance_threshold]
    if above_chance:
        # Include bottom 5% of above-chance as "near chance"
        if higher:
            near_cutoff = np.percentile(above_chance, 5)  # 5th percentile (worst of good)
        else:
            near_cutoff = np.percentile(above_chance, 95)  # 95th percentile (worst of good)
    else:
        near_cutoff = chance_threshold

    for version in ("num", "str", "str_errors"):
        # Filter for errors version
        if version == "str_errors":
            if higher:
                mask = [i for i, m in enumerate(means) if m <= near_cutoff]
            else:
                mask = [i for i, m in enumerate(means) if m >= near_cutoff]
            if len(mask) < 2:
                continue
            v_labels = [labels_q[i] for i in mask]
            v_means = [means[i] for i in mask]
            v_los = [los[i] for i in mask]
            v_his = [his[i] for i in mask]
            v_qdata = [q_data[i] for i in mask]
            v_outcomes = [outcomes[i] for i in mask]
            n_v = len(mask)
            title_extra = f" (errors + near-chance; {n_v}/{n_q} questions)"
        else:
            v_labels = labels_q
            v_means = means
            v_los = los
            v_his = his
            v_qdata = q_data
            v_outcomes = outcomes
            n_v = n_q
            title_extra = ""

        if version == "num":
            fig_w = max(12, n_v * 0.15 + 2)
            fig_h = 5
        else:
            fig_w = max(14, n_v * 0.3 + 3)
            fig_h = 7

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        x = np.arange(n_v)
        # Color by outcome: green=YES(1), red=NO(0)
        colors = ['#27ae60' if o == 1 else '#c0392b' for o in v_outcomes]
        ax.scatter(x, v_means, s=10, c=colors, zorder=3)

        # Draw CI bars colored by outcome too
        for i, (m, lo_i, hi_i, c) in enumerate(zip(v_means, v_los, v_his, colors)):
            ax.plot([i, i], [lo_i, hi_i], color=c, linewidth=0.8, alpha=0.5)

        # Baseline
        ax.axhline(y=chance_threshold, color='gray', linestyle='--', alpha=0.4)

        # Two mean lines (from full data, not filtered)
        ax.axhline(y=single_trial_mean, color='#e67e22', linestyle='-', alpha=0.6,
                   label=f'E[single trial]={single_trial_mean:{fmt_str}}')
        ax.axhline(y=avg_pred_mean, color='#27ae60', linestyle='-', alpha=0.6,
                   label=f'E[avg of {K} trials]={avg_pred_mean:{fmt_str}}')
        # Outcome legend
        ax.scatter([], [], s=10, color='#27ae60', label='outcome=YES')
        ax.scatter([], [], s=10, color='#c0392b', label='outcome=NO')

        ax.set_ylabel(label)
        ax.set_title(f"{label} per Question — {config}{title_extra}\n"
                     f"({subtitle}; {xid_name})", fontsize=10)
        ax.legend(fontsize=8, loc='lower left' if higher else 'upper left')

        if version in ("str", "str_errors"):
            tick_fs = max(4, min(8, int(500 / max(n_v, 1))))
            ax.set_xticks(x)
            ax.set_xticklabels(v_labels, rotation=45, ha='right', fontsize=tick_fs)
        else:
            ax.set_xticks([])
            ax.set_xlabel(f"Question index (sorted by mean score; nquestions={n_q})")

        plt.tight_layout()
        out_path = os.path.join(box_dir, f"{safe_metric}_vs_que_{version}_{config}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def generate_heatmap(output_dir: str, eval_names: list[str],
                     all_scores: dict[str, dict[str, dict]],
                     metric: str, source: str | None = None,
                     xid_name: str = "") -> str | None:
    """Generate a heatmap PNG: rows=configs, columns=questions, color=metric value."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  WARNING: matplotlib not available, skipping heatmap")
        return None

    # Collect all question keys (for this source, or all sources)
    all_keys = set()
    for scores in all_scores.values():
        if source:
            all_keys.update(k for k in scores if k[0] == source)
        else:
            all_keys.update(scores.keys())
    if not all_keys:
        return None

    # Sort questions by mean metric across configs
    reverse = metric in HIGHER_IS_BETTER

    def mean_for_key(key):
        vals = []
        for c in eval_names:
            rec = all_scores.get(c, {}).get(key)
            if rec and metric in rec:
                v = rec[metric]
                if not (isinstance(v, float) and math.isnan(v)):
                    vals.append(v)
        return sum(vals) / len(vals) if vals else (1e9 if not reverse else -1e9)

    sorted_keys = sorted(all_keys, key=mean_for_key, reverse=reverse)

    configs = [c for c in eval_names
               if any(k in all_scores.get(c, {}) for k in all_keys)]
    if not configs:
        return None

    n_cols = len(sorted_keys)
    n_rows = len(configs)

    matrix = np.full((n_rows, n_cols), np.nan)
    for i, config in enumerate(configs):
        for j, key in enumerate(sorted_keys):
            sc = all_scores.get(config, {}).get(key)
            if sc and metric in sc:
                matrix[i, j] = sc[metric]

    # Figure sizing: cap cell width so very wide exams are still readable
    cell_w = max(0.05, min(0.35, 600 / max(n_cols, 1) / 72))
    fig_w = max(10, n_cols * cell_w + 3)
    fig_h = max(2, n_rows * 0.45 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    cmap = "RdYlBu_r" if metric not in HIGHER_IS_BETTER else "RdYlBu"

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(configs, fontsize=8)

    # Use question title (truncated to 120 chars) as x-axis labels
    q_labels = []
    for key in sorted_keys:
        title = None
        for c in eval_names:
            rec = all_scores.get(c, {}).get(key)
            if rec and rec.get("question"):
                title = rec["question"][:120]
                break
        q_labels.append(title or key[1])
    tick_fs = max(3, min(7, int(400 / max(n_cols, 1))))
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(q_labels, rotation=45, ha='right', fontsize=tick_fs)

    source_label = source or "overall"
    ax.set_title(f"{label} Heatmap \u2014 {source_label} ({xid_name})\n"
                 f"{subtitle}; columns sorted by mean score", fontsize=10)

    plt.tight_layout()
    safe_metric = metric.replace("-", "_")
    heatmap_dir = os.path.join(output_dir, "figs", "metric_by_question_heatmaps")
    out_path = os.path.join(heatmap_dir, f"{safe_metric}_heatmap_{source_label}.png")
    os.makedirs(heatmap_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Tool-call histogram
# ---------------------------------------------------------------------------

_TOOL_SHORT = {
    "web_search": "web_search",
    "summarize_results": "summarize",
    "lookup_url": "lookup_url",
    "submit": "submit",
    "fetch_ts_yfinance": "yfinance",
    "fetch_ts_fred": "fred",
    "fetch_ts_dbnomics": "dbnomics",
    "fetch_wikipedia_toc": "wiki_toc",
    "fetch_wikipedia_section": "wiki_section",
    "fetch_polymarket_info": "polymarket",
    "fetch_manifold_info": "manifold",
}


def _tool_short(name: str) -> str:
    return _TOOL_SHORT.get(name, name)


def generate_tool_histogram(output_dir: str, eval_names: list[str],
                            all_scores: dict[str, dict[str, dict]],
                            source_filter: str | None = None,
                            xid_name: str = "") -> str | None:
    """Grouped bar chart of tool-call counts per type per config."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    # Aggregate tool counts per config
    config_tool_totals = {}  # config -> {tool_short: count}
    all_tools = set()
    for config in eval_names:
        totals = {}
        for (src, _), rec in all_scores.get(config, {}).items():
            if source_filter and src != source_filter:
                continue
            for tool, cnt in rec.get("tool_counts", {}).items():
                short = _tool_short(tool)
                totals[short] = totals.get(short, 0) + cnt
                all_tools.add(short)
        if totals:
            config_tool_totals[config] = totals

    if not config_tool_totals or not all_tools:
        return None

    # Sort tools by total frequency (most common first)
    tool_freq = {}
    for totals in config_tool_totals.values():
        for t, c in totals.items():
            tool_freq[t] = tool_freq.get(t, 0) + c
    tools_sorted = sorted(all_tools, key=lambda t: tool_freq.get(t, 0), reverse=True)

    configs = [c for c in eval_names if c in config_tool_totals]
    n_tools = len(tools_sorted)
    n_configs = len(configs)
    x = np.arange(n_tools)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(max(8, n_tools * 1.2), 5))
    cmap = matplotlib.colormaps["tab10"]
    for i, config in enumerate(configs):
        vals = [config_tool_totals[config].get(t, 0) for t in tools_sorted]
        offset = (i - n_configs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=config, color=cmap(i % 10))
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(v), ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(tools_sorted, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Total calls")
    source_label = source_filter or "overall"
    ax.set_title(f"Tool calls by type \u2014 {source_label} ({xid_name})", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()

    histo_dir = os.path.join(output_dir, "figs", "tool_histograms")
    out_path = os.path.join(histo_dir, f"tool_histo_{source_label}.png")
    os.makedirs(histo_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Steps histogram
# ---------------------------------------------------------------------------

_STEP_BUCKETS = [
    (1, 1, "1"),
    (2, 4, "2\u20134"),
    (5, 7, "5\u20137"),
    (8, 10, "8\u201310"),
    (11, None, "11+"),
]


def _bucket_index(n_steps: int) -> int:
    for i, (lo, hi, _) in enumerate(_STEP_BUCKETS):
        if hi is None or n_steps <= hi:
            if n_steps >= lo:
                return i
    return len(_STEP_BUCKETS) - 1


def generate_steps_histogram(output_dir: str, eval_names: list[str],
                             all_scores: dict[str, dict[str, dict]],
                             source_filter: str | None = None,
                             xid_name: str = "") -> str | None:
    """Grouped bar chart of step-count distribution (bucketed) per config."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    bucket_labels = [b[2] for b in _STEP_BUCKETS]
    n_buckets = len(bucket_labels)

    config_buckets = {}
    for config in eval_names:
        counts = [0] * n_buckets
        for (src, _), rec in all_scores.get(config, {}).items():
            if source_filter and src != source_filter:
                continue
            counts[_bucket_index(rec.get("n_steps", 0))] += 1
        if any(c > 0 for c in counts):
            config_buckets[config] = counts

    if not config_buckets:
        return None

    configs = [c for c in eval_names if c in config_buckets]
    n_configs = len(configs)
    x = np.arange(n_buckets)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(max(7, n_buckets * 1.5), 5))
    cmap = matplotlib.colormaps["tab10"]
    for i, config in enumerate(configs):
        vals = config_buckets[config]
        offset = (i - n_configs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=config, color=cmap(i % 10))
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(v), ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=10)
    ax.set_xlabel("Number of agent steps")
    ax.set_ylabel("Number of questions")
    source_label = source_filter or "overall"
    ax.set_title(f"Agent steps distribution \u2014 {source_label} ({xid_name})", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()

    histo_dir = os.path.join(output_dir, "figs", "tool_histograms")
    out_path = os.path.join(histo_dir, f"steps_histo_{source_label}.png")
    os.makedirs(histo_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Steps heatmap
# ---------------------------------------------------------------------------

def generate_steps_heatmap(output_dir: str, eval_names: list[str],
                           all_scores: dict[str, dict[str, dict]],
                           source: str,
                           xid_name: str = "") -> str | None:
    """Heatmap of n_steps: rows=configs, columns=questions sorted by mean steps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    all_keys = set()
    for scores in all_scores.values():
        all_keys.update(k for k in scores if k[0] == source)
    if not all_keys:
        return None

    def mean_steps(key):
        vals = [all_scores[c][key].get("n_steps", 0) for c in eval_names
                if key in all_scores.get(c, {})]
        return sum(vals) / len(vals) if vals else 0

    sorted_keys = sorted(all_keys, key=mean_steps)

    configs = [c for c in eval_names
               if any(k in all_scores.get(c, {}) for k in all_keys)]
    if not configs:
        return None

    n_cols = len(sorted_keys)
    n_rows = len(configs)

    matrix = np.full((n_rows, n_cols), np.nan)
    for i, config in enumerate(configs):
        for j, key in enumerate(sorted_keys):
            sc = all_scores.get(config, {}).get(key)
            if sc:
                matrix[i, j] = sc.get("n_steps", 0)

    cell_w = max(0.05, min(0.35, 600 / max(n_cols, 1) / 72))
    fig_w = max(10, n_cols * cell_w + 3)
    fig_h = max(2, n_rows * 0.45 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="steps")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(configs, fontsize=8)

    q_labels = []
    for key in sorted_keys:
        title = None
        for c in eval_names:
            rec = all_scores.get(c, {}).get(key)
            if rec and rec.get("question"):
                title = rec["question"][:120]
                break
        q_labels.append(title or key[1])
    tick_fs = max(3, min(7, int(400 / max(n_cols, 1))))
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(q_labels, rotation=45, ha='right', fontsize=tick_fs)

    ax.set_title(f"Agent Steps Heatmap \u2014 {source} ({xid_name})\n"
                 f"columns sorted by mean steps", fontsize=10)

    plt.tight_layout()
    heatmap_dir = os.path.join(output_dir, "figs", "metric_by_question_heatmaps")
    out_path = os.path.join(heatmap_dir, f"steps_heatmap_{source}.png")
    os.makedirs(heatmap_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Metric vs. category (tag) plot
# ---------------------------------------------------------------------------

def generate_metric_vs_category(output_dir: str, config: str,
                                scores: dict[tuple, dict],
                                tags: dict[tuple, str],
                                metric: str,
                                xid_name: str = "") -> str | None:
    """Horizontal dot plot with bootstrap CI of metric per category for one config."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    # Group scores by category
    cat_vals = {}  # category -> list of metric values
    for key, rec in scores.items():
        cat = tags.get(key)
        if not cat:
            continue
        cat_vals.setdefault(cat, []).append(rec[metric])

    if not cat_vals:
        return None

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    reverse = metric in HIGHER_IS_BETTER

    # Bootstrap CI per category
    rows = []  # (category, mean, lo, hi, n)
    for cat, vals in cat_vals.items():
        mean, lo, hi = _bootstrap_ci(vals)
        rows.append((cat, mean, lo, hi, len(vals)))

    # Sort: best categories first (lowest brier / highest metaculus at top)
    rows.sort(key=lambda r: r[1], reverse=not reverse)

    categories = [r[0] for r in rows]
    means = [r[1] for r in rows]
    lo = [r[2] for r in rows]
    hi = [r[3] for r in rows]
    ns = [r[4] for r in rows]

    n_cats = len(categories)
    fig_h = max(4, n_cats * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y = np.arange(n_cats)
    xerr_lo = np.array(means) - np.array(lo)
    xerr_hi = np.array(hi) - np.array(means)
    ax.errorbar(means, y, xerr=[xerr_lo, xerr_hi],
                fmt='o', color='#2980b9', capsize=4, markersize=7,
                ecolor='#555', elinewidth=1.2)

    # Annotate with n and mean value
    for i, (m, n) in enumerate(zip(means, ns)):
        fmt = ".4f" if metric == "brier-score" else ".1f" if metric == "metaculus-score" else ".3f"
        ax.annotate(f"{m:{fmt}} (n={n})", (m, i),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=8, va='center', color='#333')

    # Category labels on y-axis
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=9)
    ax.invert_yaxis()  # best at top
    ax.set_xlabel(label)
    ax.set_title(f"{label} by Category \u2014 {config}\n({subtitle}; {xid_name})",
                 fontsize=10)

    # Baseline
    if metric == "brier-score":
        ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label="chance")
    elif metric in ("metaculus-score",):
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label="chance")
    elif metric in ("brier-index",):
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label="chance")

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    safe_metric = metric.replace("-", "_")
    cat_dir = os.path.join(output_dir, "figs", "tags")
    out_path = os.path.join(cat_dir, f"{safe_metric}_vs_category_{config}.png")
    os.makedirs(cat_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_metric_vs_category_composite(output_dir: str, eval_names: list[str],
                                          all_scores: dict[str, dict[str, dict]],
                                          tags: dict[tuple, str],
                                          metric: str,
                                          xid_name: str = "",
                                          tag_version: str = "xinghua",
                                          ref_names: list[str] | None = None,
                                          exam_name: str = "") -> str | None:
    """Composite category plot: K grouped horizontal bars per category, one per config.

    Categories sorted by mean score across all configs (best first for higher-is-better).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    higher = metric in HIGHER_IS_BETTER

    # Filter to configs with scores (skip sota etc.)
    configs = [c for c in eval_names if all_scores.get(c)]
    if not configs:
        return None

    # Group scores by category for each config
    # {config: {category: [metric_values]}}
    config_cat_vals = {c: {} for c in configs}
    cat_counts = {}  # {category: n_questions}
    for config in configs:
        for key, rec in all_scores[config].items():
            cat = tags.get(key)
            if not cat:
                continue
            if metric not in rec:
                continue
            v = rec[metric]
            if isinstance(v, float) and math.isnan(v):
                continue
            config_cat_vals[config].setdefault(cat, []).append(v)
            cat_counts[cat] = cat_counts.get(cat, 0)
    # Get accurate counts from first config with data
    for config in configs:
        for cat, vals in config_cat_vals[config].items():
            cat_counts[cat] = len(vals)
        break

    all_cats = set()
    for cv in config_cat_vals.values():
        all_cats.update(cv.keys())
    if not all_cats:
        return None

    # Compute mean per (config, category) and bootstrap CI
    cat_data = {}  # {cat: [(config, mean, lo, hi)]}
    for cat in all_cats:
        cat_data[cat] = []
        for config in configs:
            vals = config_cat_vals[config].get(cat, [])
            if vals:
                mean, lo, hi = _bootstrap_ci(np.array(vals))
                cat_data[cat].append((config, mean, lo, hi))
            else:
                cat_data[cat].append((config, None, None, None))

    # Add OVERALL pseudo-category (all questions combined)
    total_n = sum(cat_counts.values())
    cat_data["OVERALL"] = []
    for config in configs:
        all_vals = []
        for cv in config_cat_vals[config].values():
            all_vals.extend(cv)
        if all_vals:
            mean, lo, hi = _bootstrap_ci(np.array(all_vals))
            cat_data["OVERALL"].append((config, mean, lo, hi))
        else:
            cat_data["OVERALL"].append((config, None, None, None))
    cat_counts["OVERALL"] = total_n

    # Sort categories by overall mean (across all configs), keep OVERALL last
    def cat_sort_key(cat):
        means = [m for _, m, _, _ in cat_data[cat] if m is not None]
        return np.mean(means) if means else (-1e9 if higher else 1e9)
    sorted_cats = sorted(all_cats, key=cat_sort_key, reverse=not higher)
    sorted_cats.append("OVERALL")

    n_cats = len(sorted_cats)
    n_configs = len(configs)
    bar_height = 0.8 / n_configs

    fig_h = max(6, n_cats * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    cmap = matplotlib.colormaps["tab10"]
    y_base = np.arange(n_cats)

    # Separator line above OVERALL
    ax.axhline(y=n_cats - 1.5, color='#999', linewidth=0.8, linestyle='-')

    for ci, config in enumerate(configs):
        means_c, los_c, his_c = [], [], []
        for cat in sorted_cats:
            _, m, lo, hi = cat_data[cat][ci]
            means_c.append(m if m is not None else 0)
            los_c.append(lo if lo is not None else 0)
            his_c.append(hi if hi is not None else 0)

        y_pos = y_base + (ci - n_configs / 2 + 0.5) * bar_height
        xerr_lo = [m - l for m, l in zip(means_c, los_c)]
        xerr_hi = [h - m for m, h in zip(means_c, his_c)]

        display = _DISPLAY_NAMES.get(config, config)
        ax.errorbar(means_c, y_pos, xerr=[xerr_lo, xerr_hi],
                    fmt='o', color=cmap(ci % 10), capsize=3, markersize=5,
                    elinewidth=1, label=display)

    # Category labels with counts
    cat_labels = []
    for cat in sorted_cats:
        n = cat_counts.get(cat, '?')
        cat_labels.append(f"{'OVERALL' if cat == 'OVERALL' else cat} (n={n})")
    ax.set_yticks(y_base)
    ax.set_yticklabels(cat_labels, fontsize=8)
    for tick_label in ax.get_yticklabels():
        if tick_label.get_text().startswith("OVERALL"):
            tick_label.set_fontweight("bold")
    ax.invert_yaxis()

    # Baseline
    if metric in ("brier-score", "adjusted-brier-score"):
        ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5)
    elif metric in ("metaculus-score",):
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    elif metric in ("brier-index", "adjusted-brier-index"):
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # Reference markers (diamonds at per-group values)
    for ri, rn in enumerate(ref_names or []):
        ref = _lookup_reference(rn, exam_name, metric)
        if not ref:
            continue
        group_vals = ref[2] if len(ref) > 2 else {}
        overall_val = ref[0]
        ref_color = '#222'
        ref_marker = 'D'
        ref_ms = 5
        added_label = False
        for cat_i, cat in enumerate(sorted_cats):
            val = None
            if cat == "OVERALL":
                val = overall_val
            else:
                val = group_vals.get(cat)
            if val is not None:
                lbl = f"{rn}*" if not added_label else None
                ax.plot(val, cat_i, marker=ref_marker, color=ref_color,
                        markersize=ref_ms, zorder=10, label=lbl,
                        markeredgecolor='white', markeredgewidth=0.5)
                added_label = True

    ax.set_xlabel(label)
    ax.set_title(f"{label} by {tag_version}\n"
                 f"({subtitle}; {xid_name})", fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    safe_metric = metric.replace("-", "_")
    cat_dir = os.path.join(output_dir, "figs", "metric_by_tag", tag_version)
    out_path = os.path.join(cat_dir, f"{safe_metric}_composite.png")
    os.makedirs(cat_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Metric vs source (composite)
# ---------------------------------------------------------------------------

def generate_metric_vs_source_composite(output_dir: str, eval_names: list[str],
                                        all_scores: dict[str, dict[str, dict]],
                                        metric: str,
                                        xid_name: str = "") -> str | None:
    """Composite source plot: grouped horizontal bars per source, one per config.

    Similar to category composite but groups by question source instead of tag.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    higher = metric in HIGHER_IS_BETTER

    configs = [c for c in eval_names if all_scores.get(c)]
    if not configs:
        return None

    # Group scores by source for each config
    config_src_vals = {c: {} for c in configs}
    src_counts = {}
    for config in configs:
        for (source, qid), rec in all_scores[config].items():
            if metric not in rec:
                continue
            v = rec[metric]
            if isinstance(v, float) and math.isnan(v):
                continue
            config_src_vals[config].setdefault(source, []).append(v)
    # Counts from first config
    for config in configs:
        for src, vals in config_src_vals[config].items():
            src_counts[src] = len(vals)
        break

    all_sources = set()
    for cv in config_src_vals.values():
        all_sources.update(cv.keys())
    if not all_sources:
        return None

    # Compute mean per (config, source) with bootstrap CI
    src_data = {}
    for src in all_sources:
        src_data[src] = []
        for config in configs:
            vals = config_src_vals[config].get(src, [])
            if vals:
                mean, lo, hi = _bootstrap_ci(np.array(vals))
                src_data[src].append((config, mean, lo, hi))
            else:
                src_data[src].append((config, None, None, None))

    # Add "overall" pseudo-source (all questions combined)
    total_n = sum(src_counts.values())
    src_data["overall"] = []
    for config in configs:
        all_vals = []
        for src_vals in config_src_vals[config].values():
            all_vals.extend(src_vals)
        if all_vals:
            mean, lo, hi = _bootstrap_ci(np.array(all_vals))
            src_data["overall"].append((config, mean, lo, hi))
        else:
            src_data["overall"].append((config, None, None, None))
    src_counts["overall"] = total_n

    # Sort sources by overall mean (keep "overall" last)
    def src_sort_key(src):
        means = [m for _, m, _, _ in src_data[src] if m is not None]
        return np.mean(means) if means else (-1e9 if higher else 1e9)
    sorted_sources = sorted(all_sources, key=src_sort_key, reverse=not higher)
    sorted_sources.append("overall")

    n_src = len(sorted_sources)
    n_configs = len(configs)
    bar_height = 0.8 / n_configs

    fig_h = max(4, n_src * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    cmap = matplotlib.colormaps["tab10"]
    y_base = np.arange(n_src)

    # Draw separator line above "overall"
    ax.axhline(y=n_src - 1.5, color='#999', linewidth=0.8, linestyle='-')

    for ci, config in enumerate(configs):
        means_c, los_c, his_c = [], [], []
        for src in sorted_sources:
            _, m, lo, hi = src_data[src][ci]
            means_c.append(m if m is not None else 0)
            los_c.append(lo if lo is not None else 0)
            his_c.append(hi if hi is not None else 0)

        y_pos = y_base + (ci - n_configs / 2 + 0.5) * bar_height
        xerr_lo = [m - l for m, l in zip(means_c, los_c)]
        xerr_hi = [h - m for m, h in zip(means_c, his_c)]

        display = _DISPLAY_NAMES.get(config, config)
        ax.errorbar(means_c, y_pos, xerr=[xerr_lo, xerr_hi],
                    fmt='o', color=cmap(ci % 10), capsize=3, markersize=5,
                    elinewidth=1, label=display)

    src_labels = []
    for src in sorted_sources:
        n = src_counts.get(src, '?')
        if src == "overall":
            src_labels.append(f"OVERALL (n={n})")
        else:
            src_labels.append(f"{src} (n={n})")
    ax.set_yticks(y_base)
    ax.set_yticklabels(src_labels, fontsize=9)
    # Bold the "overall" label
    for tick_label in ax.get_yticklabels():
        if tick_label.get_text().startswith("OVERALL"):
            tick_label.set_fontweight("bold")
    ax.invert_yaxis()

    if metric in ("brier-score", "adjusted-brier-score"):
        ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5)
    elif metric in ("metaculus-score",):
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    elif metric in ("brier-index", "adjusted-brier-index"):
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel(label)
    ax.set_title(f"{label} by Source — all configs\n"
                 f"({subtitle}; {xid_name})", fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    safe_metric = metric.replace("-", "_")
    src_dir = os.path.join(output_dir, "figs", "metric_by_source")
    out_path = os.path.join(src_dir, f"{safe_metric}_vs_source_composite.png")
    os.makedirs(src_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Metric vs forecast horizon
# ---------------------------------------------------------------------------

_HORIZON_BINS = [
    (1, 14, "1–14d"),
    (15, 30, "15–30d"),
    (31, 45, "31–45d"),
    (46, 60, "46–60d"),
    (61, 9999, "61+d"),
]


def _compute_horizon(rec):
    """Compute forecast horizon in days from forecast_due_date to resolution_date."""
    fdd = rec.get("forecast_due_date", "")[:10]
    rd = rec.get("resolution_date", "")[:10]
    if not fdd or not rd:
        return None
    try:
        from datetime import datetime
        d1 = datetime.strptime(fdd, "%Y-%m-%d")
        d2 = datetime.strptime(rd, "%Y-%m-%d")
        return (d2 - d1).days
    except ValueError:
        return None


def _horizon_bin(days):
    """Map horizon days to a bin label."""
    if days is None:
        return None
    for lo, hi, label in _HORIZON_BINS:
        if lo <= days <= hi:
            return label
    return None


def generate_metric_vs_horizon(output_dir: str, eval_names: list[str],
                               all_scores: dict[str, dict[str, dict]],
                               metric: str,
                               xid_name: str = "") -> str | None:
    """Composite horizon plot: grouped horizontal bars per horizon bin, one per config.

    Similar to generate_metric_vs_category_composite but using forecast horizon bins.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    higher = metric in HIGHER_IS_BETTER

    configs = [c for c in eval_names if all_scores.get(c)]
    if not configs:
        return None

    # Group scores by horizon bin for each config
    config_bin_vals = {c: {} for c in configs}
    bin_counts = {}
    for config in configs:
        for key, rec in all_scores[config].items():
            if metric not in rec:
                continue
            v = rec[metric]
            if isinstance(v, float) and math.isnan(v):
                continue
            h = _compute_horizon(rec)
            b = _horizon_bin(h)
            if b is None:
                continue
            config_bin_vals[config].setdefault(b, []).append(v)
            bin_counts[b] = bin_counts.get(b, 0)
    # Get accurate counts from first config
    for config in configs:
        for b, vals in config_bin_vals[config].items():
            bin_counts[b] = len(vals)
        break

    all_bins = [label for _, _, label in _HORIZON_BINS if label in bin_counts]
    if not all_bins:
        return None

    # Compute mean + CI per (config, bin)
    bin_data = {}
    for b in all_bins:
        bin_data[b] = []
        for config in configs:
            vals = config_bin_vals[config].get(b, [])
            if vals:
                mean, lo, hi = _bootstrap_ci(np.array(vals))
                bin_data[b].append((config, mean, lo, hi))
            else:
                bin_data[b].append((config, None, None, None))

    n_bins = len(all_bins)
    n_configs = len(configs)
    bar_height = 0.8 / n_configs

    fig_h = max(4, n_bins * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    cmap = matplotlib.colormaps["tab10"]
    y_base = np.arange(n_bins)

    for ci, config in enumerate(configs):
        means_c, los_c, his_c = [], [], []
        for b in all_bins:
            _, m, lo, hi = bin_data[b][ci]
            means_c.append(m if m is not None else 0)
            los_c.append(lo if lo is not None else 0)
            his_c.append(hi if hi is not None else 0)

        y_pos = y_base + (ci - n_configs / 2 + 0.5) * bar_height
        xerr_lo = [m - l for m, l in zip(means_c, los_c)]
        xerr_hi = [h - m for m, h in zip(means_c, his_c)]

        ax.errorbar(means_c, y_pos, xerr=[xerr_lo, xerr_hi],
                    fmt='o', color=cmap(ci % 10), capsize=3, markersize=5,
                    elinewidth=1, label=config)

    bin_labels = [f"{b} (n={bin_counts.get(b, '?')})" for b in all_bins]
    ax.set_yticks(y_base)
    ax.set_yticklabels(bin_labels, fontsize=9)
    ax.invert_yaxis()

    if metric in ("brier-score", "adjusted-brier-score"):
        ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5)
    elif metric in ("metaculus-score",):
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    elif metric in ("brier-index", "adjusted-brier-index"):
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel(label)
    ax.set_title(f"{label} by Forecast Horizon — all configs\n"
                 f"({subtitle}; {xid_name})", fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    safe_metric = metric.replace("-", "_")
    horizon_dir = os.path.join(output_dir, "figs", "metric_by_horizon")
    out_path = os.path.join(horizon_dir, f"{safe_metric}_vs_horizon_composite.png")
    os.makedirs(horizon_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Tag distribution heatmap (moved to plot_exams.py)
# ---------------------------------------------------------------------------

# Re-export for backwards compatibility (called by make_exam.py)
from data.plot_exams import (generate_tag_distribution,      # noqa: F401
                        generate_rdate_by_fdate_scatter, # noqa: F401
                        generate_horizon_histogram)      # noqa: F401

# Canonical category orders (kept here for reference; authoritative copy in plot_exams.py)
_TAG_CATEGORIES = {
    "xinghua": [
        "Politics & Elections",
        "Economics & Macro Finance",
        "Geopolitics & International Affairs",
        "Crypto & Digital Assets",
        "Science & Technology",
        "AI & Artificial General Intelligence",
        "Companies & Business",
        "Culture & Entertainment",
        "Sports & Athletics",
        "Environment, Climate & Weather",
        "Health & Medicine",
        "Law & Justice",
        "Existential & Long-Horizon",
        "Forecasting Meta & Social Issues",
        "Miscellaneous & Novelty",
    ],
    "ben": [
        "Sports",
        "Security & Defense",
        "Arts & Recreation",
        "Science & Tech",
        "Economics & Business",
        "Politics & Governance",
        "Healthcare & Biology",
        "Environment & Energy",
        "Other",
    ],
}


# Original implementations moved to plot_exams.py.
# The re-exports above (generate_tag_distribution, generate_rdate_by_fdate_scatter,
# generate_horizon_histogram) ensure backwards compatibility.

_MOVED_TO_PLOT_EXAMS = True  # marker so grep can find this


def _generate_tag_distribution_REMOVED():
    """MOVED to plot_exams.py — this stub prevents accidental shadowing."""


# Calibration plots
# ---------------------------------------------------------------------------

def _compute_calibration(forecasts: list[float], outcomes: list[float],
                         n_bins: int = 10) -> tuple:
    """Compute calibration curve and ECE using quantile-based bins.

    Quantile bins ensure roughly equal samples per bin, giving more stable
    estimates than equal-width bins when forecasts cluster near 0 and 1.

    Returns (bin_midpoints, bin_accuracies, bin_counts, ece).
    """
    import numpy as np
    ps = np.array(forecasts)
    os_ = np.array(outcomes)
    order = np.argsort(ps)
    ps = ps[order]
    os_ = os_[order]

    # Split into n_bins quantile groups
    splits = np.array_split(np.arange(len(ps)), n_bins)
    midpoints = []
    accuracies = []
    counts = []
    ece = 0.0
    total = len(ps)

    for idx in splits:
        if len(idx) == 0:
            continue
        bin_p = float(np.mean(ps[idx]))
        bin_acc = float(np.mean(os_[idx]))
        midpoints.append(bin_p)
        accuracies.append(bin_acc)
        counts.append(len(idx))
        ece += abs(bin_acc - bin_p) * len(idx) / total

    return midpoints, accuracies, counts, float(ece)


def generate_calibration_curves(output_dir: str, eval_names: list[str],
                                all_scores: dict[str, dict[str, dict]],
                                xid_name: str = "") -> list[str]:
    """Reliability diagrams: combined, raw-only, and calibrated-only.

    Returns list of output paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    configs = [c for c in eval_names if c in all_scores and all_scores[c]]
    if not configs:
        return []

    cal_dir = os.path.join(output_dir, "figs", "calibration")
    os.makedirs(cal_dir, exist_ok=True)

    # Split into raw and calibrated
    raw_configs = [c for c in configs if not c.endswith("_calibrated")]
    cal_configs = [c for c in configs if c.endswith("_calibrated")]

    def _plot_one(config_list, title_suffix, filename):
        if not config_list:
            return None
        n = len(config_list)
        colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')

        for i, config in enumerate(config_list):
            scores = all_scores[config]
            forecasts = [s["forecast"] for s in scores.values()]
            outcomes = [s["outcome"] for s in scores.values()]
            midpoints, accuracies, counts, ece = _compute_calibration(
                forecasts, outcomes)
            display = _DISPLAY_NAMES.get(config, config)
            ax.plot(midpoints, accuracies, 'o-', color=colors[i], markersize=5,
                    label=f'{display} (ECE={ece:.3f})')

        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(f'Calibration{title_suffix} — {xid_name}', fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.2)
        plt.tight_layout()
        out_path = os.path.join(cal_dir, filename)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    paths = []
    # Combined (all configs)
    p = _plot_one(configs, "", "calibration_curves.png")
    if p:
        paths.append(p)
    # Raw only
    p = _plot_one(raw_configs, " (raw)", "calibration_curves_raw.png")
    if p:
        paths.append(p)
    # Calibrated only
    p = _plot_one(cal_configs, " (calibrated)", "calibration_curves_calibrated.png")
    if p:
        paths.append(p)

    return paths


def generate_ece_histogram(output_dir: str, eval_names: list[str],
                           all_scores: dict[str, dict[str, dict]],
                           xid_name: str = "") -> str | None:
    """Bar chart of ECE scores across configs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    configs = [c for c in eval_names if c in all_scores and all_scores[c]]
    if not configs:
        return None

    eces = []
    labels = []
    for config in configs:
        scores = all_scores[config]
        forecasts = [s["forecast"] for s in scores.values()]
        outcomes = [s["outcome"] for s in scores.values()]
        _, _, _, ece = _compute_calibration(forecasts, outcomes)
        eces.append(ece)
        labels.append(_DISPLAY_NAMES.get(config, config))

    # Sort by ECE (best first)
    order = np.argsort(eces)
    eces = [eces[i] for i in order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
    x = range(len(labels))
    bars = ax.bar(x, eces, color='#2980b9')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Expected Calibration Error (ECE)')
    ax.set_title(f'ECE by config — {xid_name}', fontsize=11)

    for bar, val in zip(bars, eces):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    cal_dir = os.path.join(output_dir, "figs", "calibration")
    os.makedirs(cal_dir, exist_ok=True)
    out_path = os.path.join(cal_dir, "ece_histo.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Metric by time (forecast due date)
# ---------------------------------------------------------------------------

def _get_knowledge_cutoff(config_name: str) -> str | None:
    """Extract knowledge cutoff date for the LLM used in a config."""
    cfg = load_results_config(config_name)
    if not cfg:
        return None
    llm = cfg.get("llm", "")
    from config.knowledge_cutoffs import KNOWLEDGE_CUTOFFS
    # Try exact match, then strip provider prefix
    if llm in KNOWLEDGE_CUTOFFS:
        return KNOWLEDGE_CUTOFFS[llm]
    # Strip "openrouter/" prefix
    short = "/".join(llm.split("/")[1:]) if llm.count("/") >= 2 else llm
    return KNOWLEDGE_CUTOFFS.get(short)


def generate_metric_by_time(output_dir: str, eval_names: list[str],
                            all_scores: dict[str, dict[str, dict]],
                            metric: str,
                            xid_name: str = "") -> str | None:
    """Plot metric vs forecast_due_date, one line per config.

    Includes vertical lines for knowledge cutoffs to detect leakage.
    Points are binned by biweekly forecast_due_date, showing mean ± CI.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        import pandas as pd
    except ImportError:
        return None

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))
    higher = metric in HIGHER_IS_BETTER

    configs = [c for c in eval_names if all_scores.get(c)]
    if not configs:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(configs), 10)))

    cutoffs_plotted = set()

    for ci, config in enumerate(configs):
        scores = all_scores[config]
        # Group by forecast_due_date
        date_vals = {}
        for key, rec in scores.items():
            fdd = rec.get("forecast_due_date", "")[:10]
            if not fdd or metric not in rec:
                continue
            v = rec[metric]
            if isinstance(v, float) and math.isnan(v):
                continue
            date_vals.setdefault(fdd, []).append(v)

        if not date_vals:
            continue

        dates = sorted(date_vals)
        means = [np.mean(date_vals[d]) for d in dates]
        x_dates = [pd.to_datetime(d) for d in dates]

        display = _DISPLAY_NAMES.get(config, config)
        ax.plot(x_dates, means, 'o-', color=cmap[ci], markersize=4,
                linewidth=1.5, alpha=0.8, label=display)

        # Knowledge cutoff vertical line
        cutoff = _get_knowledge_cutoff(config)
        if cutoff and cutoff not in cutoffs_plotted:
            cutoff_dt = pd.to_datetime(cutoff)
            if x_dates and x_dates[0] <= cutoff_dt <= x_dates[-1]:
                ax.axvline(x=cutoff_dt, color=cmap[ci], linestyle=':',
                           alpha=0.6, linewidth=1.5)
                ax.text(cutoff_dt, ax.get_ylim()[1], f" cutoff\n {display}",
                        fontsize=7, color=cmap[ci], va='top', ha='left')
                cutoffs_plotted.add(cutoff)

    ax.set_xlabel("Forecast due date")
    ax.set_ylabel(label)
    ax.set_title(f"{label} by Forecast Date — {xid_name}\n({subtitle})",
                 fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    safe_metric = metric.replace("-", "_")
    time_dir = os.path.join(output_dir, "figs", "metric_by_time")
    os.makedirs(time_dir, exist_ok=True)
    out_path = os.path.join(time_dir, f"{safe_metric}_by_time.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_metric_time_histos(output_dir: str, config: str,
                                scores: dict[str, dict],
                                metric: str,
                                xid_name: str = "") -> str | None:
    """Overlapping histograms of metric scores: all, first half, second half by date.

    Detects distribution shift between early and late questions.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError:
        return None

    label, subtitle = METRIC_LABELS.get(metric, (metric, ""))

    # Collect (date, score) pairs
    date_scores = []
    for key, rec in scores.items():
        fdd = rec.get("forecast_due_date", "")[:10]
        if not fdd or metric not in rec:
            continue
        v = rec[metric]
        if isinstance(v, float) and math.isnan(v):
            continue
        date_scores.append((fdd, v))

    if len(date_scores) < 4:
        return None

    date_scores.sort()
    all_vals = [v for _, v in date_scores]
    mid = len(date_scores) // 2
    first_half = [v for _, v in date_scores[:mid]]
    second_half = [v for _, v in date_scores[mid:]]
    first_end = date_scores[mid - 1][0]
    second_start = date_scores[mid][0]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = 20

    ax.hist(all_vals, bins=bins, alpha=0.3, color='gray',
            label=f"All (n={len(all_vals)}, mean={np.mean(all_vals):.3f})",
            edgecolor='gray', linewidth=0.5)
    ax.hist(first_half, bins=bins, alpha=0.4, color='blue',
            label=f"First half ≤{first_end} (n={len(first_half)}, mean={np.mean(first_half):.3f})",
            edgecolor='blue', linewidth=0.5)
    ax.hist(second_half, bins=bins, alpha=0.4, color='orange',
            label=f"Second half ≥{second_start} (n={len(second_half)}, mean={np.mean(second_half):.3f})",
            edgecolor='orange', linewidth=0.5)

    display = _DISPLAY_NAMES.get(config, config)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"{label} distribution — {display}\n"
                 f"({subtitle}; {xid_name})", fontsize=10)
    ax.legend(fontsize=7)

    plt.tight_layout()
    safe_metric = metric.replace("-", "_")
    time_dir = os.path.join(output_dir, "figs", "metric_by_time")
    os.makedirs(time_dir, exist_ok=True)
    out_path = os.path.join(time_dir, f"{safe_metric}_histos_{config}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
