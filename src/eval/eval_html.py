"""eval_html.py — HTML generation for leaderboard and dashboard pages."""

import os
import re

from config.config_display import config_struct, load_results_config, canonical_name
from core.eval import (
    METRIC_LABELS, HIGHER_IS_BETTER,
    _REFERENCE_NAMES, _lookup_reference, _esc, _compute_config_stats,
    compute_group_means,
)


def _config_suffix(name: str) -> str:
    """Extract display suffix from config name."""
    if name.endswith("_aggregated_calibrated"):
        return " +agg+cal"
    if name.endswith("_aggregated"):
        return " +agg"
    if name.endswith("_calibrated"):
        return " +cal"
    return ""


def generate_leaderboard(output_dir: str, eval_names: list[str],
                         all_scores: dict[str, dict[str, dict]],
                         metrics: list[str], groups: dict[str, list[str]],
                         xid_name: str = "",
                         exam_name: str = "",
                         ref_names: list[str] | None = None) -> str:
    """Generate leaderboard.html with group columns plus steps/cost/time columns."""
    group_names = list(groups.keys())

    # Drop redundant groups: if there's only one non-overall group and it
    # contains the same sources as overall, just show overall.
    non_overall = [g for g in group_names if g != "overall"]
    if len(non_overall) == 1 and "overall" in group_names:
        sole = non_overall[0]
        if set(groups.get(sole, [])) == set(groups.get("overall", [])):
            group_names = ["overall"]

    rows = []
    for config in eval_names:
        scores = all_scores.get(config, {})
        if not scores:
            continue
        row = {"name": config, "n": len(scores)}
        for metric in metrics:
            for gname in group_names:
                means = compute_group_means(scores, {gname: groups[gname]}, metric)
                row[f"{metric}_{gname}"] = means.get(gname)
        row["stats"] = _compute_config_stats(scores)
        rows.append(row)

    # Add reference rows
    for rn in (ref_names or []):
        row = {"name": rn, "n": "\u2014", "stats": {}, "is_ref": True}
        has_any = False
        for metric in metrics:
            ref = _lookup_reference(rn, exam_name, metric)
            if not ref:
                for gname in group_names:
                    row[f"{metric}_{gname}"] = None
                continue
            group_vals = ref[2] if len(ref) > 2 else {}
            for gname in group_names:
                val = group_vals.get(gname, ref[0])
                row[f"{metric}_{gname}"] = val
                has_any = True
        if has_any:
            rows.append(row)

    # Sort by first metric's last group (overall), descending for higher-is-better
    sort_key = f"{metrics[0]}_{group_names[-1]}"
    reverse = metrics[0] in HIGHER_IS_BETTER
    rows.sort(key=lambda r: (r.get(sort_key) or (-1e9 if reverse else 1e9)),
              reverse=reverse)

    # Build cost-breakdown modal data (JSON embedded in page)
    modal_data = {}
    for r in rows:
        st = r["stats"]
        if st:
            modal_data[r["name"]] = {
                "n": st.get("n", 0),
                "n_trials": st.get("n_trials", 1),
                "llm": st.get("llm", ""),
                "search_engine": st.get("search_engine", ""),
                "total_in": st.get("total_in", 0),
                "total_out": st.get("total_out", 0),
                "total_searches": st.get("total_searches", 0),
                "llm_cost": st.get("llm_cost"),
                "search_cost": st.get("search_cost"),
                "total_cost": st.get("total_cost"),
            }

    import json as _json
    modal_data_js = _json.dumps(modal_data)

    col_idx = 2  # after # and Config

    parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Leaderboard — {_esc(xid_name)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; max-width: 1400px; margin: 0 auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f5f5f5; }}
td.num {{ text-align: right; }}
a {{ color: #2980b9; }}
th.group {{ text-align: center; border-bottom: 2px solid #999; }}
th.sortable {{ cursor: pointer; user-select: none; }}
th.sortable:hover {{ background: #e8e8e8; }}
.cost-link {{ color: #2980b9; text-decoration: underline; cursor: pointer; }}
/* Modal */
#modal-overlay {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%;
  background:rgba(0,0,0,0.4); z-index:1000; align-items:center; justify-content:center; }}
#modal-overlay.open {{ display:flex; }}
#modal-box {{ background:#fff; border-radius:8px; padding:28px 32px; max-width:600px; width:90%;
  box-shadow:0 8px 32px rgba(0,0,0,0.2); position:relative; }}
#modal-close {{ position:absolute; top:12px; right:16px; font-size:20px; cursor:pointer;
  color:#888; background:none; border:none; }}
#modal-close:hover {{ color:#333; }}
#modal-box h2 {{ margin:0 0 4px; font-size:1.1em; }}
#modal-box p {{ margin:0 0 14px; color:#666; font-size:0.9em; }}
#modal-box table {{ font-size:0.9em; }}
#modal-box td, #modal-box th {{ padding:5px 10px; }}
#modal-box tfoot td {{ font-weight:bold; border-top:2px solid #999; }}
</style>
<script>
var COST_DATA = {modal_data_js};

function sortTable(el, colIdx, numeric) {{
  var table = document.getElementById('leaderboard');
  var tbody = table.tBodies[0];
  var rows = Array.from(tbody.rows);
  var asc = el.dataset.sortDir !== 'asc';
  document.querySelectorAll('#leaderboard .sort-arrow').forEach(function(s) {{ s.textContent = ' \\u21C5'; }});
  el.querySelector('.sort-arrow').textContent = asc ? ' \\u25B2' : ' \\u25BC';
  el.dataset.sortDir = asc ? 'asc' : 'desc';
  rows.sort(function(a, b) {{
    var av = a.cells[colIdx].dataset.val !== undefined ? a.cells[colIdx].dataset.val : a.cells[colIdx].textContent.replace(/[^0-9.\\-]/g,'');
    var bv = b.cells[colIdx].dataset.val !== undefined ? b.cells[colIdx].dataset.val : b.cells[colIdx].textContent.replace(/[^0-9.\\-]/g,'');
    if (numeric) {{
      av = av === '' ? (asc ? 1e9 : -1e9) : parseFloat(av);
      bv = bv === '' ? (asc ? 1e9 : -1e9) : parseFloat(bv);
      return asc ? av - bv : bv - av;
    }}
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(function(row, i) {{ row.cells[0].textContent = i + 1; tbody.appendChild(row); }});
}}

function showModal(config) {{
  var d = COST_DATA[config];
  if (!d) return;
  var n = d.n;
  var rows = '';
  var fmt = function(x) {{ return x != null ? '$' + x.toFixed(2) : '\u2014'; }};
  var fmtN = function(x) {{ return x != null ? x.toLocaleString() : '\u2014'; }};
  if (d.llm) {{
    rows += '<tr><td>LLM</td><td>' + d.llm + '</td><td>' + fmtN(d.total_in) + '</td><td>' + fmtN(d.total_out) + '</td><td>' + fmt(d.llm_cost) + '</td></tr>';
  }}
  if (d.search_engine) {{
    rows += '<tr><td>Search API</td><td>' + d.search_engine + '</td><td colspan="2">' + fmtN(d.total_searches) + ' queries</td><td>' + fmt(d.search_cost) + '</td></tr>';
  }}
  document.getElementById('modal-title').textContent = config + ' \u2014 cost breakdown';
  var nt = d.n_trials || 1;
  var sub = 'Across ' + n + ' question(s)';
  if (nt > 1) sub += ', ' + nt + ' trials per question';
  document.getElementById('modal-subtitle').textContent = sub;
  document.getElementById('modal-rows').innerHTML = rows;
  document.getElementById('modal-total').textContent = fmt(d.total_cost);
  document.getElementById('modal-overlay').classList.add('open');
}}

function closeModal() {{
  document.getElementById('modal-overlay').classList.remove('open');
}}
</script>
</head><body>

<div id="modal-overlay" onclick="if(event.target===this)closeModal()">
  <div id="modal-box">
    <button id="modal-close" onclick="closeModal()">&#x2715;</button>
    <h2 id="modal-title"></h2>
    <p id="modal-subtitle"></p>
    <table style="width:100%">
      <thead><tr><th>Stage</th><th>Model</th><th>Tokens in</th><th>Tokens out</th><th>Cost ($)</th></tr></thead>
      <tbody id="modal-rows"></tbody>
      <tfoot><tr><td colspan="4">Total</td><td id="modal-total"></td></tr></tfoot>
    </table>
  </div>
</div>

<h1>Leaderboard</h1>
<p>XID: {_esc(xid_name)} | <a href="dashboard.html">Per-question details</a></p>
<p style="font-size:11px;color:#666;margin:4px 0 8px">
Config notation: <b>think/search/c<i>crowd</i>/t<i>tools</i></b> [/n=<i>trials</i>]
&nbsp; Suffixes: <b>+cal</b> = Platt-calibrated.
&nbsp; Click config name for full config JSON.
</p>
<table id="leaderboard">
<thead>
<tr>
<th rowspan="2">#</th><th rowspan="2">Config</th><th rowspan="2" class="sortable num" onclick="sortTable(this,2,false)">N<span class="sort-arrow"> &#x21C5;</span></th>"""]

    col_idx = 3  # after #, Config, N

    # First header row: metric group headers
    for metric in metrics:
        label = METRIC_LABELS.get(metric, (metric, ""))[0]
        parts.append(f'<th class="group" colspan="{len(group_names)}">{_esc(label)}</th>')
    # Extra stat columns span header
    stat_start = col_idx + len(metrics) * len(group_names)
    parts.append(f'<th rowspan="2" class="sortable num" onclick="sortTable(this,'
                 f'{stat_start},true)">Avg Steps'
                 '<span class="sort-arrow"> &#x21C5;</span></th>')
    parts.append(f'<th rowspan="2" class="sortable num" onclick="sortTable(this,'
                 f'{stat_start + 1},true)">Timeouts'
                 '<span class="sort-arrow"> &#x21C5;</span></th>')
    parts.append(f'<th rowspan="2" class="sortable num" onclick="sortTable(this,'
                 f'{stat_start + 2},true)">Cost ($)'
                 '<span class="sort-arrow"> &#x21C5;</span></th>')
    parts.append(f'<th rowspan="2" class="sortable num" onclick="sortTable(this,'
                 f'{stat_start + 3},true)">Time/q (s)'
                 '<span class="sort-arrow"> &#x21C5;</span></th>')
    parts.append("</tr><tr>")

    # Compute per-group sample sizes from the first config with scores
    group_counts = {}
    for scores in all_scores.values():
        if scores:
            for gname in group_names:
                sources = groups.get(gname, [])
                group_counts[gname] = sum(
                    1 for key in scores if key[0] in sources)
            break

    # Second header row: group names with sample sizes (sortable)
    for metric in metrics:
        for gname in group_names:
            n = group_counts.get(gname)
            n_str = f"<br><span style='font-weight:normal;font-size:0.85em'>n={n}</span>" if n else ""
            parts.append(f'<th class="sortable" onclick="sortTable(this,{col_idx},true)">'
                         f'{_esc(gname)}{n_str}<span class="sort-arrow"> &#x21C5;</span></th>')
            col_idx += 1
    parts.append("</tr></thead><tbody>")

    for i, r in enumerate(rows, 1):
        is_ref = r.get("is_ref", False)
        st = r.get("stats", {})
        avg_steps = st.get("avg_steps")
        cost_per_q = st.get("cost_per_q")
        total_cost = st.get("total_cost")
        avg_elapsed = st.get("avg_elapsed")
        n_timeouts = st.get("n_timeouts", 0)
        n_questions = st.get("n", 0)

        # Omit cost/steps/time for calibrated/bon variants (same underlying runs)
        is_derivative = r["name"].endswith(("_calibrated",))
        steps_str = f"{avg_steps:.1f}" if avg_steps is not None and not is_derivative else "\u2014"
        elapsed_str = f"{avg_elapsed:.1f}" if avg_elapsed and not is_derivative else "\u2014"
        timeout_str = (f"{n_timeouts}/{n_questions}" if n_timeouts and not is_derivative
                       else "\u2014" if is_derivative else "0")

        if total_cost is not None and not is_derivative:
            safe_name = r["name"].replace("'", "\\'")
            cost_str = f"<span class=\"cost-link\" onclick=\"showModal('{safe_name}')\">${total_cost:.2f}</span>"
        else:
            cost_str = "\u2014"

        row_style = ' style="background:#f0f8ff;font-style:italic"' if is_ref else ""
        if is_ref:
            disp = f'{r["name"]}*'
            name_cell = f'<td><em>{_esc(disp)} (ref)</em></td>'
        else:
            base_name = (r["name"].removesuffix("_calibrated")
                         .removesuffix("_aggregated"))
            cfg = load_results_config(r["name"])
            if cfg:
                cs = config_struct(cfg)
                sub_parts = []
                if cs["ntrials"] > 1:
                    sub_parts.append(f'n={cs["ntrials"]}')
                suffix = _config_suffix(r["name"])
                detail = (f'{cs["think"]}/{cs["search"]}/c{cs["crowd"]}/t{cs["tools"]}'
                          f'{"/" + "/".join(sub_parts) if sub_parts else ""}'
                          f'{suffix}')
                display = f'{cs["model"]} <span style="font-size:0.85em">{detail}</span>'
            else:
                display = _esc(r["name"])
            name_cell = (f'<td><a href="../../../experiments/forecasts_raw/'
                         f'{base_name}/config.json" style="text-decoration:none">'
                         f'{display}</a></td>')
        parts.append(f'<tr{row_style}><td>{i}</td>'
                     f'{name_cell}'
                     f'<td class="num">{r["n"]}</td>')
        for metric in metrics:
            fmt = ".4f" if metric == "brier-score" else ".1f" if metric == "metaculus-score" else ".3f"
            for gname in group_names:
                v = r.get(f"{metric}_{gname}")
                parts.append(f'<td class="num">{f"{v:{fmt}}" if v is not None else "\u2014"}</td>')

        cost_val = f"{total_cost:.4f}" if total_cost is not None else ""
        parts.append(f'<td class="num">{steps_str}</td>'
                     f'<td class="num">{timeout_str}</td>'
                     f'<td class="num" data-val="{cost_val}">{cost_str}</td>'
                     f'<td class="num">{elapsed_str}</td>'
                     f'</tr>')

    parts.append("</tbody></table></body></html>")

    out_path = os.path.join(output_dir, "leaderboard.html")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    return out_path


# ---------------------------------------------------------------------------
# Details HTML (dashboard)
# ---------------------------------------------------------------------------

def _brier_color(bs: float) -> str:
    if bs < 0.05:
        return "#c6efce"
    if bs < 0.15:
        return "#e2efda"
    if bs < 0.50:
        return "#fce4d6"
    return "#ffc7ce"


def generate_details(output_dir: str, eval_names: list[str],
                     all_scores: dict[str, dict[str, dict]],
                     xid_name: str = "") -> str:
    """Generate dashboard.html with one row per question, columns per config.

    Calibrated variants are merged into the base config's cell rather than
    getting their own columns (they share the same trace page).
    """
    # Filter out configs with no scored questions, and virtual configs (no trace files)
    eval_names = [c for c in eval_names
                  if all_scores.get(c) and "[" not in c]

    # Derivative scores are shown inline in the base config's cell.
    base_names = []
    cal_names = set()
    for c in eval_names:
        if c.endswith("_calibrated"):
            cal_names.add(c.removesuffix("_calibrated"))
        else:
            base_names.append(c)

    # Collect all question keys
    all_keys = set()
    for scores in all_scores.values():
        all_keys.update(scores.keys())
    all_keys = sorted(all_keys)

    # Get metadata from first available config
    key_meta = {}
    for key in all_keys:
        for cname in eval_names:
            if key in all_scores.get(cname, {}):
                key_meta[key] = all_scores[cname][key]
                break

    # Build suffix legend
    suffix_parts = []
    if cal_names:
        suffix_parts.append("<b>C</b> = calibrated")
    suffix_legend = (f" &nbsp; {', '.join(suffix_parts)}." if suffix_parts else "")

    parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Details — {_esc(xid_name)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f5f5f5; position: sticky; top: 0; z-index: 1; }}
td.brier {{ text-align: center; }}
td.outcome {{ text-align: center; font-weight: bold; }}
a {{ color: #2980b9; }}
.cal {{ color: #888; font-size: 11px; }}
</style></head><body>
<h1>Forecast Details</h1>
<p>XID: {_esc(xid_name)} | {len(all_keys)} questions | {len(base_names)} config(s)
| <a href="leaderboard.html">Leaderboard</a></p>
<p style="font-size:12px;color:#555">
Brier score color key:
<span style="background:#c6efce;padding:2px 7px;border-radius:3px">&lt;0.05</span>
<span style="background:#e2efda;padding:2px 7px;border-radius:3px">0.05\u20130.15</span>
<span style="background:#fce4d6;padding:2px 7px;border-radius:3px">0.15\u20130.50</span>
<span style="background:#ffc7ce;padding:2px 7px;border-radius:3px">\u22650.50</span>
&nbsp; Cell value = forecast probability. * = timed out.{suffix_legend}
</p>
<p style="font-size:11px;color:#666;margin:4px 0 8px">
Config notation: <b>model-think-search-c<i>crowd</i>-t<i>tools</i></b>[-n<i>trials</i>]
</p>
<table>
<tr><th>#</th><th>Source</th><th>Question</th><th>Forecast date</th><th>Res date</th><th>Outcome</th>"""]

    for cname in base_names:
        cfg = load_results_config(cname)
        if cfg:
            cname_display = canonical_name(cfg)
            suffix = _config_suffix(cname)
            parts.append(f'<th>{_esc(cname_display)}{suffix}</th>')
        else:
            parts.append(f'<th>{_esc(cname)}</th>')
    parts.append("</tr>")

    for i, key in enumerate(all_keys, 1):
        source, qid = key
        meta = key_meta.get(key, {})
        question_full = meta.get("question", "")
        question_short = question_full[:80]
        if len(question_full) > 80:
            q_td = (f'<td><details><summary>{_esc(question_short)}...</summary>'
                    f'{_esc(question_full)}</details></td>')
        else:
            q_td = f'<td>{_esc(question_full)}</td>'
        fdd = meta.get("forecast_due_date", "")
        url = meta.get("url", "")
        rdates = meta.get("resolution_dates", [])
        rdate_single = meta.get("resolution_date", "")
        resolved_to = meta.get("resolved_to")
        outcome = meta.get("outcome")

        # Build resolution date + outcome display
        if rdates and isinstance(rdates, list) and len(rdates) > 1:
            # Multi-resolution: show each date with horizon and outcome
            import pandas as _pd
            rdate_lines = []
            outcome_lines = []
            fdd_dt = _pd.to_datetime(fdd) if fdd else None
            outcomes_list = resolved_to if isinstance(resolved_to, list) else [resolved_to]
            for j, rd in enumerate(rdates):
                horizon = ""
                if fdd_dt:
                    try:
                        delta = (_pd.to_datetime(rd) - fdd_dt).days
                        horizon = f' <span style="color:#999;font-size:0.8em">\u0394={delta}d</span>'
                    except Exception:
                        pass
                rdate_lines.append(f"{rd}{horizon}")
                o_val = outcomes_list[j] if j < len(outcomes_list) else None
                if o_val is not None:
                    o_color = "#27ae60" if float(o_val) >= 0.5 else "#e74c3c"
                    outcome_lines.append(
                        f'<span style="color:{o_color}">{float(o_val):.3f}</span>')
                else:
                    outcome_lines.append("?")
            rdate_td = "<br>".join(rdate_lines)
            outcome_td = f'<td class="outcome">{"<br>".join(outcome_lines)}</td>'
        else:
            rdate_td = _esc(rdate_single or (rdates[0] if rdates else ""))
            outcome_str = f"{outcome:.0f}" if outcome is not None else "?"
            outcome_td = f'<td class="outcome">{outcome_str}</td>'

        source_display = f'<a href="{_esc(url)}" target="_blank">{_esc(source)}</a>' if url else _esc(source)

        parts.append(
            f'<tr><td>{i}</td>'
            f'<td>{source_display}</td>'
            f'{q_td}'
            f'<td>{_esc(fdd)}</td><td>{rdate_td}</td>{outcome_td}')

        for cname in base_names:
            sc = all_scores.get(cname, {}).get(key)
            if sc:
                # Multi-resolution: show per-date forecasts
                multi_ps = sc.get("forecasts")
                if multi_ps and isinstance(multi_ps, list) and len(multi_ps) > 1:
                    cell_lines = []
                    outcomes_list = (sc.get("resolved_to")
                                     if isinstance(sc.get("resolved_to"), list)
                                     else [sc.get("outcome")])
                    for j, pj in enumerate(multi_ps):
                        oj = outcomes_list[j] if j < len(outcomes_list) else None
                        if pj is not None and oj is not None:
                            bsj = (float(pj) - float(oj)) ** 2
                            bg_j = _brier_color(bsj)
                            cell_lines.append(
                                f'<span style="background:{bg_j};padding:1px 3px;'
                                f'border-radius:2px">{float(pj):.3f}</span>')
                        elif pj is not None:
                            cell_lines.append(f'{float(pj):.3f}')
                    base_cname = (cname.removesuffix("_calibrated")
                                  .removesuffix("_aggregated"))
                    detail_path = (f"../../../experiments/forecasts_raw/{base_cname}/{source}"
                                   f"/{re.sub(r'[/\\\\:]', '_', str(qid))}_trace.html")
                    cell = f'<a href="{detail_path}" style="text-decoration:none">{"<br>".join(cell_lines)}</a>'
                    parts.append(f'<td class="brier" style="font-size:0.9em">{cell}</td>')
                    continue

                p = sc["forecast"]
                bs = sc.get("brier-score", (p - (outcome or 0)) ** 2)
                bg = _brier_color(bs)
                timeout_mark = "*" if not sc.get("submitted", True) else ""
                # Link to detail page
                base_cname = (cname.removesuffix("_calibrated")
                              .removesuffix("_aggregated"))
                detail_path = (f"../../../experiments/forecasts_raw/{base_cname}/{source}"
                               f"/{re.sub(r'[/\\\\:]', '_', str(qid))}_trace.html")
                cell = f'<a href="{detail_path}">{p:.3f}{timeout_mark}</a>'
                # Append calibrated inline
                extras = []
                if cname in cal_names:
                    cal_sc = all_scores.get(f"{cname}_calibrated", {}).get(key)
                    if cal_sc:
                        extras.append(f'{cal_sc["forecast"]:.3f}C')
                if extras:
                    cell += f' <span class="cal">({", ".join(extras)})</span>'
                parts.append(f'<td class="brier" style="background:{bg};">{cell}</td>')
            else:
                parts.append('<td>\u2014</td>')

        parts.append("</tr>")

    parts.append("</table></body></html>")

    out_path = os.path.join(output_dir, "dashboard.html")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    return out_path
