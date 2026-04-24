"""eval_trace.py — Per-question detail/trace HTML pages."""

import json
import math
import os
import re

from core.eval import _resolve_outcome, _esc, forecast_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_detail_pages(config_name: str, exam: dict[str, list[str]]):
    """Generate per-question HTML detail pages for a config.

    For single-trial: renders experiments/forecasts_raw/{config}/{source}/{id}_trace.html
    For multi-trial: renders per-trial traces and a summary page linking them.
    """
    import glob as _glob

    # Discover trials
    trial_dirs = sorted(_glob.glob(
        os.path.join("experiments", "forecasts_raw", config_name, "trial_*")))
    trial_nums = []
    for d in trial_dirs:
        m = re.match(r'.*trial_(\d+)$', d)
        if m:
            trial_nums.append(int(m.group(1)))

    n = 0
    for source, ids in exam.items():
        for qid in ids:
            safe_id = re.sub(r'[/\\:]', '_', str(qid))

            if trial_nums:
                # Multi-trial: render each trial trace, then summary
                trial_fcs = []
                for t in trial_nums:
                    trial_path = os.path.join("experiments", "forecasts_raw", config_name,
                                              f"trial_{t}", source, f"{safe_id}.json")
                    if not os.path.exists(trial_path):
                        continue
                    with open(trial_path) as f:
                        tfc = json.load(f)
                    if _resolve_outcome(tfc.get("resolved_to")) is None:
                        continue
                    trial_dir = os.path.join("experiments", "forecasts_raw", config_name,
                                            f"trial_{t}", source)
                    _render_detail_page(tfc, f"{config_name}/trial_{t}",
                                       source, safe_id, trial_dir)
                    trial_fcs.append((t, tfc))

                if not trial_fcs:
                    # Fall back to single-trial mode for this question
                    # (happens when trial dirs exist for other sources but not this one)
                    base_path = os.path.join("experiments", "forecasts_raw", config_name,
                                             source, f"{safe_id}.json")
                    if os.path.exists(base_path):
                        with open(base_path) as f:
                            fc = json.load(f)
                        source_dir = os.path.join("experiments", "forecasts_raw",
                                                   config_name, source)
                        _render_detail_page(fc, config_name, source, safe_id, source_dir)
                        n += 1
                    continue

                # Load averaged forecast for the summary
                avg_path = forecast_path(config_name, source, qid)
                if os.path.exists(avg_path):
                    with open(avg_path) as f:
                        avg_fc = json.load(f)
                else:
                    avg_fc = trial_fcs[0][1]

                source_dir = os.path.join("experiments", "forecasts_raw", config_name, source)
                _render_trial_summary_page(avg_fc, trial_fcs, config_name,
                                           source, safe_id, source_dir)
                n += 1
            else:
                # Single trial: standard trace page
                path = forecast_path(config_name, source, qid)
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    fc = json.load(f)
                if _resolve_outcome(fc.get("resolved_to")) is None:
                    continue
                # Slim forecasts (from forecasts_final/) lack belief_history
                # and tool_log — skip trace generation since there's nothing
                # to visualize.
                if not fc.get("belief_history") and not fc.get("tool_log"):
                    continue
                source_dir = os.path.join("experiments", "forecasts_raw", config_name, source)
                os.makedirs(source_dir, exist_ok=True)
                _render_detail_page(fc, config_name, source, safe_id, source_dir)
                n += 1

    if n:
        suffix = f" ({len(trial_nums)} trials each)" if trial_nums else ""
        print(f"  [{config_name}] Generated {n} detail pages{suffix}")


# ---------------------------------------------------------------------------
# Belief SVG
# ---------------------------------------------------------------------------

def _belief_svg(belief_history: list) -> str:
    """Render belief evolution as an inline SVG line chart with x-axis ticks."""
    pts = [b.get("p", 0.5) for b in belief_history]
    if not pts:
        return ""
    W, H, PAD_L, PAD_R, PAD_T, PAD_B = 320, 150, 28, 16, 16, 28
    inner_w = W - PAD_L - PAD_R
    inner_h = H - PAD_T - PAD_B
    n = len(pts)

    def cx(i):
        return PAD_L + (i / max(n - 1, 1)) * inner_w if n > 1 else W / 2

    def cy(p):
        return PAD_T + (1 - p) * inner_h

    # Polyline
    points = " ".join(f"{cx(i):.1f},{cy(p):.1f}" for i, p in enumerate(pts))
    # Y-axis gridlines at 0, 0.25, 0.5, 0.75, 1.0
    gridlines = ""
    for gp in [0, 0.25, 0.5, 0.75, 1.0]:
        y = cy(gp)
        dash = 'stroke-dasharray="3,3"' if gp != 0.5 else 'stroke-dasharray="6,3"'
        gridlines += (f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                      f'stroke="#ccc" stroke-width="1" {dash}/>'
                      f'<text x="{PAD_L-4}" y="{y+4:.1f}" font-size="9" text-anchor="end" fill="#999">'
                      f'{gp:.2f}</text>')
    # Dots and x-axis ticks
    dots = ""
    tick_interval = 1 if n <= 12 else 2
    for i, p in enumerate(pts):
        dots += (f'<circle cx="{cx(i):.1f}" cy="{cy(p):.1f}" r="4" fill="#2980b9" stroke="white" stroke-width="1.5"/>')
        if i % tick_interval == 0 or i == n - 1:
            label = "init" if i == 0 else str(i)
            dots += (f'<line x1="{cx(i):.1f}" y1="{H-PAD_B}" x2="{cx(i):.1f}" y2="{H-PAD_B+4}" '
                     f'stroke="#999" stroke-width="1"/>'
                     f'<text x="{cx(i):.1f}" y="{H-PAD_B+14}" font-size="8" text-anchor="middle" '
                     f'fill="#666">{label}</text>')

    return (f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" style="display:block">'
            f'{gridlines}'
            f'<polyline points="{points}" fill="none" stroke="#2980b9" stroke-width="2"/>'
            f'{dots}'
            f'</svg>')


# ---------------------------------------------------------------------------
# Single-trial detail page
# ---------------------------------------------------------------------------

def _render_detail_page(fc: dict, config_name: str, source: str,
                        safe_id: str, source_dir: str):
    """Write a rich per-question detail page to {source_dir}/{safe_id}_trace.html."""
    question = fc.get("question", "")
    background = fc.get("background", "")
    resolution_criteria = fc.get("resolution_criteria", "")
    forecast_due_date = fc.get("forecast_due_date", "")
    resolution_date = fc.get("resolution_date", "")
    market_value = fc.get("market_value")
    market_date = fc.get("market_date", "")
    market_explanation = fc.get("market_value_explanation", "")
    reasoning = fc.get("reasoning", "")
    system_prompt = fc.get("system_prompt", "")
    question_prompt = fc.get("question_prompt", "")
    forecast_p = fc.get("forecast", 0.5)
    outcome = _resolve_outcome(fc.get("resolved_to"))
    belief_history = fc.get("belief_history", [])
    tool_log = fc.get("tool_log", [])
    n_steps = fc.get("n_steps", 0)
    tokens_in = fc.get("tokens_in", 0)
    tokens_out = fc.get("tokens_out", 0)
    elapsed = fc.get("elapsed_seconds", 0)
    config_info = fc.get("config", {})

    brier = (forecast_p - outcome) ** 2 if outcome is not None else None
    metaculus = (100 * (1 + math.log2(max(0.001, min(0.999, forecast_p)) if outcome == 1
                                     else max(0.001, min(0.999, 1 - forecast_p))))
                 if outcome is not None else None)
    outcome_label = f"YES (1)" if outcome == 1 else f"NO (0)" if outcome == 0 else "?"
    correct_color = ("#27ae60" if (outcome == 1 and forecast_p > 0.5) or
                                  (outcome == 0 and forecast_p < 0.5)
                    else "#c0392b" if outcome is not None else "#888")

    # Build belief_history index by step for fast lookup
    belief_by_step = {b.get("step", i): b for i, b in enumerate(belief_history)}

    # Search dir (absolute, for reading files during HTML generation)
    searches_dir = os.path.join(source_dir, "searches", safe_id)

    # Tool colors
    TOOL_COLORS = {
        "web_search": "#2c3e8c",
        "lookup_url": "#1a6b5e",
        "summarize_results": "#6b2d8c",
        "submit": "#27ae60",
    }

    def tool_color(tool):
        return TOOL_COLORS.get(tool, "#555")

    def _read_file(path):
        """Read a file, return (first_line, full_content) or (None, None) if missing."""
        if not os.path.exists(path):
            return None, None
        content = open(path, encoding="utf-8", errors="replace").read()
        first_line = content.split("\n")[0][:300]
        return first_line, content

    def _clean_first_line(text):
        """Strip 'Search result N:' prefix and '[ok, before ...]' suffix from preview."""
        if not text:
            return text
        # Strip "Search result N: " prefix
        text = re.sub(r'^Search result \d+:\s*', '', text)
        # Strip "[ok, before YYYY-MM-DD]" or "[!! AFTER CUTOFF ...]" suffix
        text = re.sub(r'\s*\[(?:ok, before|!! AFTER CUTOFF|no known date|\?\?).*?\]\s*$', '', text)
        return text.strip()

    def _extract_url_from_content(content):
        """Extract URL from second line of search result file."""
        if not content:
            return None
        for line in content.split("\n")[1:4]:
            line = line.strip()
            if line.startswith("http"):
                return line
        return None

    def _inline_file(fname, file_id, first_line, content, open_=False):
        """Render a single file as an inline expandable <details> block."""
        open_attr = " open" if open_ else ""
        clean_line = _clean_first_line(first_line)
        # Make filename a link to the URL if found in content
        url = _extract_url_from_content(content)
        if url:
            fname_html = f'<a href="{_esc(url)}" target="_blank" style="color:#2980b9"><strong>{_esc(fname)}</strong></a>'
        else:
            fname_html = f'<strong>{_esc(fname)}</strong>'
        return (f'<details id="{_esc(file_id)}"{open_attr} style="margin:2px 0 2px 12px">'
                f'<summary style="cursor:pointer;font-size:0.88em">'
                f'{fname_html}'
                f'{": " + _esc(clean_line) if clean_line else ""}'
                f'</summary>'
                f'<pre style="white-space:pre-wrap;font-size:0.8em;background:#f5f5f5;'
                f'padding:8px;max-height:300px;overflow-y:auto">{_esc(content)}</pre>'
                f'</details>')

    # Render agent trace entries
    trace_html = ""

    # Show initial belief state (step 0)
    b0 = belief_by_step.get(0, {})
    if b0:
        trace_html += (
            '<div style="margin:8px 0;padding:8px;background:#f0f5ff;border-radius:4px;'
            'font-size:0.9em">'
            f'<strong>Initial belief:</strong> p={b0.get("p", 0.5)}'
            '</div>')

    for entry in tool_log:
        etype = entry.get("type", "tool_call")
        if etype == "thinking":
            preview = _esc((entry.get("text", "") or "")[:80].replace("\n", " "))
            full = _esc(entry.get("text", ""))
            trace_html += (
                f'<details style="margin:6px 0;border-left:3px solid #f0c060;padding-left:8px;'
                f'background:#fffde7;border-radius:0 4px 4px 0">'
                f'<summary style="cursor:pointer;font-size:0.88em;padding:4px 0">'
                f'Thinking: {preview}{"..." if len(entry.get("text",""))>80 else ""}</summary>'
                f'<pre style="font-size:0.82em;white-space:pre-wrap;padding:4px 8px">{full}</pre>'
                f'</details>')
            continue
        if etype == "reasoning":
            preview = _esc((entry.get("text", "") or "")[:80].replace("\n", " "))
            full = _esc(entry.get("text", ""))
            trace_html += (
                f'<details style="margin:6px 0;border-left:3px solid #aaa;padding-left:8px">'
                f'<summary style="cursor:pointer;color:#666;font-size:0.88em">'
                f'Reasoning: {preview}{"..." if len(entry.get("text",""))>80 else ""}</summary>'
                f'<pre style="font-size:0.82em;background:#f8f8f8;padding:8px;'
                f'white-space:pre-wrap;max-height:300px;overflow:auto">{full}</pre>'
                f'</details>')
            continue
        if etype == "timeout":
            trace_html += ('<div style="background:#e74c3c;color:#fff;padding:6px 10px;'
                           'border-radius:4px;margin:6px 0;font-size:0.88em">'
                           'Timeout \u2014 final answer requested</div>')
            continue

        # tool_call entry
        tool = entry.get("tool", "?")
        step = entry.get("step", "?")
        color = tool_color(tool)
        step_tokens = ""
        if entry.get("tokens_in"):
            step_tokens = (f' <span style="font-size:0.8em;opacity:0.8">'
                           f'{entry["tokens_in"]:,}in+{entry.get("tokens_out",0):,}out</span>')

        # Build tool-specific header and inline content
        if tool == "web_search":
            query = _esc(entry.get("query", ""))
            n_res = entry.get("n_results", 0)
            sidx = entry.get("search_index", 0)
            header_body = f'<code style="background:#eef2ff;padding:2px 6px;border-radius:3px">{query}</code> <span style="color:#888">({n_res} results)</span>'

            # Inline result files
            result_items = ""
            for j in range(n_res):
                fpath = os.path.join(searches_dir, f"search_{sidx}_result_{j}.md")
                first_line, content = _read_file(fpath)
                fname = f"search_{sidx}_result_{j}.md"
                fid = fname
                if content is not None:
                    result_items += _inline_file(fname, fid, first_line, content)
                else:
                    result_items += f'<div style="font-size:0.85em;color:#888;margin:2px 12px">{fname} (not found)</div>'

            # Filter log inline
            flog_path = os.path.join(searches_dir, f"search_{sidx}_filter_log.txt")
            flog_first, flog_content = _read_file(flog_path)
            filter_html = ""
            if flog_content is not None:
                filter_html = _inline_file(f"search_{sidx}_filter_log.txt",
                                           f"search_{sidx}_filter_log",
                                           None, flog_content)

            extra = (f'<details style="margin-top:6px">'
                     f'<summary style="cursor:pointer;color:#2c6fad;font-size:0.9em">'
                     f'Raw results ({n_res} files)</summary>'
                     f'<div style="margin-top:4px">{result_items}{filter_html}</div>'
                     f'</details>')

        elif tool == "summarize_results":
            sidx = entry.get("search_index", 0)
            ridxs = entry.get("result_indices", [])
            header_body = f'Summarized search #{sidx}, results {ridxs}'
            fpath = os.path.join(searches_dir, f"search_{sidx}_summary.md")
            first_line, content = _read_file(fpath)
            fname = f"search_{sidx}_summary.md"
            extra = (_inline_file(fname, fname, first_line, content)
                     if content is not None
                     else f'<div style="font-size:0.85em;color:#888">{fname} (not found)</div>')

        elif tool == "lookup_url":
            url = entry.get("url", "")
            title = _esc(entry.get("title", url))
            lidx = entry.get("lookup_index", 0)
            header_body = f'<a href="{_esc(url)}" target="_blank" style="color:#1a6b5e">{title}</a>'
            fpath = os.path.join(searches_dir, f"lookup_{lidx}.md")
            first_line, content = _read_file(fpath)
            fname = f"lookup_{lidx}.md"
            extra = (_inline_file(fname, fname, None, content)
                     if content is not None
                     else f'<div style="font-size:0.85em;color:#888">{fname} (not found)</div>')

        elif tool == "submit":
            final_p = entry.get("final_p", forecast_p)
            header_body = f"p={final_p}"
            extra = ""

        else:
            # Source tools (fetch_ts_*, fetch_wikipedia_*, etc.)
            args_dict = entry.get("args", {})
            if args_dict:
                args_str = ", ".join(f'{k}=<code>{_esc(str(v))}</code>'
                                     for k, v in args_dict.items())
                header_body = f'<span style="font-size:0.9em">{args_str}</span>'
            else:
                header_body = ""
            # Check for saved tool result file
            result_fname = entry.get("result_file", "")
            if result_fname:
                fpath = os.path.join(searches_dir, result_fname)
                first_line, content = _read_file(fpath)
                if content is not None:
                    extra = _inline_file(result_fname, result_fname, first_line, content)
                else:
                    extra = ""
            else:
                extra = ""

        # Render thinking/reasoning text if present in the tool call entry
        thinking_html = ""
        thinking_text = entry.get("thinking", "")
        reasoning_text = entry.get("reasoning_text", "")
        if thinking_text:
            # Clean up \n\n patterns
            clean = thinking_text.replace("\\n\\n", "\n").replace("\\n", "\n")
            preview = _esc(clean[:80].replace("\n", " "))
            thinking_html = (
                f'<details style="margin:4px 0 4px 12px;border-left:3px solid #f0c060;'
                f'padding-left:8px;background:#fffde7;border-radius:0 4px 4px 0">'
                f'<summary style="cursor:pointer;font-size:0.85em;color:#888">Thinking: {preview}...</summary>'
                f'<pre style="font-size:0.8em;white-space:pre-wrap;padding:4px 8px">{_esc(clean)}</pre>'
                f'</details>')
        if reasoning_text:
            preview = _esc(reasoning_text[:80].replace("\n", " "))
            thinking_html += (
                f'<details style="margin:4px 0 4px 12px;border-left:3px solid #a0c0ff;'
                f'padding-left:8px;background:#f0f5ff;border-radius:0 4px 4px 0">'
                f'<summary style="cursor:pointer;font-size:0.85em;color:#888">Reasoning: {preview}...</summary>'
                f'<pre style="font-size:0.8em;white-space:pre-wrap;padding:4px 8px">{_esc(reasoning_text)}</pre>'
                f'</details>')

        # Belief state after this step
        belief_html = ""
        b = belief_by_step.get(step)
        if b:
            def _linkify_citations(text):
                """Convert search_X_result_Y references into clickable links.

                Handles: (search_0_result_1), search_0_result_1, search_0_result_1,
                """
                escaped = _esc(text)
                return re.sub(
                    r'\(?search_(\d+)_result_(\d+)\)?',
                    r'<a href="#search_\1_result_\2.md" style="color:#2980b9">'
                    r'search_\1_result_\2</a>',
                    escaped)

            def li_items(items):
                return "".join(f"<li>{_linkify_citations(x)}</li>" for x in items) if items else "<li><em>(none)</em></li>"

            p_val = b.get("p", "?")
            upd = _linkify_citations(b.get("update_reasoning", ""))
            base = _esc(b.get("base_rate_anchor", ""))
            conf = _esc(b.get("confidence", ""))
            ev_for = b.get("evidence_for", [])
            ev_against = b.get("evidence_against", [])
            uncerts = b.get("key_uncertainties", [])

            rows = f"""
<tr><td style="color:#aaa;white-space:nowrap;padding-right:12px;vertical-align:top">p</td>
    <td><strong>{p_val}</strong></td></tr>"""
            if upd:
                rows += f"""
<tr><td style="color:#aaa;white-space:nowrap;vertical-align:top">Update reasoning</td>
    <td style="font-size:0.9em">{upd}</td></tr>"""
            if base:
                rows += f"""
<tr><td style="color:#aaa;white-space:nowrap;vertical-align:top">Base rate</td>
    <td style="font-size:0.9em">{base}</td></tr>"""
            if conf:
                rows += f"""
<tr><td style="color:#aaa;white-space:nowrap;vertical-align:top">Confidence</td>
    <td style="font-size:0.9em">{conf}</td></tr>"""
            if ev_for or ev_against:
                rows += f"""
<tr><td style="color:#aaa;white-space:nowrap;vertical-align:top">Evidence FOR</td>
    <td><ol style="margin:2px 0;padding-left:18px;font-size:0.88em">{li_items(ev_for)}</ol></td></tr>
<tr><td style="color:#aaa;white-space:nowrap;vertical-align:top">Evidence AGAINST</td>
    <td><ol style="margin:2px 0;padding-left:18px;font-size:0.88em">{li_items(ev_against)}</ol></td></tr>"""
            if uncerts:
                rows += f"""
<tr><td style="color:#aaa;white-space:nowrap;vertical-align:top">Uncertainties</td>
    <td><ol style="margin:2px 0;padding-left:18px;font-size:0.88em">{li_items(uncerts)}</ol></td></tr>"""

            belief_html = f"""
<div style="margin:6px 0 6px 12px;padding:8px 12px;background:#f8f9fa;
     border:1px solid #e0e0e0;border-radius:4px;max-width:860px">
  <table style="border-collapse:collapse;width:100%;font-size:0.88em">{rows}</table>
</div>"""

        belief_p_after = entry.get("belief_p")
        belief_tag = (f' <span style="font-weight:normal;color:#555">\u2192 p={belief_p_after:.3f}</span>'
                      if belief_p_after is not None else "")
        tok_tag = (f' <span style="font-weight:normal;color:#888;font-size:0.85em">'
                   f'{entry["tokens_in"]:,}in+{entry.get("tokens_out",0):,}out</span>'
                   if entry.get("tokens_in") else "")

        trace_html += f"""
<div style="margin:12px 0;border-left:3px solid #ddd;padding-left:10px">
  {thinking_html}
  {belief_html}
  <div style="margin-bottom:4px">
    <span style="background:{color};color:white;padding:4px 10px;border-radius:4px;
          font-family:monospace;font-size:0.88em;font-weight:bold">[{step}] {_esc(tool)}</span>
    {belief_tag}{tok_tag}
  </div>
  <div style="margin:4px 0 4px 12px;color:#333">{header_body}</div>
  <div style="margin:4px 0 4px 12px">{extra}</div>
</div>"""

    # Question prompt HTML
    qp_parts = []
    if question:
        qp_parts.append(f'<p style="font-weight:bold;font-size:1.05em">{_esc(question)}</p>')
    if background:
        qp_parts.append(f'<p><strong>Background:</strong> {_esc(background)}</p>')
    if resolution_criteria:
        qp_parts.append(f'<p><strong>Resolution criteria:</strong><br>{_esc(resolution_criteria)}</p>')
    resolution_dates = fc.get("resolution_dates", [])
    if resolution_dates and isinstance(resolution_dates, list) and len(resolution_dates) > 1:
        dates_str = ", ".join(str(d) for d in resolution_dates)
        qp_parts.append(f'<p><strong>Resolution dates:</strong> {_esc(dates_str)}</p>')
    elif resolution_date:
        qp_parts.append(f'<p><strong>Resolution date:</strong> {_esc(resolution_date)}</p>')
    elif resolution_dates:
        qp_parts.append(f'<p><strong>Resolution date:</strong> {_esc(str(resolution_dates[0]))}</p>')
    if forecast_due_date:
        qp_parts.append(f'<p><strong>Forecast due date:</strong> {_esc(forecast_due_date)}</p>')
    if market_value and str(market_value).strip() not in ("", "unknown", "None") and config_info.get("show_crowd"):
        try:
            mv_str = f"{float(market_value):.3f}"
        except (TypeError, ValueError):
            mv_str = str(market_value)
        if market_explanation:
            expl = market_explanation.rstrip(".")
            qp_parts.append(f'<p><strong>Market estimate:</strong> {expl} on {market_date} was {mv_str}</p>')
        else:
            qp_parts.append(f'<p><strong>Market estimate:</strong> {mv_str} (as of {market_date})</p>')
    question_prompt_html = "\n".join(qp_parts)

    # Config info — add ntrials from trial_stats if available
    if config_info:
        display_config = dict(config_info)
        trial_stats = fc.get("trial_stats", {})
        if trial_stats and "n_trials" in trial_stats:
            display_config["ntrials"] = trial_stats["n_trials"]
        elif not display_config.get("ntrials"):
            display_config["ntrials"] = 1
        config_html = _esc(json.dumps(display_config, indent=2))
    else:
        config_html = "(not recorded)"

    # Scores — handle multi-resolution
    multi_ps = fc.get("forecasts")
    multi_os = fc.get("resolved_to")
    rdates_list = fc.get("resolution_dates", [])
    if (multi_ps and isinstance(multi_ps, list) and len(multi_ps) > 1
            and isinstance(multi_os, list)):
        # Multi-resolution: show table of dates × forecasts × outcomes
        score_rows = ""
        for j, (p_j, o_j) in enumerate(zip(multi_ps, multi_os)):
            rd = rdates_list[j] if j < len(rdates_list) else f"date_{j}"
            o_j_f = float(o_j) if o_j is not None else None
            p_j_f = float(p_j) if p_j is not None else 0.5
            if o_j_f is not None:
                bs_j = (p_j_f - o_j_f) ** 2
                o_color = "#27ae60" if o_j_f >= 0.5 else "#e74c3c"
                correct = (o_j_f >= 0.5 and p_j_f > 0.5) or (o_j_f < 0.5 and p_j_f < 0.5)
                p_color = "#27ae60" if correct else "#c0392b"
            else:
                bs_j = None
                o_color = "#888"
                p_color = "#888"
            score_rows += (
                f'<tr><td>{rd}</td>'
                f'<td style="color:{p_color};font-weight:bold">{p_j_f:.3f}</td>'
                f'<td style="color:{o_color}">{f"{o_j_f:.0f}" if o_j_f is not None else "?"}</td>'
                f'<td>{f"{bs_j:.4f}" if bs_j is not None else "—"}</td></tr>')
        score_boxes = (
            f'<table style="font-size:0.9em;border-collapse:collapse;margin:8px 0">'
            f'<tr><th>Resolution date</th><th>Forecast</th><th>Outcome</th><th>Brier Score</th></tr>'
            f'{score_rows}</table>')
    else:
        score_boxes = (
            f'<span class="sbox" style="background:{correct_color};color:white">Forecast: {forecast_p:.3f}</span>'
            f'<span class="sbox" style="background:#ecf0f1">Outcome: {outcome_label}</span>'
        )
        if brier is not None:
            score_boxes += f'<span class="sbox" style="background:#f0f0f0">Brier Score: {brier:.4f}</span>'
        if metaculus is not None:
            score_boxes += f'<span class="sbox" style="background:#f0f0f0">Metaculus: {metaculus:.1f}</span>'

    elapsed_str = f" | {elapsed:.0f}s" if elapsed else ""
    subtitle = (f'<strong>{_esc(config_name)}</strong> | {_esc(source)} | '
                f'Steps: {n_steps} | Tokens: {tokens_in:,}in + {tokens_out:,}out{elapsed_str}')
    if fc.get("submitted"):
        subtitle += ' | <span style="color:#27ae60">Submitted</span>'

    chart_svg = _belief_svg(belief_history)

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{_esc(safe_id)} \u2014 {_esc(config_name)}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0 auto;
       padding: 24px 32px; max-width: 960px; background: #fff; color: #222; font-size: 14px;
       line-height: 1.5; }}
h1 {{ font-size: 1.3em; margin: 0 0 6px; line-height: 1.4; }}
h2 {{ font-size: 1.05em; margin: 24px 0 10px; border-bottom: 2px solid #e0e0e0; padding-bottom: 4px; }}
h3 {{ font-size: 0.95em; margin: 16px 0 6px; color: #444; }}
.subtitle {{ color: #555; font-size: 0.88em; margin-bottom: 10px; }}
.sbox {{ display: inline-block; padding: 6px 12px; margin: 3px 3px 3px 0;
         border-radius: 4px; font-weight: bold; font-size: 0.92em; }}
details summary {{ cursor: pointer; font-size: 0.9em; color: #2980b9; font-weight: bold;
                   padding: 2px 0; user-select: none; }}
details summary:hover {{ color: #1a5c8a; }}
.qp {{ font-size: 0.88em; line-height: 1.6; color: #333; border: 1px solid #eee;
       border-radius: 4px; padding: 10px 14px; background: #fafafa; margin-top: 6px; }}
.qp p {{ margin: 6px 0; }}
pre {{ white-space: pre-wrap; font-size: 0.82em; background: #f5f5f5;
       border: 1px solid #e0e0e0; border-radius: 4px; padding: 8px 12px;
       overflow-x: auto; max-height: 300px; overflow-y: auto; }}
a {{ color: #2980b9; }}
code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 0.92em; }}
</style>
</head>
<body>
<h1>{_esc(question[:200])}</h1>
<div class="subtitle">{subtitle}</div>
<div style="margin: 10px 0 18px">{score_boxes}</div>

<details open>
  <summary>Question Prompt</summary>
  <div class="qp">{question_prompt_html}</div>
</details>

{'<details style="margin-top:6px"><summary>System Prompt</summary><pre style="margin-top:6px">' + _esc(system_prompt) + '</pre></details>' if system_prompt else ''}

<details style="margin-top:6px">
  <summary>Config</summary>
  <pre style="margin-top:6px">{config_html}</pre>
</details>

<h2>Belief Evolution</h2>
{chart_svg if chart_svg else '<p style="color:#999;font-size:0.88em">(no belief history)</p>'}

<h2>Agent Trace</h2>
{trace_html if trace_html else '<p style="color:#888">(no tool calls recorded)</p>'}

{'<h2>Final Reasoning</h2><pre>' + _esc(reasoning) + '</pre>' if reasoning else ''}
</body>
</html>"""

    out_path = os.path.join(source_dir, f"{safe_id}_trace.html")
    with open(out_path, "w") as f:
        f.write(page)


# ---------------------------------------------------------------------------
# Multi-trial summary page
# ---------------------------------------------------------------------------

def _render_trial_summary_page(avg_fc: dict, trial_fcs: list[tuple[int, dict]],
                               config_name: str, source: str,
                               safe_id: str, source_dir: str):
    """Render a summary page for multi-trial questions."""
    question = avg_fc.get("question", "")
    forecast_p = avg_fc.get("forecast", 0.5)
    outcome = _resolve_outcome(avg_fc.get("resolved_to"))
    trial_stats = avg_fc.get("trial_stats", {})
    resolution_date = avg_fc.get("resolution_date", "")
    forecast_due_date = avg_fc.get("forecast_due_date", "")

    outcome_label = "YES (1)" if outcome == 1 else "NO (0)" if outcome == 0 else "?"
    brier = (forecast_p - outcome) ** 2 if outcome is not None else None
    metaculus = (100 * (1 + math.log2(max(0.001, min(0.999, forecast_p)) if outcome == 1
                                     else max(0.001, min(0.999, 1 - forecast_p))))
                 if outcome is not None else None)

    correct_color = ("#27ae60" if (outcome == 1 and forecast_p > 0.5) or
                                  (outcome == 0 and forecast_p < 0.5)
                    else "#c0392b" if outcome is not None else "#888")

    # Trial table rows
    trial_rows = ""
    trial_ps = []
    for t, tfc in trial_fcs:
        tp = tfc.get("forecast", 0.5)
        trial_ps.append(tp)
        t_brier = (tp - outcome) ** 2 if outcome is not None else None
        t_metaculus = (100 * (1 + math.log2(max(0.001, min(0.999, tp)) if outcome == 1
                                           else max(0.001, min(0.999, 1 - tp))))
                      if outcome is not None else None)
        t_steps = tfc.get("n_steps", 0)
        t_submitted = "Yes" if tfc.get("submitted") else "No"
        t_elapsed = tfc.get("elapsed_seconds", 0)
        trace_link = f"../../{config_name}/trial_{t}/{source}/{safe_id}_trace.html"
        trial_rows += (
            f'<tr>'
            f'<td><a href="{trace_link}">Trial {t}</a></td>'
            f'<td style="text-align:right">{tp:.3f}</td>'
            f'<td style="text-align:right">{t_brier:.4f}</td>'
            f'<td style="text-align:right">{t_metaculus:.1f}</td>'
            f'<td style="text-align:center">{t_submitted}</td>'
            f'<td style="text-align:right">{t_steps}</td>'
            f'<td style="text-align:right">{t_elapsed:.0f}s</td>'
            f'</tr>'
        )

    # Compute stats
    import numpy as _np
    ps_arr = _np.array(trial_ps)
    stats_html = (
        f'<p style="font-size:0.9em;color:#555">'
        f'Mean: {_np.mean(ps_arr):.3f} | '
        f'Median: {_np.median(ps_arr):.3f} | '
        f'Min: {_np.min(ps_arr):.3f} | '
        f'Max: {_np.max(ps_arr):.3f} | '
        f'Std: {_np.std(ps_arr):.3f}'
        f'</p>'
    )

    # Score boxes
    score_boxes = (
        f'<span class="sbox" style="background:{correct_color};color:white">'
        f'Forecast (avg): {forecast_p:.3f}</span>'
        f'<span class="sbox" style="background:#ecf0f1">Outcome: {outcome_label}</span>'
    )
    if brier is not None:
        score_boxes += f'<span class="sbox" style="background:#f0f0f0">Brier Score: {brier:.4f}</span>'
    if metaculus is not None:
        score_boxes += f'<span class="sbox" style="background:#f0f0f0">Metaculus: {metaculus:.1f}</span>'

    # Belief evolution overlay SVG (all trials on one chart, common x-axis)
    W, H, PAD_L, PAD_R, PAD_T, PAD_B = 400, 180, 32, 28, 22, 28
    inner_w = W - PAD_L - PAD_R
    inner_h = H - PAD_T - PAD_B
    colors = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad',
              '#1abc9c', '#d35400', '#2c3e50', '#c0392b', '#7f8c8d']

    # Find max step count across trials for common x-axis
    max_steps = max((len(tfc.get("belief_history", []))
                     for _, tfc in trial_fcs), default=1)
    max_steps = max(max_steps, 2)

    def cx(i):
        return PAD_L + (i / (max_steps - 1)) * inner_w
    def cy(p):
        return PAD_T + (1 - p) * inner_h

    # Y-axis gridlines
    gridlines = ""
    for gp in [0, 0.25, 0.5, 0.75, 1.0]:
        y = cy(gp)
        dash = 'stroke-dasharray="3,3"' if gp != 0.5 else 'stroke-dasharray="6,3"'
        gridlines += (f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" '
                      f'stroke="#ccc" stroke-width="1" {dash}/>'
                      f'<text x="{PAD_L-4}" y="{y+4:.1f}" font-size="9" text-anchor="end" '
                      f'fill="#999">{gp:.2f}</text>')

    # X-axis tick marks (every step or every 2 if many steps)
    tick_interval = 1 if max_steps <= 12 else 2
    xticks = ""
    for s in range(0, max_steps, tick_interval):
        x = cx(s)
        label = "init" if s == 0 else str(s)
        xticks += (f'<line x1="{x:.1f}" y1="{H-PAD_B}" x2="{x:.1f}" y2="{H-PAD_B+4}" '
                   f'stroke="#999" stroke-width="1"/>'
                   f'<text x="{x:.1f}" y="{H-PAD_B+14}" font-size="8" text-anchor="middle" '
                   f'fill="#999">{label}</text>')

    traces_svg = ""
    for idx, (t, tfc) in enumerate(trial_fcs):
        bh = tfc.get("belief_history", [])
        pts = [b.get("p", 0.5) for b in bh]
        if not pts:
            continue
        color = colors[idx % len(colors)]
        n = len(pts)
        points = " ".join(f"{cx(i):.1f},{cy(p):.1f}" for i, p in enumerate(pts))
        traces_svg += (f'<polyline points="{points}" fill="none" stroke="{color}" '
                       f'stroke-width="1.5" opacity="0.7"/>')
        # Label at end
        traces_svg += (f'<text x="{cx(n-1)+6:.1f}" y="{cy(pts[-1])+4:.1f}" '
                       f'font-size="8" fill="{color}">t{t}</text>')

    belief_svg = (f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" '
                  f'style="display:block">{gridlines}{xticks}{traces_svg}</svg>')

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{_esc(safe_id)} \u2014 {_esc(config_name)} (summary)</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 960px; margin: 0 auto;
       padding: 24px 32px; color: #222; font-size: 14px; line-height: 1.5; }}
h1 {{ font-size: 1.3em; margin: 0 0 6px; }}
h2 {{ font-size: 1.05em; margin: 24px 0 10px; border-bottom: 2px solid #e0e0e0;
      padding-bottom: 4px; }}
.subtitle {{ color: #555; font-size: 0.88em; margin-bottom: 10px; }}
.sbox {{ display: inline-block; padding: 6px 12px; margin: 3px 3px 3px 0;
         border-radius: 4px; font-weight: bold; font-size: 0.92em; }}
table {{ border-collapse: collapse; font-size: 0.9em; margin: 12px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; }}
th {{ background: #f5f5f5; }}
a {{ color: #2980b9; }}
</style></head><body>

<h1>{_esc(question[:200])}</h1>
<div class="subtitle"><strong>{_esc(config_name)}</strong> | {_esc(source)} |
{len(trial_fcs)} trials | Forecast date: {_esc(forecast_due_date)} |
Resolution date: {_esc(resolution_date)}</div>

<div style="margin: 10px 0 18px">{score_boxes}</div>
{stats_html}

<h2>Belief Evolution (all trials)</h2>
{belief_svg}

<h2>Per-trial Results</h2>
<table>
<tr><th>Trial</th><th>Forecast</th><th>Brier Score</th><th>Metaculus</th>
<th>Submitted</th><th>Steps</th><th>Time</th></tr>
{trial_rows}
</table>

</body></html>"""

    os.makedirs(source_dir, exist_ok=True)
    out_path = os.path.join(source_dir, f"{safe_id}_trace.html")
    with open(out_path, "w") as f:
        f.write(page)
