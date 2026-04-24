#!/usr/bin/env python3
"""show_prompts.py — Display system + question prompts for inspection.

Two modes:
1. XID mode: show prompts for all configs × first question per source
   python3 src/show_prompts.py --xid xid-tranche-a1

2. Question mode: show prompt variants for a specific question
   python3 src/show_prompts.py --source polymarket --id 0x310c3d... --fdd 2025-10-26

Generates an HTML file for easy inspection in a browser.
"""

import argparse
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
import html as html_lib

from config.config import resolve_config, pprint as cfg_pprint
from agent.prompts import get_system_prompt, format_question_prompt
from agent.tools import get_tool_schemas


def _esc(s):
    return html_lib.escape(str(s))


def _load_question(source, qid):
    """Load a question file."""
    safe = re.sub(r'[/\\:]', '_', str(qid))
    path = os.path.join("data", "questions", source, f"{safe}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 2em; }
h1 { font-size: 1.4em; }
h2 { font-size: 1.1em; margin-top: 2em; color: #333; }
h3 { font-size: 0.95em; margin-top: 1.5em; color: #555; }
.prompt-box { background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;
              padding: 12px; margin: 8px 0; white-space: pre-wrap;
              font-family: monospace; font-size: 0.85em; max-height: 400px;
              overflow-y: auto; }
.system { border-left: 4px solid #2980b9; }
.question { border-left: 4px solid #27ae60; }
.tools { border-left: 4px solid #e67e22; }
.diff { background: #fff3cd; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9em; }
th { background: #f0f0f0; }
.meta { color: #888; font-size: 0.85em; }
details { margin: 4px 0; }
summary { cursor: pointer; font-weight: bold; font-size: 0.9em; }
"""


def _tool_reference_html():
    """Generate HTML reference for all available tools across all sources."""
    all_sources = ["(base)", "fred", "dbnomics", "yfinance",
                   "wikipedia", "polymarket", "manifold"]
    seen = set()
    tools_info = []

    for source in all_sources:
        cfg = resolve_config("pro/thk:high/crowd:1/tools:1")
        src = source if source != "(base)" else ""
        schemas = get_tool_schemas(cfg, source=src)
        for t in schemas:
            fn = t["function"]
            name = fn["name"]
            if name in seen:
                continue
            seen.add(name)
            desc = fn.get("description", "")
            params = fn.get("parameters", {}).get("properties", {})
            param_list = [(k, p.get("type", "?"), p.get("description", ""))
                          for k, p in params.items() if k != "updated_belief"]
            tools_info.append((name, source, desc, param_list))

    parts = ['<details><summary><b>Tool Reference (all sources)</b></summary>']
    parts.append('<table><tr><th>Tool</th><th>Source</th>'
                 '<th>Description</th><th>Parameters</th></tr>')
    for name, source, desc, param_list in tools_info:
        params_html = "<br>".join(
            f'<code>{k}</code> ({t}): {_esc(d)[:80]}'
            for k, t, d in param_list) or "<em>none</em>"
        parts.append(
            f'<tr><td><code>{_esc(name)}</code></td>'
            f'<td>{_esc(source)}</td>'
            f'<td style="font-size:0.85em">{_esc(desc)}</td>'
            f'<td style="font-size:0.85em">{params_html}</td></tr>')
    parts.append('</table></details>')
    return "\n".join(parts)


def generate_xid_html(xid_name, xid_data):
    """Generate HTML showing prompts for all configs × one question per source."""
    from core.eval import load_exam

    exam_name = xid_data["exam"]
    exam = load_exam(exam_name)
    config_deltas = xid_data.get("config", [])

    # Load one sample question per source
    sample_questions = {}
    for source, ids in sorted(exam.items()):
        if ids:
            q = _load_question(source, ids[0])
            if q:
                sample_questions[source] = q

    tool_ref = _tool_reference_html()

    parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Prompts — {_esc(xid_name)}</title>
<style>{_CSS}</style>
</head><body>
<h1>Prompt Inspector: {_esc(xid_name)}</h1>
<p class="meta">Exam: {_esc(exam_name)} |
Sources: {', '.join(sorted(sample_questions.keys()))} |
Configs: {len(config_deltas)}</p>
{tool_ref}
"""]

    for delta in config_deltas:
        cfg = resolve_config(delta)
        name = cfg_pprint(cfg)
        cutoff = "2025-10-26"  # example

        # Pick a representative source
        source = next(iter(sample_questions), "")
        q = sample_questions.get(source, {})
        if q:
            cutoff = q.get("forecast_due_date", cutoff)[:10]

        sys_prompt = get_system_prompt(
            cfg.max_steps, live=not cfg.backtesting, source=source,
            nobelief=cfg.nobelief, use_tools=cfg.use_tools,
            use_search=(cfg.search_engine != "none"))

        tools = get_tool_schemas(cfg, source=source, question=q)
        tool_names = [t["function"]["name"] for t in tools]

        parts.append(f'<h2>{_esc(name)}</h2>')
        parts.append(f'<p class="meta">Delta: <code>{_esc(delta)}</code> | '
                     f'Tools: {", ".join(tool_names)} | '
                     f'Steps: {cfg.max_steps} | '
                     f'Timeout: {cfg.question_timeout}s</p>')

        parts.append('<details><summary>System Prompt</summary>')
        parts.append(f'<div class="prompt-box system">{_esc(sys_prompt)}</div>')
        parts.append('</details>')

        # Show question prompt for each source
        for src, sq in sorted(sample_questions.items()):
            src_cutoff = sq.get("forecast_due_date", cutoff)[:10]
            qp = format_question_prompt(sq, src_cutoff,
                                         show_crowd=cfg.show_crowd,
                                         use_tools=cfg.use_tools,
                                         backtesting=cfg.backtesting,
                                         nobelief=cfg.nobelief)

            src_sys = get_system_prompt(
                cfg.max_steps, live=not cfg.backtesting, source=src,
                nobelief=cfg.nobelief, use_tools=cfg.use_tools,
                use_search=(cfg.search_engine != "none"))

            src_tools = get_tool_schemas(cfg, source=src, question=sq)
            src_tool_names = [t["function"]["name"] for t in src_tools]

            # Only show source-specific system prompt if different from generic
            sys_diff = src_sys != sys_prompt

            qtext = sq.get("question", "")[:80]
            parts.append(f'<details><summary>{_esc(src)} — {_esc(qtext)}...</summary>')
            if sys_diff:
                parts.append(f'<div class="meta">Source-specific tools: {", ".join(src_tool_names)}</div>')
                parts.append(f'<div class="prompt-box system diff">{_esc(src_sys)}</div>')
            parts.append(f'<div class="prompt-box question">{_esc(qp)}</div>')
            parts.append('</details>')

    parts.append("</body></html>")
    return "\n".join(parts)


def generate_question_html(source, qid, fdd):
    """Generate HTML showing all prompt variants for a specific question."""
    q = _load_question(source, qid)
    if not q:
        # Try with fdd suffix
        q = _load_question(source, f"{qid}_{fdd}")
    if not q:
        print(f"ERROR: question not found: {source}/{qid}")
        return None

    cutoff = (fdd or q.get("forecast_due_date", ""))[:10]
    qtext = q.get("question", "")[:100]

    # Variants to show
    variants = [
        ("Backtesting, crowd=0, tools=0, search=brave",
         {"show_crowd": 0, "use_tools": False, "search": "brave",
          "backtesting": True, "nobelief": False}),
        ("Backtesting, crowd=1, tools=0, search=brave",
         {"show_crowd": 1, "use_tools": False, "search": "brave",
          "backtesting": True, "nobelief": False}),
        ("Backtesting, crowd=0, tools=1, search=brave",
         {"show_crowd": 0, "use_tools": True, "search": "brave",
          "backtesting": True, "nobelief": False}),
        ("Backtesting, crowd=1, tools=1, search=brave",
         {"show_crowd": 1, "use_tools": True, "search": "brave",
          "backtesting": True, "nobelief": False}),
        ("Backtesting, crowd=0, tools=0, search=none (zero-shot)",
         {"show_crowd": 0, "use_tools": False, "search": "none",
          "backtesting": True, "nobelief": False}),
        ("Backtesting, crowd=0, tools=1, search=none",
         {"show_crowd": 0, "use_tools": True, "search": "none",
          "backtesting": True, "nobelief": False}),
        ("Backtesting, nobelief=1",
         {"show_crowd": 0, "use_tools": True, "search": "brave",
          "backtesting": True, "nobelief": True}),
        ("Live mode, crowd=1, tools=1",
         {"show_crowd": 1, "use_tools": True, "search": "brave",
          "backtesting": False, "nobelief": False}),
    ]

    tool_ref = _tool_reference_html()

    parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Prompt Variants — {_esc(source)}/{_esc(qid)}</title>
<style>{_CSS}</style>
</head><body>
<h1>Prompt Variants</h1>
<p class="meta">Source: {_esc(source)} | ID: {_esc(qid)} |
Forecast date: {_esc(cutoff)}</p>
<p><b>Question:</b> {_esc(qtext)}</p>
{tool_ref}
"""]

    for label, settings in variants:
        use_search = settings["search"] != "none"
        sys_prompt = get_system_prompt(
            10, live=not settings["backtesting"], source=source,
            nobelief=settings["nobelief"], use_tools=settings["use_tools"],
            use_search=use_search)

        qp = format_question_prompt(q, cutoff,
                                     show_crowd=settings["show_crowd"],
                                     use_tools=settings["use_tools"],
                                     backtesting=settings["backtesting"],
                                     nobelief=settings["nobelief"])

        # Build a minimal config for tool schemas
        cfg = resolve_config("pro/thk:high")
        cfg.show_crowd = settings["show_crowd"]
        cfg.use_tools = settings["use_tools"]
        cfg.search_engine = settings["search"]
        cfg.nobelief = settings["nobelief"]
        cfg.backtesting = settings["backtesting"]
        tools = get_tool_schemas(cfg, source=source, question=q)
        tool_names = [t["function"]["name"] for t in tools]

        parts.append(f'<h2>{_esc(label)}</h2>')
        parts.append(f'<p class="meta">Tools: {", ".join(tool_names)}</p>')
        parts.append('<details open><summary>System Prompt</summary>')
        parts.append(f'<div class="prompt-box system">{_esc(sys_prompt)}</div>')
        parts.append('</details>')
        parts.append('<details open><summary>Question Prompt</summary>')
        parts.append(f'<div class="prompt-box question">{_esc(qp)}</div>')
        parts.append('</details>')

    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Display system + question prompts for inspection")
    parser.add_argument("--xid", default=None,
                        help="Show prompts for all configs in an xid")
    parser.add_argument("--source", default=None,
                        help="Question source (for single-question mode)")
    parser.add_argument("--id", default=None,
                        help="Question ID (for single-question mode)")
    parser.add_argument("--fdd", default=None,
                        help="Forecast due date (for single-question mode)")
    args = parser.parse_args()

    out_dir = os.path.join("experiments", "generated_prompts")
    os.makedirs(out_dir, exist_ok=True)

    if args.xid:
        from core.eval import load_xid
        xid_data = load_xid(args.xid)
        html = generate_xid_html(args.xid, xid_data)
        out_path = os.path.join(out_dir, f"{args.xid}.html")
        with open(out_path, "w") as f:
            f.write(html)
        print(f"  {out_path}")
        return

    if args.source and args.id:
        html = generate_question_html(args.source, args.id, args.fdd)
        if html:
            safe_id = re.sub(r'[/\\:]', '_', str(args.id))[:40]
            out_path = os.path.join(out_dir, f"{args.source}_{safe_id}.html")
            with open(out_path, "w") as f:
                f.write(html)
            print(f"  {out_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
