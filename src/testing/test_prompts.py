#!/usr/bin/env python3
"""test_prompts.py — Verify prompt generation for all ablation configs.

Checks that system and question prompts are consistent with config settings:
- Tools mentioned in prompt match tools in schema
- No mention of unavailable tools
- Crowd/market info present iff show_crowd=1
- Belief state instructions present iff nobelief=False
- Step count matches config

Usage:
    python3 src/test_prompts.py                    # run all checks
    python3 src/test_prompts.py --exam smoke       # test with specific exam
    python3 src/test_prompts.py --verbose           # show each check
"""

import argparse
import json
import os
import re
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

from config.config import resolve_config, AgentConfig
from agent.prompts import get_system_prompt, format_question_prompt
from agent.tools import get_tool_schemas


# All ablation configs from the paper
# Short names are used in figure legends, tables, and CLI output.
# Most ablations use crowd=0 so differences are visible (crowd dominates when on).
ABLATION_CONFIGS = {
    "full":                  "pro/thk:high/crowd:1/tools:1",
    "full-nocrowd":          "pro/thk:high/crowd:0/tools:1",
    "notools-nocrowd":       "pro/thk:high/crowd:0/tools:0",
    "nosearch-nocrowd":      "pro/thk:high/search:none/crowd:0/tools:1",
    "medthink-nocrowd":      "pro/thk:default/crowd:0/tools:1",
    "flash-nocrowd":         "flash/thk:high/crowd:0/tools:1",
    "nobelief-nocrowd":      "pro/thk:high/crowd:0/tools:1/nobelief:1",
    "batch-nocrowd":         "pro/thk:high/crowd:0/tools:1/batch_queries:5",
    "kimi-nocrowd":          "kimi-k2t/thk:high/crowd:0/tools:1",
    "zs":                    "pro/thk:high/search:none/crowd:0/tools:0/steps:1",
    "zs-crowd":              "pro/thk:high/search:none/crowd:1/tools:0/steps:1",
    "full-clairv":           "pro/thk:high/crowd:1/tools:1/clairvoyant:1",
}

# Source-specific tool names
SOURCE_TOOLS = {
    "yfinance": "fetch_ts_yfinance",
    "fred": "fetch_ts_fred",
    "dbnomics": "fetch_ts_dbnomics",
    "wikipedia": "fetch_wikipedia_toc",
    "polymarket": "fetch_polymarket_info",
    "manifold": "fetch_manifold_info",
}

MARKET_SOURCES = {"infer", "manifold", "metaculus", "polymarket"}
DATASET_SOURCES = {"acled", "dbnomics", "fred", "wikipedia", "yfinance"}


def check_config(config_id: str, delta: str, source: str,
                 question: dict, verbose: bool = False) -> list[str]:
    """Run all checks for one config + source combination. Returns list of errors."""
    errors = []
    cfg = resolve_config(delta)

    # Generate system prompt
    live_mode = False  # backtesting
    sys_prompt = get_system_prompt(
        cfg.max_steps, live=live_mode, source=source,
        nobelief=cfg.nobelief,
        use_tools=cfg.use_tools,
        use_search=(cfg.search_engine != "none"))

    # Generate question prompt
    cutoff = question.get("forecast_due_date", "2025-10-26")[:10]
    q_prompt = format_question_prompt(
        question, cutoff,
        show_crowd=cfg.show_crowd,
        use_tools=cfg.use_tools,
        backtesting=True,
        nobelief=cfg.nobelief)

    # Get tool schemas
    tools = get_tool_schemas(cfg, source=source, question=question)
    tool_names = {t["function"]["name"] for t in tools}

    # Force-submit filtering (simulates what agent.py does)
    if cfg.max_steps == 1:
        tool_names = {n for n in tool_names if n == "submit"}

    combined = sys_prompt + "\n" + q_prompt

    def check(condition, msg):
        if not condition:
            errors.append(f"[{config_id}/{source}] {msg}")
        elif verbose:
            print(f"  OK [{config_id}/{source}] {msg}")

    # --- Check 1: submit tool always available ---
    check("submit" in tool_names,
          "submit tool should always be available")

    # --- Check 2: source-specific tool availability ---
    src_tool = SOURCE_TOOLS.get(source)
    if src_tool:
        if cfg.use_tools:
            check(src_tool in tool_names,
                  f"{src_tool} should be in tool schemas (use_tools=True)")
            # Also check prompt mentions it (unless forced submit)
            if cfg.max_steps > 1:
                check(src_tool in sys_prompt,
                      f"{src_tool} should be mentioned in system prompt")
        else:
            check(src_tool not in tool_names,
                  f"{src_tool} should NOT be in tool schemas (use_tools=False)")
            check(src_tool not in sys_prompt,
                  f"{src_tool} should NOT be in system prompt (use_tools=False)")

    # --- Check 3: web_search availability ---
    if cfg.search_engine != "none":
        if cfg.max_steps > 1:
            check("web_search" in tool_names,
                  "web_search should be available (search != none)")
    else:
        check("web_search" not in tool_names,
              "web_search should NOT be available (search=none)")

    # --- Check 4: crowd/market info in question prompt ---
    has_crowd = ("Market estimate" in q_prompt or "market price" in q_prompt.lower()
                 or "Market probability" in q_prompt or "current value" in q_prompt.lower()
                 or "most recent" in q_prompt.lower())
    mv = question.get("market_value")
    if cfg.show_crowd and mv is not None and str(mv).strip():
        check(has_crowd,
              "crowd/market info should be in question prompt (show_crowd=1)")
    elif not cfg.show_crowd:
        check("Market estimate" not in q_prompt,
              "market estimate should NOT be in prompt (show_crowd=0)")

    # --- Check 5: belief state instructions ---
    has_belief = "updated_belief" in combined or "belief state" in combined.lower()
    if cfg.nobelief:
        # Check tool schemas don't have updated_belief
        for t in tools:
            props = t["function"]["parameters"].get("properties", {})
            check("updated_belief" not in props,
                  f"updated_belief should NOT be in {t['function']['name']} schema (nobelief)")
        check("updated_belief" not in q_prompt,
              "updated_belief should NOT be in question prompt (nobelief)")
    else:
        if cfg.max_steps > 1:
            check("updated_belief" in q_prompt or "belief state" in q_prompt.lower(),
                  "belief state instructions should be in question prompt")

    # --- Check 6: max_steps mentioned correctly ---
    step_str = f"step {cfg.max_steps}"
    check(step_str in sys_prompt,
          f"system prompt should mention 'step {cfg.max_steps}'")

    # --- Check 7: zero-shot configs should only have submit ---
    if cfg.max_steps == 1 and cfg.search_engine == "none" and not cfg.use_tools:
        check(tool_names == {"submit"},
              "zero-shot config should only have submit tool")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Test prompt generation")
    parser.add_argument("--exam", default="smoke",
                        help="Exam to load questions from (default: smoke)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load a sample question from each source
    exam_path = os.path.join("data", "exams", args.exam, "indices.json")
    if not os.path.exists(exam_path):
        sys.exit(f"Exam not found: {exam_path}")
    with open(exam_path) as f:
        exam = json.load(f)

    sample_questions = {}
    for source, ids in exam.items():
        if ids:
            safe = re.sub(r'[/\\:]', '_', str(ids[0]))
            q_path = os.path.join("data", "questions", source, f"{safe}.json")
            if os.path.exists(q_path):
                with open(q_path) as f:
                    sample_questions[source] = json.load(f)

    print(f"Testing {len(ABLATION_CONFIGS)} configs × {len(sample_questions)} sources")
    print(f"Sources: {sorted(sample_questions.keys())}")
    print()

    all_errors = []
    n_checks = 0
    for config_id, delta in sorted(ABLATION_CONFIGS.items()):
        cfg = resolve_config(delta)
        if args.verbose:
            print(f"\n--- Config {config_id}: {delta} ---")
        for source, question in sorted(sample_questions.items()):
            errs = check_config(config_id, delta, source, question,
                               verbose=args.verbose)
            all_errors.extend(errs)
            n_checks += 1

    print(f"\n{'=' * 60}")
    print(f"Ran checks on {n_checks} (config, source) combinations")
    if all_errors:
        print(f"\nFAILED — {len(all_errors)} errors:")
        for e in all_errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("PASSED — all checks OK")


if __name__ == "__main__":
    main()
