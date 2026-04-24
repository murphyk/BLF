# Forecast Bench v6

LLM-based agentic forecasting system for binary questions. Supports backtesting on AIBQ2 (Metaculus) and ForecastBench (markets + datasets), plus live competition submission.

## Context
- `docs/workflow.md` — full pipeline walkthrough
- `docs/TODO.md` — tracked TODOs for paper and code
- `src/MODULES.md` — categorized guide to all source modules
- `docs/shrinkage.tex` — logit-space shrinkage derivation
- `../nips26/main.tex` — NeurIPS 2026 paper draft (moved to top-level)

## Layout
- `src/` — all Python source (see `src/MODULES.md` for module guide)
- `data/` — inputs: questions, exams, tags, caches
- `experiments/` — xid definitions + outputs (forecasts, eval)
- Configs use hyphenated directory names (e.g. `flash-high-brave-c1-t1`); delta strings (e.g. `flash/thk:high/crowd:1`) are used on the CLI but resolved to directory names internally
