# Source Module Guide

Quick reference for navigating the codebase. See `docs/workflow.md` for
the full pipeline walkthrough.

## Directory Structure

```
src/
├── core/       Main pipeline entry points
├── agent/      Agent loop, belief state, prompts, tools
├── config/     Configuration, tags, paths
├── data/       Data download, exam creation, tagging
├── eval/       Evaluation plots and HTML
├── analysis/   Statistical analysis scripts
├── search/     Web search engine implementations
├── compete/    Live competition submission
├── testing/    Tests and validation
└── misc/       Legacy and one-off scripts
```

## Core Scripts (`core/`)

Main entry points, run from the command line.

| Module | Workflow step | Purpose |
|---|---|---|
| `core/predict.py` | 6. Predict | Run agentic forecaster on an exam |
| `core/eval.py` | 8. Evaluate | Score forecasts, generate leaderboard + plots |
| `core/calibrate.py` | 9. Calibrate | Platt scaling (global or hierarchical) |
| `core/ensemble.py` | 10. Ensemble | Greedy forward selection of config ensemble |
| `core/aggregate.py` | 6. Post-predict | Compute aggregation variants (mean, shrink) |
| `core/reaggregate.py` | — | Re-aggregate trial forecasts (arithmetic mean) |

## Data Scripts (`data/`)

| Module | Workflow step | Purpose |
|---|---|---|
| `data/fb_make_data.py` | 1. Download | Download ForecastBench question sets |
| `data/aibq2_make_data.py` | 1. Download | Download AIBQ2 question data |
| `data/make_exam.py` | 3. Create exams | Build exam indices from mixture spec |
| `data/make_exams_all.py` | 3. Create exams | Rebuild all exams at once |
| `data/plot_exams.py` | 3b. Visualize | Plot exam scatter, histogram, tag distribution |
| `data/classify_questions.py` | 2. Tag | LLM-classify questions into categories |
| `data/fb_leaderboard.py` | — | Discover/import/compare FB methods |

## Agent Library (`agent/`)

Imported by core scripts; no `__main__`.

| Module | Purpose |
|---|---|
| `agent/agent.py` | Agent loop: LLM → tool → belief update → repeat |
| `agent/belief_state.py` | Structured belief state (p, confidence, evidence, uncertainties) |
| `agent/prompts.py` | System + question prompt generation |
| `agent/tools.py` | Tool schemas and dispatch (web_search, submit, etc.) |
| `agent/source_tools.py` | Source-specific data tools (yfinance, FRED, etc.) |
| `agent/data_tools.py` | Lower-level data fetching utilities |
| `agent/llm_client.py` | Thin wrapper around litellm for chat calls |

## Config Library (`config/`)

| Module | Purpose |
|---|---|
| `config/config.py` | `AgentConfig` dataclass, delta string parsing |
| `config/config_display.py` | Config struct for plot labels |
| `config/tags.py` | Unified tagging system (Qsource, Qtype, FBQtype, etc.) |
| `config/paths.py` | Centralized directory layout constants |
| `config/knowledge_cutoffs.py` | Model knowledge cutoff date lookups |
| `config/empirical_prior.py` | Empirical base rates for dataset questions |

## Eval Library (`eval/`)

| Module | Purpose |
|---|---|
| `eval/eval_html.py` | HTML generation for leaderboard + dashboard |
| `eval/eval_plots.py` | All matplotlib plot generation |
| `eval/eval_trace.py` | Per-question trace detail pages |
| `eval/adjusted_brier.py` | Difficulty-adjusted Brier score computation |

## Analysis Scripts (`analysis/`)

| Module | Purpose |
|---|---|
| `analysis/mixed_effects.py` | Paired analysis (ANOVA + pairwise comparisons) |
| `analysis/eval_ablations.py` | Ablation-specific eval plots |
| `analysis/shrinkage_evaluation.py` | Evaluate shrinkage aggregation variants |
| `analysis/fb_analyze_data.py` | Analyze ForecastBench question distributions |
| `analysis/eval_ts_models.py` | Evaluate time-series statistical models |

## Testing & Validation (`testing/`)

| Module | Purpose |
|---|---|
| `testing/test_prompts.py` | Verify prompt/tool consistency across configs × sources |
| `testing/test_smoke.py` | Sanity-check forecast outputs (submit rate, valid probs) |
| `testing/leak_detective.py` | Post-hoc leakage audit on search results + reasoning |

## Other

| Module | Purpose |
|---|---|
| `compete/fb_compete.py` | Submit forecasts to live FB competition |
| `misc/show_prompts.py` | Dump generated prompts to disk for inspection |
| `misc/convert_legacy_data.py` | Convert v5 forecasts/exams to v6 format |
| `misc/fred_models.py` | Enhanced FRED statistical models |
| `misc/run.py` | Legacy runner (deprecated) |
| `search/` | Search engine implementations (Brave, Serper, etc.) |

## Key Data Flow

```
data/fb_make_data.py → data/questions/{source}/{id}.json
                              ↓
data/make_exam.py → data/exams/{name}/indices.json
data/plot_exams.py → data/exams/{name}/*.png
                              ↓
core/predict.py → experiments/forecasts_raw/{config}/{source}/{id}.json
    uses: agent/agent.py → agent/prompts.py + agent/tools.py + agent/belief_state.py
                              ↓
core/eval.py → experiments/eval/{xid}/leaderboard.html + plots
    uses: eval/eval_html.py + eval/eval_plots.py + eval/eval_trace.py
```
