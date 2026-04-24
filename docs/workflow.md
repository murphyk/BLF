# Forecasting Workflow

End-to-end pipeline for the agentic forecaster.
Supports backtesting on
[AIBQ2](https://thinkingmachines.ai/news/training-llms-to-predict-world-events/#datasets)
(aka Q2 2025 Metaculus AI Benchmark Tournament test set of 113 questions),
[ForecastBench](https://www.forecastbench.org/) (market and dataset questions),
and live competition submission.

---

# Setup

## API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required keys:**

| Key | Purpose | Where to get it |
|---|---|---|
| `OPENROUTER_API_KEY` | All LLM calls (via litellm) | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `BRAVE_API_KEY` | Web search (default engine) | [brave.com/search/api](https://brave.com/search/api/) |

**Optional keys:**

| Key | Purpose |
|---|---|
| `FRED_API_KEY` | FRED economic data tool |
| `SERPER_API_KEY` | Alternative search engine |
| `GOOGLE_API_KEY` | Alternative search engine |
| `EXA_API_KEY` | Alternative search engine |

Keys are loaded automatically from `.env` via `python-dotenv`.
You can also set them as shell environment variables:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export BRAVE_API_KEY=...
python3 src/core/predict.py --xid my-xid --ntrials 1 --verbose
```

**macOS SSL fix:** If you get SSL certificate errors, add to `.env`:
```
SSL_CERT_FILE=/path/to/cacert.pem
```
Find the path with: `python3 -c "import certifi; print(certifi.where())"`

---

# Smoke tests

Two prebuilt smoke xids let you verify the install (API keys, dotenv
loading, question data, agent loop, eval pipeline) before running a real
experiment. Both use `flash` at `thk:high, crowd:1, tools:1` — fast, cheap,
and exercise the full predict → eval path.

## `xid-smoke-aibq2-n2` — minimal (2 questions)

Two AIBQ2 questions. Fastest end-to-end check; finishes in 1–2 minutes.

```bash
python3 src/core/predict.py --xid xid-smoke-aibq2-n2 --verbose
python3 src/core/eval.py --xid xid-smoke-aibq2-n2
open experiments/eval/xid-smoke-aibq2-n2/leaderboard.html
```

Use when: first-time setup verification, or after a change that could
break basic agent-loop mechanics (prompt formatting, tool dispatch,
forecast file I/O).

## `xid-smoke-tranche-a-n1each` — per-source coverage (9 questions)

One question from each of the 9 ForecastBench sources: `acled`,
`dbnomics`, `fred`, `infer`, `manifold`, `metaculus`, `polymarket`,
`wikipedia`, `yfinance`. Exam built from
`data/exams/tranche-a-n1each/mixture.json` with `select: {<source>: 1}`
for each (uses the same `seed=42` as `tranche-a`, so picks the first
question per source after shuffling).

```bash
python3 src/core/predict.py --xid xid-smoke-tranche-a-n1each --verbose
python3 src/core/eval.py --xid xid-smoke-tranche-a-n1each
open experiments/eval/xid-smoke-tranche-a-n1each/leaderboard.html
```

Wall time: ~3 minutes with 50 parallel workers (default) — limited by the
slowest question.

Use when: you've changed anything source-specific (the per-source tool
set in `config.py`, the meta-controller, data-fetching tools for
`yfinance`/`fred`/`dbnomics`, or prompt scaffolding that differs by
`Qsource`/`FBQtype`). Covers every tool path in a single run.

## Post-prediction sanity check

`test_smoke.py` validates the forecast JSONs produced by *any* xid
(not just the smoke ones) without running the eval pipeline:

```bash
python3 src/testing/test_smoke.py --xid my-xid --verbose
```

It checks that:
- every config produced forecasts for every question;
- probabilities are in range (0.01–0.99);
- submit rate is acceptable per config type;
- no unauthorized tool calls (configs with `tools=0` didn't call source
  tools, configs with `search=none` didn't call `web_search`);
- forecasts aren't all stuck at 0.5 (model actually engaged);
- Brier Index is above chance on average.

Hallucinated tool calls (model invokes a tool not in the schema) surface
as **warnings** — the agent loop rejects them and retries, so they
indicate model-quality issues rather than bugs.

---

# Pipeline

## 1. Download data

**AIBQ2:**
```bash
python3 src/data/aibq2_make_data.py          # skip existing
python3 src/data/aibq2_make_data.py --force  # overwrite all
```

**ForecastBench** (resolved questions for backtesting):
```bash
python3 src/data/fb_make_data.py --start-date 2025-10-26 --end-date 2026-03-31
python3 src/data/fb_make_data.py --github-url https://github.com/.../resolution_set.json --exam my-exam
```

Sources: `polymarket`, `manifold`, `metaculus`, `infer` (markets) and
`acled`, `dbnomics`, `fred`, `wikipedia`, `yfinance` (datasets).

All questions are written to `data/questions/{source}/{id}.json`.

## 2. Tag questions (optional)

Classify questions into topical categories for analysis in eval plots:

```bash
python3 src/data/classify_questions.py --version kevin
python3 src/data/classify_questions.py --source aibq2
```

Tags are written to `data/tags_{version}/{source}/{id}.json`.

The system also exposes **virtual tag spaces** derived automatically from
question metadata (no classification step needed). These drive the
`groups` field on an xid and the `group_by` argument to calibrate
(see `src/config/tags.py`):

| Tag space | Values | Use |
|---|---|---|
| `Qsource` | polymarket, fred, etc. | Per-source analysis + default calibration grouping |
| `FBQtype` | market, dataset | ForecastBench-style grouping |
| `Qtype` | timeseries, wikipedia, acled, market | Per-type policy dispatch in `predict.py` |
| `Atype` | binary-single, binary-multi | Answer format |

## 3. Create exams {#create-exams}

An exam is a named view of questions defined by `data/exams/{name}/mixture.json`.
Running `make_exam.py` materializes it into `indices.json`, `meta.json`, and
data visualization plots.

### Mixture fields

| Field | Description | Default |
|---|---|---|
| `ask-start` | Only include questions with `forecast_due_date >= value` | `1900-01-01` |
| `ask-end` | Only include questions with `forecast_due_date <= value` | `2999-12-31` |
| `resolution-start` | Only include questions resolved on or after this date | *(no filter)* |
| `resolution-end` | Only include questions resolved on or before this date | *(no filter)* |
| `seed` | Random seed for shuffling (`null` = alphabetical, integer = deterministic) | `0` |
| `select` | Dict of `{source: N}` where N is count, `[offset, count]`, or `"all"` | all sources |

### Select syntax

| Value | Meaning |
|---|---|
| `N` | First N questions (after shuffling) |
| `"all"` | All questions matching date filters |
| `[offset, count]` | Skip `offset`, take `count` (for train/test splits) |

### Safety filtering (backtesting) {#safety-filtering}

Set `ask-start` to a date well after the LLM's knowledge cutoff to prevent
parametric knowledge leakage. Set `resolution-end` to ensure ground truth.

| Start date | Safe for |
|---|---|
| `2025-03-02` | Gemini (cutoff 2025-01-31) |
| `2025-10-26` | All models (cutoff ≤ 2025-08-31) |

### Examples {#exam-examples}

**AIBQ2** — all 113 questions, alphabetical order (`data/exams/aibq2-all/mixture.json`):
```json
{ "seed": null, "select": { "aibq2": "all" } }
```

**Tranche A** — single forecast date, 100 market + 100 dataset (`data/exams/tranche-a/mixture.json`):
```json
{
    "ask-start": "2025-10-26",
    "ask-end": "2025-10-26",
    "resolution-end": "2026-04-10",
    "seed": 42,
    "select": {
        "infer": "all", "manifold": "all", "metaculus": "all", "polymarket": "all",
        "acled": 20, "dbnomics": 20, "fred": 20, "wikipedia": 20, "yfinance": 20
    }
}
```

### Build {#build-exams}

```bash
python3 src/data/make_exam.py --name aibq2-all
python3 src/data/make_exam.py --name tranche-a
```

`make_exam.py` generates `indices.json`, `meta.json`, and data visualizations.
To regenerate the plots separately (e.g., after adding tags), use:

```bash
python3 src/data/plot_exams.py --name aibq2-all
python3 src/data/plot_exams.py --name tranche-a1 --name tranche-b1
```

The following plots are produced:

**Forecast horizon histogram** (`horizon_histogram.png`):
Shows the distribution of forecast horizons (resolution date minus forecast
date, in days), colored by outcome. The right panel shows the per-source
question count with outcome balance.

![horizon histogram example](../data/exams/tranche-a/horizon_histogram.png)

*Example: tranche-a (n=200 questions, 398 resolution dates). Most questions
resolve within 2 weeks or ~3 months. Dataset sources are balanced; market
sources skew negative (29% positive overall).*

**Resolution date vs forecast date scatter** (`rdate_by_fdate_scatter.png`):
Shows the forecast horizon as a 2D scatter (useful for multi-date exams).
Points above the diagonal have longer horizons. Color indicates outcome.

**Source × category heatmap** (`tag_{version}_distribution.png`):
Shows how questions are distributed across sources and categories.
Auto-generated when LLM-classified tags are present
(run `classify_questions.py --version {version}` first).

## 4. Configure the agent {#configure-agent}

The agent is parameterized by an `AgentConfig` dataclass. There are
three interchangeable ways to specify one to `predict.py` and
`fb_compete.py`:

- **Delta string** (shorthand): `"pro/thk:high/crowd:1/tools:1"` —
  starts from the default config and applies the listed overrides.
  Resolves to a directory name like `pro-high-brave-c1-t1`.
- **Directory name**: `"pro-high-brave-c1-t1"` — refers to an existing
  config directory.
- **Config file**: `--config-file experiments/configs/sota.json` — a
  full AgentConfig JSON. Its `name` field becomes the config-directory
  name. Useful for pinning the exact submission config across
  competition rounds. An example `experiments/configs/sota.json` is
  checked in.

Every run also writes `config.json` next to the forecasts
(`experiments/forecasts_raw/{config}/config.json`), and the collated
file in `experiments/forecasts_final/{date}/{config}.json` stores the
full AgentConfig dict inline under its `config` key — both serve as
documentation of exactly what was run.

Configs are specified as **delta strings** relative to a default template,
not as JSON files. The delta format uses `/` to separate fields:

```
flash/thk:high/crowd:1/tools:0
pro/thk:med/search:none/crowd:0
```

| Delta key | Config field | Short values | Description |
|---|---|---|---|
| (bare) | `llm` | flash, pro, sonnet, opus, grok4, ... | LLM model |
| `thk` | `reasoning_effort` | none, low, med, high | Thinking budget |
| `search` | `search_engine` | brave, serper, pplx, none | Web search provider |
| `crowd` | `show_crowd` | 0, 1 | Include crowd/market estimate in prompt |
| `tools` | `use_tools` | 0, 1 | Enable source-specific data tools |
| `steps` | `max_steps` | integer | Max agent loop iterations |
| `timeout` | `question_timeout` | integer | Seconds before forced submit |

**`show_crowd`**: When 1, the question prompt includes the market price or
crowd estimate (if available in the question data). For market sources this
is the prediction market probability; for dataset sources it's the most recent
known value of the time series. When 0, this information is hidden.

**`use_tools`**: When 1, source-specific data-fetching tools are available
to the agent alongside web search. When 0, only web search + URL lookup are
available. All tool queries are date-filtered: the `end_date` parameter is
clamped to the knowledge cutoff (= forecast_due_date) to prevent data leakage.
The start date (how far back to fetch) depends on the source.

| Source | Tool | Description |
|---|---|---|
| yfinance | `fetch_ts_yfinance` | Historical stock prices (1 year of daily data) |
| fred | `fetch_ts_fred` | FRED economic data series (full history) |
| dbnomics | `fetch_ts_dbnomics` | DBnomics time series (full history) |
| wikipedia | `fetch_wikipedia_toc`, `fetch_wikipedia_section` | Wikipedia page as of cutoff date |
| polymarket | `fetch_polymarket_info` | Market probability history (up to 90 days) |
| manifold | `fetch_manifold_info` | Market probability history (up to 90 days) |
| metaculus, infer, acled, aibq2 | *(none)* | Web search only |

The default config (`DEFAULT_CONFIG` in `src/config/config.py`) uses:
flash, high thinking, brave search, crowd=1, tools=1, max_steps=10,
timeout=300, agg_method=std-shrinkage.

Legacy config JSON files in `experiments/configs/` are still supported.

### Question-specific policy {#qtype-policy}

The agent adapts its behavior (policy) per question depending on the
question's `Qtype` (determined by `src/config/tags.py`). The Qtype affects the
prompt (source-specific tool hints) and available tools.
The agent ultimately decides which tools to call —
it could call only web_search, only source tools, or a mix. All Qtypes
have access to web search (unless `search_engine="none"`).

| Qtype | Sources | Source tools |
|---|---|---|
| `timeseries` | yfinance, fred, dbnomics | fetch_ts_* (combo tool: history + model estimate) |
| `wikipedia` | wikipedia | fetch_wikipedia_toc, fetch_wikipedia_section |
| `acled` | acled | *(none — web search only)* |
| `market` | polymarket, manifold, metaculus, infer | fetch_market_history (polymarket, manifold) |

> **Note**: Per-Qtype `max_steps` and `ntrials` overrides are defined in
> `_QTYPE_MAX_STEPS` and `_QTYPE_NTRIALS` in `predict.py`, but are
> **currently disabled** (commented out) for fair ablation comparison.
> All questions use the same `max_steps` and `ntrials` from the config.
> DBnomics is the one exception: it bypasses the LLM entirely and uses
> the KNN harmonic model directly (`_dbnomics_harmonic_forecast`).

### Multi-resolution dates {#multi-resolution}

For dataset questions with multiple resolution dates (e.g. "will this value
increase by date X, Y, Z?"), the agent submits a list of probabilities —
one per resolution date — with increasing uncertainty for more distant dates.
The prompt instructs:

```
## Resolution dates
2025-11-02, 2025-11-25, 2026-01-24

You must submit **3 probabilities** (one per resolution date) when you call submit.
Your uncertainty should INCREASE with forecast horizon.
```

### Prompt examples {#prompt-examples}

The system prompt is the same for all question types. It instructs the agent
to work in a tool-use loop: gather evidence, update beliefs, then submit.

The question prompt varies based on config flags. Key differences:

**`crowd=1` vs `crowd=0`** — with crowd=1, the prompt includes:
```
## Market estimate
The market probability on 2025-12-07 was 0.45.
```

**`tools=1`** — the prompt adds source-specific hints:
```
For this question, you MUST call `fetch_ts_yfinance` as your FIRST action
to retrieve the stock's price history. The tool output includes a statistical
trend analysis with P(increase) — use this as your PRIMARY probability anchor.
```

### Prompt inspection {#prompt-inspect}

Inspect the exact prompts that will be sent to the LLM for a given xid
or question:

```bash
# All configs × one question per source
python3 src/misc/show_prompts.py --xid xid-tranche-a1
open experiments/generated_prompts/xid-tranche-a1.html

# All prompt variants for a specific question
python3 src/misc/show_prompts.py --source polymarket --id 0x310c3d... --fdd 2025-10-26
open experiments/generated_prompts/polymarket_0x310c3d....html
```

The XID mode shows each config as a collapsible section with system prompt
and per-source question prompts. Source-specific differences (tool lists)
are highlighted. The question mode sweeps over crowd, tools, search,
nobelief, and live/backtest.

### Prompt verification {#prompt-test}

After changing configs or prompts, verify that system and question prompts
are consistent with config settings (tools match schema, crowd info
present iff `show_crowd=1`, belief state instructions match `nobelief`, etc.):

```bash
python3 src/testing/test_prompts.py --exam smoke2           # quick check (9 sources)
python3 src/testing/test_prompts.py --exam smoke2 --verbose  # show each check
```

This tests all 11 ablation configs (A–J) × all available sources,
checking 7 categories of invariants per combination.

## 5. Create experiment (xid) {#create-xid}

An xid ties together an exam, configs, metrics, and groupings.
Create as `experiments/xids/{name}.json`.

| Field | Description |
|---|---|
| `exam` | Exam name (required) |
| `config` | List of config names (hyphenated directory names) |
| `metrics` | List of metrics (default: `["brier-index", "adjusted-brier-index", "metaculus-score"]`) |
| `groups` | List of tag spaces for grouping (default: `["FBQtype"]`) |
| `manual_reference` | List of hard-coded reference scores to display on leaderboard/plots (e.g. `["sota"]`). Point estimates only — defined in `eval.py:REFERENCE_SCORES`. |
| `fb_reference` | List of ForecastBench method keys to auto-import and include in eval (e.g. `["external.Google DeepMind.2"]`). Use `fb_leaderboard.py --xid ...` to discover available method keys. Data is cached locally; delete `data/fb_cache/forecastbench-processed-forecast-sets/` to refresh. |

**`groups`** controls two things:
1. **Plots**: For each tag space in the list, eval generates a composite bar
   chart showing per-tag-value metric scores (e.g. grouping by `FBQtype` shows
   market vs dataset; by `Qsource` shows per-source breakdown).
2. **Leaderboard columns**: The first tag space determines the leaderboard
   columns. Each unique tag value becomes a column, plus an "overall" column
   that averages across the displayed groups (equal-weighted).

**`reference`** adds external baseline scores (from `REFERENCE_SCORES` in
`eval.py`) to the leaderboard and metric plots. Available references:
`sota`, `superhuman` (both from the ForecastBench leaderboard).
References are matched by exam name prefix and metric.

Optional overrides: `plot_groups`, `leaderboard_groups` (if you want
different groupings for plots vs leaderboard).

```json
{
    "exam": "tranche-a",
    "config": ["flash-high-brave-c1-t0", "pro-high-brave-c1-t0"],
    "metrics": ["brier-index", "metaculus-score"],
    "groups": ["FBQtype", "Qsource"],
    "manual_reference": ["sota"]
}
```

## 6. Run experiment (predict) {#predict}

```bash
caffeinate -s python3 src/core/predict.py --xid my-xid --ntrials 5 --verbose --monitor
caffeinate -s python3 src/core/predict.py --xid my-xid --ntrials 1 --verbose
```

Per-trial results: `experiments/forecasts/{config}/trial_{t}/{source}/{id}.json`.
Aggregated forecasts: `experiments/forecasts/{config}/{source}/{id}.json`.

When `ntrials > 1`, each trial runs the agent independently, then the trials
are aggregated into a single forecast using the method specified by
`config.agg_method` (default: `"std-shrinkage"`). The aggregation uses
logit-space James-Stein shrinkage: when trials disagree (high cross-trial
std), the forecast is pulled toward 0.5. With `"plain-mean"`, the forecast
is the simple average. See `docs/shrinkage.tex` for the derivation.

### Forecast output layout: `forecasts_raw/` vs `forecasts_final/` {#forecasts-layout}

Two parallel directories, with two different roles and two different
file-count regimes:

| Dir | Unit | Written by | Size | Gitignored? |
|---|---|---|---|---|
| `experiments/forecasts_raw/{config}/{source}/{id}.json` | one file per (config, source, question), with per-trial sub-dirs | `predict.py` (parallel, in-place) | ~5–30 KB/file, hundreds of thousands of files | **Yes** |
| `experiments/forecasts_final/{YYYY-MM-DD}/{config}.json` | one file per (forecast_due_date, config), FB-tarball shape | `collate.py`, `aggregate.py`, `calibrate.py` | ~50–500 KB/file, ~100–1000 files total | **No** — checked in |

Everything `eval.py` reads for leaderboard/plots/calibration comes from
`forecasts_final/`. The dashboard's per-question trace links point
into `forecasts_raw/` (which is the only place that has the belief
history, tool log, and prompts needed to render the trace).

External FB leaderboard methods live next to our configs as
`experiments/forecasts_final/{date}/fb-{key}.json`, using the native FB
tarball shape. They are imported by `fb_leaderboard.py --import-method`,
which reads from the processed-forecast tarball cached under
`data/fb_cache/`.

#### Per-date file shape

Matches ForecastBench's own published tarball:

```json
{
  "organization":        "sirbayes",
  "model":               "pro-high-brave-c1-t1",
  "model_organization":  "sirbayes",
  "forecast_due_date":   "2025-10-26",
  "leaderboard_eligible": true,
  "config":              { ... full AgentConfig dump ... },
  "forecasts": [
    { "id": "...", "source": "polymarket", "resolution_date": "2026-02-01",
      "forecast": 0.72,
      "raw_trials": [0.70, 0.75, 0.68, 0.72, 0.74],
      "resolved_to": 1,
      "reasoning": "...", "n_steps": 7, "submitted": true,
      "tokens_in": 23145, "tokens_out": 891, "elapsed_seconds": 87.3,
      "tool_counts": {...}, "n_searches": 5,
      "market_value": 0.72, "market_date": "2025-10-25",
      "forecasts_aggregated": { "mean:5": 0.73, "shrink5-loo": 0.68 },
      "forecasts_calibrated": { "global-cal": 0.58, "hier-cal": 0.61 }
    },
    ...
  ]
}
```

Multi-resolution dataset questions contribute **one entry per
`resolution_date`** (the same convention FB uses), so a 3-date acled
question appears three times with the same `id` and different
`resolution_date` / `forecast` / `resolved_to`.

`forecast` is the **default aggregation** — arithmetic mean of trials
for legacy v7 runs, logit-mean of trials for new runs. Further
aggregation variants (computed by `aggregate.py`) land in
`forecasts_aggregated: {"mean:1": 0.71, "shrink5-loo": 0.65, ...}`.
Calibrated versions (computed by `calibrate.py`) land in
`forecasts_calibrated: {"global-cal": 0.58, "hier-cal": 0.61, ...}`.
Both are plain dicts mapping variant-key → scalar forecast, replacing
the old approach of materializing separate `{config}_calibrated_global/`
directories.

#### Collate step

After `predict.py` finishes, collate the per-question raw files into
per-date payloads:

```bash
python3 src/core/collate.py --xid my-xid                 # every config in the xid
python3 src/core/collate.py --config pro-high-brave-c1-t1
python3 src/core/collate.py --all                        # every config under forecasts_raw/
python3 src/core/collate.py --from-raw-root /path/to/other-tree/experiments/forecasts_raw --all
```

Collate populates `forecast` with the default aggregation and
`raw_trials` with the per-trial probability vector.

### Aggregation variants (optional) {#aggregation}

`aggregate.py` adds `forecasts_aggregated: {"mean:1": ..., "mean:5":
..., "shrink5-loo": ...}` to each entry in the collated files, without
touching the default `forecast` value.

```bash
python3 src/core/aggregate.py --xid my-xid
python3 src/core/aggregate.py --xid my-xid --variants "mean:1,mean:5,shrink5-loo"
```

LOO shrinkage (`shrink5-loo`) fits the shrinkage strength λ by leave-one-
out CV, which requires labeled outcomes — not usable at live-submission
time in `fb_compete.py`. Use `mean:N` or `logit-mean:N` for live runs.

To pull variants into leaderboard columns, three paths with descending
precedence:

1. **`--add-calibration [KEYS]` / `--add-aggregation [KEYS]` on the
   command line.** No value = auto-discover every key available in the
   collated files; a comma list = only those keys.
   `python3 src/core/eval.py --xid my-xid --add-calibration global-cal,hier-cal --add-aggregation mean:5`
2. **Xid fields `eval_calibration` / `eval_aggregation`.** Persistent
   lists (or the literal `"*"` for auto-discover):
   ```json
   { "eval_calibration": ["global-cal", "hier-cal"],
     "eval_aggregation": ["mean:5", "shrink5-loo"] }
   ```
3. **Explicit bracket entry in `eval`**, same as a regular config:
   `"eval": ["pro-high-brave-c1-t1[mean:5]"]`.

Per-method plots (scatter, ntrials, per-question heatmaps) can explode
when you add many variants. Gate them with **`--plot-variants
NAME,NAME,…`** or the xid field **`plot_variants`**; the leaderboard
itself still includes every variant you asked for.

## 7. Evaluate experiment {#evaluate}

End-to-end pipeline, run step by step (there is no wrapper script):

```bash
# 1. Predict — one agent run per question × trial. Parallel, in-place writes.
caffeinate -s python3 src/core/predict.py --xid my-xid --ntrials 5 --verbose --monitor
#    → experiments/forecasts_raw/{config}/trial_{t}/{source}/{id}.json  (per-trial)
#    → experiments/forecasts_raw/{config}/{source}/{id}.json            (aggregated)

# 2. (Optional) Sanity-check forecast JSONs before running eval
python3 src/testing/test_smoke.py --xid my-xid --verbose

# 3. Collate — merge raw per-question files into per-date FB-style payloads.
#    Sets `forecast` to the default aggregation (mean of trials) and stores
#    per-trial probabilities in `raw_trials` for later post-processing.
python3 src/core/collate.py --xid my-xid
#    → experiments/forecasts_final/{date}/{config}.json

# 4. (Optional) Extra aggregation variants — mean:1, mean:5, shrink5-loo, ...
#    Writes a `forecasts_aggregated: {<key>: [p0, p1, ...]}` dict at the
#    file level (arrays aligned with forecasts[]).
python3 src/core/aggregate.py --xid my-xid

# 5. (Optional) Platt calibration — global and/or hierarchical (per-Qsource).
#    Writes a `forecasts_calibrated: {"global-cal": [...], "hier-cal": [...]}`
#    dict at the file level. Fitting is exam-scoped (only labeled entries
#    matching the xid's exam questions are used for CV, keeping LOO tractable).
python3 src/core/calibrate.py --xid my-xid --cv loo
python3 src/core/calibrate.py --xid my-xid --cv loo --hierarchical Qsource
# --save-model NAME also stores the fitted model under experiments/
# calibration_models/{NAME}/{config}.json for reuse in compete.py.
python3 src/core/calibrate.py --xid my-xid --cv loo --save-model tranche-a1

# 6. Evaluate — leaderboard, plots, calibration curves, per-question traces.
python3 src/core/eval.py --xid my-xid                         # default (base forecast)
python3 src/core/eval.py --xid my-xid --add-calibration       # + calibrated variants
python3 src/core/eval.py --xid my-xid --add-aggregation       # + aggregation variants
python3 src/core/eval.py --xid my-xid --fast                  # leaderboard only, no plots
#    → experiments/eval/{xid}/leaderboard.html, dashboard.html, figs/
```

Steps 2, 4, and 5 are optional; the minimum sequence is **predict → collate → eval**.
The `--fast` flag on step 6 skips plot and trace generation; use it for quick
iteration and omit it for the final run.

### Output files {#eval-outputs}

**`experiments/eval/{xid}/`** — prediction-dependent outputs:

- `leaderboard.html` — grouped leaderboard with metrics × tag-group columns
- `dashboard.html` — per-question table with Brier-colored cells; click to view trace
- `figs/metric_by_method/`
  - `{metric}_vs_methods.png` — dot plot with bootstrap CI, one point per config
  - `{metric}_relative_vs_methods.png` — score relative to best config per question
- `figs/metric_by_tag/{label_space}/`
  - `{metric}_composite.png` — grouped horizontal bars per tag value, one per config
- `figs/metric_by_horizon/`
  - `{metric}_vs_horizon_composite.png` — performance by forecast horizon bin
- `figs/metric_by_time/`
  - `{metric}_by_time.png` — metric vs forecast_due_date with knowledge cutoff lines
  - `{metric}_histos_{config}.png` — score distribution: all vs first-half vs second-half
- `figs/calibration/`
  - `calibration_curves.png` — reliability diagrams (all, raw-only, calibrated-only)
  - `ece_histo.png` — expected calibration error bar chart
- `figs/metric_by_question_heatmaps/`
  - `{metric}_heatmap_{source}.png` — per-question × per-config score heatmap
- `figs/std_scatter/`
  - `{metric}_vs_std_scatter_{config}.png` — cross-trial std vs metric score

### Metrics

For binary outcomes `o ∈ {0, 1}` and forecast probability `p`:

| Metric | Formula | Range | Interpretation |
|---|---|---|---|
| **Brier Score** | `BS_j = (p_j - o_j)²`, then mean over questions | 0–1 | Lower is better. Always-0.5 scores 0.25. |
| **[Brier Index](https://forecastingresearch.substack.com/p/introducing-the-brier-index)** | `BI = 100 · (1 − √(mean BS))` — population-level (not a per-question mean) | 0–100 | Higher is better. Always-0.5 = 50, perfect = 100. |
| **Metaculus Score** | `S_j = 100(1 + log₂(q_j))` where `q_j = p_j` if `o_j=1` else `1-p_j`, then mean over questions | −∞ to 100 | Higher is better. Always-0.5 = 0, perfect = 100. |

Note: BI is computed on the *square-root of the mean Brier score*, not by
averaging per-question `1 − √BS_j` values — so it is a population metric
rather than an arithmetic mean of per-question scores.

**Difficulty-adjusted metrics** (from the
[ForecastBench methodology](https://www.forecastbench.org/assets/pdfs/forecastbench_updated_methodology.pdf)):

The adjusted version subtracts per-question difficulty effects and rescales so
Always-0.5 still scores 0.25. For market questions: `γ_j = (market_value - outcome)²`.
For dataset questions: from
[precomputed ForecastBench fixed effects](https://www.forecastbench.org/datasets/question-fixed-effects/)
or estimated via alternating projections.

The "overall" column for adjusted metrics uses **equal-weighted group means**
(matching ForecastBench methodology).

## 8. Calibrate (optional) {#calibrate}

Platt scaling calibration adjusts forecast probabilities to be better calibrated.

```bash
# LOO cross-validation (honest eval on same data)
python3 src/core/calibrate.py --xid my-xid --cv loo

# Save model for later use on test/live data
python3 src/core/calibrate.py --xid my-xid --cv loo --save-model my-model

# Apply a saved model (no labels needed)
python3 src/core/calibrate.py --xid my-test-xid --apply-model my-model
```

Models saved to `experiments/calibration_models/{name}/{config}.json`.

Re-evaluate with calibrated configs:
```bash
python3 src/core/eval.py --xid my-xid --add-calibration
```

![Calibration example](calibration_example.png)

*Example calibration curves (raw, before Platt scaling). Points near the
diagonal indicate good calibration. Quantile-based bins ensure equal samples
per point.*

## 9. Ensemble (optional) {#ensemble}

Greedy forward selection of up to K configs that minimize ensemble Brier
score. At each step, the member whose addition most improves the average
is selected; the process stops early if adding the next member would
increase the score.

```bash
# Auto-discover all configs with forecasts for this exam
python3 src/core/ensemble.py --exam tranche-a --out my-ens

# Specify candidates explicitly
python3 src/core/ensemble.py --exam tranche-a \
    --candidates flash-high-brave-c1-t1,pro-high-brave-c1-t1 \
    --k 3 --out my-ens

# With Platt calibration of the ensemble
python3 src/core/ensemble.py --exam tranche-a \
    --out my-ens --calibrate --cv loo
```

The ensemble forecast for each question is the simple average of the
selected members' forecasts. Output is written to
`experiments/forecasts/{out}/{source}/{id}.json`.

The ensemble definition (which members were selected, the exam, and
the selection order) is saved to `experiments/ensembles/{out}.json`.

To evaluate, add the ensemble config name to the xid or pass it via CLI:
```bash
python3 src/core/eval.py --xid my-xid --add-ensemble my-ens
```

> **Note:** Ensemble selection uses the same data it's evaluated on
> (training set). For honest evaluation, select the ensemble on a train
> exam and evaluate on a held-out test exam with the same seed.

## 10. ForecastBench live submission {#compete}

`src/compete/fb_compete.py` runs the end-to-end live-competition pipeline:
fetch the live question set, build an exam, run the agent, assemble the
submission JSON, and upload to GCS.

Pipeline steps (fetch → exam → predict → collate → calibrate → submit → upload):

```bash
# With a saved Platt calibration model (the usual case)
caffeinate -s python3 src/compete/fb_compete.py --date 2026-04-12 \
    --config "pro/thk:high/crowd:1/tools:1" --ntrials 5 \
    --calibration-model tranche-a1

# Without calibration (raw forecasts)
caffeinate -s python3 src/compete/fb_compete.py --date 2026-04-12 \
    --config "pro/thk:high/crowd:1/tools:1" --ntrials 5

# Re-assemble a submission from the existing collated file (no re-run)
python3 src/compete/fb_compete.py --date 2026-04-12 --skip-predict \
    --config pro-high-brave-c1-t1 --calibration-model tranche-a1
```

Writes (relative to repo root):

| Path | Contents |
|---|---|
| `experiments/exams/{date}/` | Live exam definition (materialized from the FB GitHub release) |
| `experiments/forecasts_raw/{config}/` | Raw agent outputs — per-trial + aggregated |
| `experiments/forecasts_final/{date}/{config}.json` | Collated FB-shape file; calibration (if any) is appended as `forecasts_calibrated["global-cal"]` on this file |
| `submissions/{date}.{org}.{N}.json` | Final submission JSON (derived from the collated file), uploaded to GCS |

The saved calibration models live in `experiments/calibration_models/{exam}/`
and are pre-trained on backtesting data (e.g. `tranche-a1` spans
2025-10-26 through the most recent resolved fortnight). These are
git-tracked so you can submit without re-running the training pipeline.
At submit time, `compete.py` calls `calibrate.apply_saved_model(...)`
which writes `forecasts_calibrated["global-cal"]` (or the key given by
`--cal-key`) into the collated file, and the assembler pulls that column
into the submission payload.

The `submissions/` directory is created on first run and its contents
are kept in git as an audit trail of what was uploaded.

---

# Directory layout

```
data/
    questions/{source}/{id}.json        Downloaded question files
    exams/{name}/mixture.json           Exam definitions
    exams/{name}/indices.json           Materialized exam indices
    tags_{version}/{source}/{id}.json   LLM-classified tags
    fb_cache/                           Cached ForecastBench downloads
experiments/
    xids/{name}.json                    Experiment definitions
    forecasts/{config}/                 Prediction outputs
    eval/{xid}/                         Evaluation outputs
    calibration_models/{name}/          Saved calibration models
src/
    core/                               Main pipeline entry points
        predict.py                      Main prediction pipeline
        eval.py                         Evaluation orchestrator
        calibrate.py                    Platt scaling calibration
        ensemble.py                     Greedy ensemble selection
        aggregate.py                    Trial aggregation
    agent/                              Agent loop + tools
        agent.py                        Agent loop (LLM + tools)
        prompts.py                      System + question prompts
        tools.py                        Tool dispatch (search, submit, etc.)
        source_tools.py                 Source-specific tools (yfinance, etc.)
        data_tools.py                   Data fetching (FRED, yfinance, etc.)
        llm_client.py                   LLM API wrapper
    config/                             Configuration
        config.py                       Config defaults + delta parsing
        tags.py                         Unified tag system
        paths.py                        Centralized directory paths
    data/                               Data download + exam creation
    eval/                               Evaluation plots + HTML
    analysis/                           Analysis scripts
    search/                             Web search providers
    compete/                            Live competition submission
    testing/                            Tests and validation
    misc/                               Legacy / one-off scripts
```

---

# Visualization Tools

The evaluation pipeline (`eval.py` + `eval_plots.py`) produces numerous
visualization types. Each can also be generated standalone using the
paper figure scripts in `docs/nips26/`.

## Eval-generated plots (automatic)

Running `python3 src/core/eval.py --xid my-xid` produces these in
`experiments/eval/{xid}/figs/`:

### Leaderboard table
- **File**: `leaderboard.html`
- **What**: HTML table ranking all configs by metric, with per-group breakdowns
- **Script**: `eval.py` (built-in)

![Leaderboard example](leaderboard_example.png)

*Example leaderboard with per-group columns and cost tracking.*

### Trace pages
- **File**: `{config}/{source}/{qid}_trace.html`
- **What**: Per-question HTML trace showing belief evolution, tool calls, and search results.

![Trace example](trace_example.png)

*Example trace page showing belief evolution and tool calls.*

### Per-question scores with error bars
- **File**: `metric_by_question_boxplots/{metric}_vs_que_num_{config}.png`
- **What**: Each question's score (dot) with cross-trial CI (error bar), sorted by mean score. Shows E[single trial] and E[averaged] reference lines.
- **Variant**: `_vs_que_str_` version includes question text as x-axis labels.
- **Script**: `eval_plots.py:generate_metric_vs_questions()`
- **Example**: ![](../nips26/figs/aibq2_ms_per_question.png)

### Score heatmaps (questions × methods)
- **File**: `metric_by_question_heatmaps/{metric}_heatmap_{source}.png`
- **What**: Heatmap of per-question scores across methods, sorted by difficulty.
- **Script**: `eval_plots.py:generate_heatmap()`

### Metric vs number of trials
- **File**: `ntrials/{metric}_vs_ntrials_{config}.png`
- **What**: How metric improves with K=1..5 trials, comparing plain mean vs std-shrinkage.
- **Script**: `eval_plots.py:generate_metric_vs_ntrials()`

### Metric vs forecast std (scatter)
- **File**: `std_scatter/{metric}_vs_std_scatter_{config}.png`
- **What**: Per-question metric vs cross-trial forecast standard deviation.
- **Script**: `eval_plots.py:generate_metric_vs_std_scatter()`

### Calibration curves
- **File**: `calibration/calibration_curves.png`
- **What**: Reliability diagrams for all configs, with ECE values.
- **Script**: `eval_plots.py:generate_calibration_curves()`

### ECE histogram
- **File**: `calibration/ece_histo.png`
- **What**: Bar chart of ECE values across configs.
- **Script**: `eval_plots.py:generate_ece_histogram()`

### Metric by category (tag)
- **File**: `metric_by_tag/{tag_version}/{metric}_composite.png`
- **What**: Per-category breakdown of scores (e.g., by kevin tag categories).
- **Script**: `eval_plots.py:generate_metric_vs_category_composite()`

### Metric by source
- **File**: `metric_by_tag/Qsource/{metric}_composite.png`
- **What**: Per-source breakdown of scores.
- **Script**: `eval_plots.py:generate_metric_vs_source_composite()`

### Metric by time/horizon
- **File**: `metric_by_time/{metric}_by_time.png`
- **What**: How scores vary with forecast horizon.
- **Script**: `eval_plots.py:generate_metric_vs_horizon()`

### Tool usage histogram
- **File**: `tool_histos/tool_histo_{config}.png`
- **What**: Distribution of tool calls per step across questions.
- **Script**: `eval_plots.py:generate_tool_histogram()`

### Steps histogram / heatmap
- **File**: `metric_by_question_heatmaps/steps_heatmap_{source}.png`
- **What**: Number of agent steps per question.
- **Script**: `eval_plots.py:generate_steps_heatmap()`

## Paper figure scripts

See [`docs/nips26/figs/make_figs.md`](../nips26/figs/make_figs.md)
for documentation of all paper figure generation scripts.
