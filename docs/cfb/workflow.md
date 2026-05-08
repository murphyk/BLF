# CFB workflow

The Continual Forecasting Benchmark (CFB) is an online backtesting environment
for binary forecasting agents. It simulates the asynchronous-reward setup of a
prediction market: each day the agent observes a fresh batch of questions,
emits forecasts, and later receives outcomes once questions resolve.

This document describes the code layout and the typical command sequences.
For the math/notation see `docs/cfb/main.tex`.

## Layout

```
src/cfb/
├── schema.py            Question, Resolution, PoolEntry  (frozen dataclasses)
├── pool.py              build_pool / freeze / load_pool — flatten multi-res,
                         drop r > t_max, optional dedupe-base + cap-per-source
├── env.py               Env: observation Q(t), reward R(t), event_days() iter
├── evaluator.py         Evaluator: forecast log + running Brier
├── score.py             event-weighted + source-weighted Brier / Brier index
├── post_process.py      apply shrink / platt to a cached run trajectory
├── run_base_agent.py    do one base run (LLM or non-LLM) and write a trajectory
├── run.py               USER-FACING front-end: agent strings -> chained xids
├── plot_pool.py         per-pool figures: |Q(t)|, |R(t)|, swim-lane lifelines
├── plot_continual.py    reward curves: cumulative + rolling Brier index
└── agents/
    ├── constant.py             ConstantAgent
    ├── empirical_prior.py      EmpiricalPriorAgent  (global running mean)
    ├── market_value.py         MarketValueAgent     (predict crowd as-is)
    ├── flash_zs.py             FlashZSAgent + Halawi prompt + crowd flag
    ├── icl.py                  ICLAgent — frozen LLM + FIFO of (q, o) pairs
    ├── online_platt.py         2-param recursive logistic calibrator
    ├── online_shrinker.py      per-source shrink-toward-prior with online α
    ├── platt_wrapper.py        composes a base agent with Platt
    └── _llm.py                 thin litellm wrapper used by LLM agents

data/cfb/
├── pool-<sha>.jsonl     frozen pools — one PoolEntry per line
├── pool-<sha>.meta.json build params + entry count
└── runs/<xid>/
    ├── trajectory.jsonl event-by-event chronological log (one row per
                         resolved event, includes p, p_raw, p_crowd, o)
    └── manifest.json    agent / params / pool reference

docs/cfb/
├── workflow.md          this file
└── main.tex             paper draft
```

`src/cfb/**` does not import from `src/agent/**` or anywhere else BLF-specific —
it is self-contained for an eventual repo split.

## Agent string grammar

The user-facing CLI takes string-form configs:

```
agent_string := base ("+" layer)*
base         := atomic | llm
atomic       := "constant" | "market-value" | "empirical"
llm          := <name> "-zs" ["-icl"] ["-c1"]
layer        := "shrink" | "platt"
```

LLM names map to litellm models in `run_base_agent.LLM_MAP`:
`flash, pro, sonnet, opus, haiku, gpt5`.

- `-c1` injects the market freeze value into the prompt (BLF's c=1). Absent ⇒ c=0.
- `-icl` wraps the LLM in an ICL memory of past `(question, outcome)` pairs.
- `+shrink` post-hoc mixes the base forecast with the freeze value via a per-source
  online α (cold-start at α=0, ridge-regularised).
- `+platt` post-hoc applies an online 2-parameter logistic calibration.

Examples:

```
constant
empirical
market-value
flash-zs
flash-zs-c1
flash-zs-icl
flash-zs-icl-c1
flash-zs-c1+shrink
flash-zs-icl+platt
flash-zs-c1+shrink+platt
empirical+platt
```

Each prefix of a string is a separate xid that gets cached, e.g.
`flash-zs-c1+shrink+platt` produces three runs: `flash-zs-c1`,
`flash-zs-c1+shrink`, `flash-zs-c1+shrink+platt`. Re-using a prefix is free.

## Typical commands

### Build the canonical pool

The first run of `run.py` builds + freezes a pool under `data/cfb/`. The default
window is `[2025-10-26, 2026-03-29]`, markets-only, dedupe-base, no cap.

```bash
# Build + run a trivial agent so the pool gets created and cached.
python -m src.cfb.run --agent constant
```

Output: `data/cfb/pool-<sha>.jsonl` + `data/cfb/runs/constant/`.

To rebuild with the dataset sources back in:

```bash
python -m src.cfb.run --agent constant --use-data-and-market \
                      --cap-per-source 150
```

### Run agents

```bash
# One agent
python -m src.cfb.run --agent flash-zs

# Several in parallel
python -m src.cfb.run \
    --agent empirical flash-zs flash-zs-icl flash-zs-icl+platt

# Force a re-run (otherwise cached steps are skipped)
python -m src.cfb.run --agent flash-zs --force
```

Each agent string is parsed into a sequence of steps; cached steps are
skipped, so iterating on a post-processing layer never re-pays the LLM cost.

### Score a cached run

```bash
python -m src.cfb.score flash-zs-icl
# or with an explicit path
python -m src.cfb.score data/cfb/runs/flash-zs-icl/trajectory.jsonl
```

Reports both event-weighted Brier (sum / N) and source-weighted Brier (mean
of per-source means), plus the corresponding Brier indices.

### Plot reward curves

```bash
python -m src.cfb.plot_continual \
  --traj "empirical:stateless=data/cfb/runs/empirical/trajectory.jsonl" \
  --traj "flash-zs:stateless=data/cfb/runs/flash-zs/trajectory.jsonl" \
  --traj "flash-zs-icl=data/cfb/runs/flash-zs-icl/trajectory.jsonl" \
  --out data/cfb/continual.png --window 50 --warmup 30
```

Suffix `:stateless` dashes the line; otherwise solid.

### Plot a pool

```bash
python -m src.cfb.plot_pool --pool data/cfb/pool-<sha>.jsonl --prefix markets_only_
# writes <prefix>q.png, <prefix>r.png, <prefix>lifelines.png next to the pool
```

## Adding a new agent

A CFB agent is anything that exposes:

```python
def act(self, questions: list[Question]) -> dict[str, float]:
    """For each question u in `questions` return p_u in [0, 1]."""

def observe(self,
            questions: list[Question],
            forecasts: dict[str, float],
            resolutions: list[Resolution]) -> None:
    """Optional online update. May be a no-op for stateless agents."""
```

To wire it into the CLI:

1. Drop the implementation in `src/cfb/agents/<my_agent>.py`.
2. Add a branch in `run_base_agent._make_agent`. If it's an LLM agent it
   probably belongs as a new mode for `--agent llm`; otherwise add a new
   atomic name.
3. If it's a new atomic name, add it to the `_ATOMIC` set in `run.py`.

If your agent is a *post-processing layer* (operates on a cached
trajectory rather than calling the env from scratch), put it in
`src/cfb/agents/online_<X>.py` and add a `apply_<X>` function to
`post_process.py`'s `_LAYERS`.

## Caching invariants

- A run directory is identified by its xid (the agent string).
- `data/cfb/runs/<xid>/trajectory.jsonl` is the source of truth for that xid.
- Post-processing layers are pure functions of the upstream trajectory plus
  pool meta, so any chain of layers is reproducible from the base run.
- The shrink layer needs `p_crowd` (the market freeze value) on every row;
  this is written by `run_base_agent.run_base()` so any base trajectory is
  immediately compatible.
- Pool builds are deterministic and content-addressed (`pool-<sha>.jsonl`);
  the manifest of a run records which pool sha it used.
