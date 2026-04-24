# TODO

Tracked items for the paper and codebase. Check off when done.

## Paper (docs/nips26/main.tex)
- [ ] Write intro section
- [ ] Write related work section
- [x] Write system description (agent loop, belief state, tools, search)
- [x] Write conclusion (draft)
- [x] Fill in main results table
- [x] Fill in mixed effects table
- [x] Fill in leakage audit confusion matrix
- [x] Create BI by source composite figure
- [x] Describe Qtype adaptive policy
- [ ] Report backtesting-live rank correlation
- [ ] Create belief trace figure (select good example question)
- [ ] Create agent trace figure (select interesting question)
- [ ] Add calibration curves figure
- [ ] Write ACLED and Wikipedia appendix sections
- [ ] Add ABI to Table 5 (currently commented out; GPT-5 leads)

## Code improvements
- [ ] Allow agent to choose starting date for fetch_ts tools
- [ ] Show injected per-step messages in trace HTML files
- [ ] Create question_analyzer (find templates per source)
- [ ] Add support for https://prophetarena.co/

## Experiments
- [ ] Re-run BLF pipeline with GPT-5 and Grok-4 as base LLM (may improve ABI)
- [ ] Implement per-series FRED policies (random-walk→p=0.5, trending→momentum)
- [ ] Make tool output configurable per source: H_q only, p_q only, or H_q∪p_q (currently hardcoded)
- [ ] ACLED question-type bypass: if 10x-spike predict ~0, if any-increase predict ~0.2 (bypass LLM like dbnomics)
- [ ] Wikipedia bypass: vaccine questions→~0, world records→~1, chess→use trend
- [x] Run tranche-a1, a2 (done)
- [x] Run tranche-b1, b2 (done)
- [x] Run AIBQ2 ablations (done)
- [x] Run aggregation variants (mean:1, mean:5, shrink:5)
- [x] Run hierarchical calibration
- [x] Run leak detective (search + reasoning traces)
- [x] Run mixed effects analysis
- [x] Cache data tool lookups (fetch_ts_*) to disk
- [ ] Fetch more history for seasonal analysis (5+ years)
- [ ] Better statistical models for time-series forecasting
- [ ] Perturbation analysis (force belief state)
- [ ] Negation consistency check (ask both p and 1-p)
