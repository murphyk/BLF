#!/usr/bin/env python3
"""likelihood_replay.py — Offline ablation of explicit-Bayes belief updates.

Tests whether an explicit per-evidence Bayesian update beats the agent's direct
posterior elicitation, holding the *evidence fixed* (replayed from logs). The
update accumulates per-evidence log-likelihood-ratios (LLRs) in logit space:

    logit(b) = logit(prior) + alpha * sum_t  lambda_t,
    lambda_t = log p(x_t | s=1) - log p(x_t | s=0)   (elicited from an LLM)

and we compare four ways of forming the lambda_t, isolating the two corrections
discussed in app:voi / the Papamarkou (2026) "Bayes-consistent orchestration"
position paper (tempering + dependence-aware pooling):

  direct  : the agent's logged forecast (no new LLM calls; the baseline).
  uncond  : one LLR per search step on p(x_t | s)        — naive, ignores both
            the action/query confound and evidence redundancy.
  qcond   : one LLR per search step on p(x_t | s, q_t)   — de-confounds the
            belief-driven query choice (the query cancels in the ratio).
  pooled  : qcond, but documents are first clustered by underlying claim and
            each cluster contributes ONE LLR — dependence-aware evidence pooling
            (≈ counting effective, not raw, observations).

On top of each we sweep a global tempering coefficient alpha (Papamarkou Eq. 1,
here applied globally and per-source), since alpha<1 is the composite-likelihood
magnitude correction for residual misspecification / correlation.

Everything is replayed from experiments/forecasts_raw/{config}/{source}/ — the
logged tool_log gives the query + search_index per step, and the retrieved text
lives in searches/{stem}/search_{idx}_result_{j}.md. Elicited LLRs and clusters
are cached to disk, so reruns are free. Use --dry-run to see how many (paid)
LLM calls a run would make before committing.

Usage:
    # cost preview (no LLM calls)
    python3 src/analysis/likelihood_replay.py --configs flash-high-brave-c1-t1 \
        --exam tranche-a --n 60 --dry-run
    # real run (flash, cached)
    python3 src/analysis/likelihood_replay.py --configs flash-high-brave-c1-t1 \
        --exam tranche-a --n 60
    # restrict methods / change tempering grid
    python3 src/analysis/likelihood_replay.py --configs ... --methods uncond,qcond \
        --alphas 0.25,0.5,0.75,1.0
"""

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Load API keys from .env (matches the convention in src/search/*.py). The core
# pipeline gets this transitively via its search imports; this standalone
# analysis script imports neither, so it must load .env itself — otherwise LLM
# calls 401 whenever OPENROUTER_API_KEY isn't already exported in the shell.
try:
    import dotenv
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
except ImportError:
    pass

from config.paths import FORECASTS_DIR, EXAMS_DIR

KNOWN_SOURCES = {
    "acled", "dbnomics", "fred", "infer", "manifold", "metaculus",
    "polymarket", "wikipedia", "yfinance", "aibq2",
}
DEFAULT_MODEL = "openrouter/google/gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def logit(p, eps=1e-6):
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def clampp(p, lo, hi):
    return max(lo, min(hi, p))


def brier(p, ys, lo, hi):
    return float(np.mean([(clampp(p, lo, hi) - y) ** 2 for y in ys]))


def ece(preds, outcomes, n_bins=10):
    """Equal-width-bin expected calibration error over (p, y) pairs."""
    preds = np.asarray(preds); outcomes = np.asarray(outcomes)
    if len(preds) == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    e, N = 0.0, len(preds)
    for b in range(n_bins):
        m = (preds >= edges[b]) & (preds < edges[b + 1] if b < n_bins - 1
                                   else preds <= edges[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / N) * abs(outcomes[m].mean() - preds[m].mean())
    return float(e)


def normalize_outcomes(resolved_to):
    if resolved_to is None:
        return None
    vals = resolved_to if isinstance(resolved_to, list) else [resolved_to]
    out = [float(v) for v in vals if v is not None]
    return out or None


def load_exam_stems(exam):
    path = os.path.join(EXAMS_DIR, exam, "indices.json")
    if not os.path.exists(path):
        sys.exit(f"Exam indices not found: {path}")
    stems = set()
    for ids in json.load(open(path)).values():
        stems.update(ids)
    return stems


# ---------------------------------------------------------------------------
# data loading: per-question evidence (query + retrieved docs) from logs
# ---------------------------------------------------------------------------

def read_search_docs(search_dir, idx, max_docs, max_chars):
    """Read the saved result files for one web_search call."""
    docs = []
    for f in sorted(glob.glob(os.path.join(search_dir, f"search_{idx}_result_*.md"))):
        try:
            txt = open(f, encoding="utf-8", errors="replace").read().strip()
        except OSError:
            continue
        if txt:
            docs.append(txt[:max_chars])
        if len(docs) >= max_docs:
            break
    return docs


def belief_summary(bdict):
    """Format a logged belief state as a prior-evidence summary for hcond.

    Deliberately EXCLUDES the probability `p` (and confidence): we want the
    sufficient statistic of the *evidence*, not the running posterior, so the
    incremental LLR isn't contaminated by reading off logit(p_t)-logit(p_{t-1}).
    """
    parts = []
    if bdict.get("evidence_for"):
        parts.append("Evidence FOR: " + "; ".join(bdict["evidence_for"]))
    if bdict.get("evidence_against"):
        parts.append("Evidence AGAINST: " + "; ".join(bdict["evidence_against"]))
    if bdict.get("key_uncertainties"):
        parts.append("Open questions: " + "; ".join(bdict["key_uncertainties"]))
    return "\n".join(parts)


def load_questions(config, exam_stems, max_docs, max_chars, trial=1):
    """Return records with replayable web-search evidence.

    Multi-trial runs store each trajectory's json + searches under
    {config}/trial_{t}/{source}/; single-run configs store them directly under
    {config}/{source}/. We read from trial `trial` when that layout exists,
    else fall back to the top level. Each record is one agent trajectory: its
    own queries, retrieved docs, and direct forecast."""
    config_dir = os.path.join(FORECASTS_DIR, config)
    if not os.path.isdir(config_dir):
        sys.exit(f"No such config dir: {config_dir}")
    trial_dir = os.path.join(config_dir, f"trial_{trial}")
    run_root = trial_dir if os.path.isdir(trial_dir) else config_dir
    records = []
    for source in sorted(os.listdir(run_root)):
        if source not in KNOWN_SOURCES:
            continue
        for f in sorted(glob.glob(os.path.join(run_root, source, "*.json"))):
            stem = os.path.splitext(os.path.basename(f))[0]
            if exam_stems is not None and stem not in exam_stems:
                continue
            try:
                d = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            ys = normalize_outcomes(d.get("resolved_to"))
            if ys is None:
                continue
            search_dir = os.path.join(run_root, source, "searches", stem)
            # belief_history[k] is the state AFTER step k (index 0 = prior), so
            # the evidence known BEFORE a web_search at step S is index S-1.
            bh = d.get("belief_history", [])
            steps = []
            for e in d.get("tool_log", []):
                if e.get("tool") != "web_search" or "search_index" not in e:
                    continue
                docs = read_search_docs(search_dir, e["search_index"],
                                        max_docs, max_chars)
                if docs:
                    sn = e.get("step")
                    prior = (belief_summary(bh[sn - 1])
                             if isinstance(sn, int) and 0 <= sn - 1 < len(bh)
                             else "")
                    steps.append({"query": e.get("query", ""), "docs": docs,
                                  "history": prior})
            if not steps:
                continue  # nothing to replay
            # market_value is logged as a string (e.g. "0.0114"); coerce it.
            # Only a valid probability in (0,1) is usable as a prior anchor —
            # dataset questions store a raw series level here, not a probability.
            try:
                mv = float(d.get("market_value"))
            except (TypeError, ValueError):
                mv = None
            records.append({
                "stem": stem, "source": source, "ys": ys,
                "question": d.get("question", ""),
                "direct_p": float(d.get("forecast", 0.5)),
                "market_value": mv if (mv is not None and 0.0 < mv < 1.0) else None,
                "steps": steps,
            })
    return records


# ---------------------------------------------------------------------------
# LLM elicitation (cached)
# ---------------------------------------------------------------------------

class LLRCache:
    """Thread-safe disk-backed cache. Writes go through put()/fail() under a
    lock so concurrent workers and incremental save() don't race."""
    def __init__(self, path):
        self.path = path
        self.d = {}
        if os.path.exists(path):
            try:
                self.d = json.load(open(path))
            except (json.JSONDecodeError, OSError):
                self.d = {}
        self.calls = 0     # successful live LLM calls this run
        self.errors = 0    # calls that failed every retry (NOT cached)
        self.last_error = ""
        self.lock = threading.Lock()

    @staticmethod
    def key(*parts):
        h = hashlib.md5("\x1f".join(parts).encode("utf-8", "replace")).hexdigest()
        return h

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.calls += 1

    def fail(self, msg):
        with self.lock:
            self.errors += 1
            self.last_error = msg

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self.lock:
            snapshot = dict(self.d)  # copy under lock so workers can't mutate mid-dump
        tmp = self.path + ".tmp"
        json.dump(snapshot, open(tmp, "w"))
        os.replace(tmp, self.path)  # atomic: a crash mid-save can't corrupt the cache


_LLR_SYS = (
    "You are a careful forecasting analyst estimating the weight of evidence "
    "for a binary question. You output a single number: the log-likelihood-ratio "
    "(in nats) of the evidence under the two hypotheses, log[P(evidence | the "
    "answer is YES) / P(evidence | the answer is NO)]. Positive favors YES, "
    "negative favors NO, 0 means the evidence is uninformative. Think "
    "generatively: how much more consistent is this specific evidence with a "
    "world where the answer is YES versus NO. Do NOT report a posterior "
    "probability. Output ONLY the number."
)


def _llm_text(prompt, system, model, retries=4):
    """Call the LLM with exponential backoff. Returns text, or None if every
    attempt raised (e.g. persistent 429s). Never swallows into a fake answer."""
    from agent.llm_client import chat
    err = None
    for attempt in range(retries):
        try:
            text, *_ = chat(prompt, model=model, system=system, max_tokens=2000)
            return text
        except Exception as e:  # rate limit / transient API error
            err = e
            time.sleep(min(30, 2 ** attempt) + random.random())
    raise RuntimeError(f"LLM call failed after {retries} retries: {err}")


_TYP_SYS = (
    "You estimate how typical an observation is under a hypothesized outcome of "
    "a binary forecasting question. Use forward simulation: assume the stated "
    "outcome is the truth, then judge how typical/representative the evidence "
    "would be of a world in which that outcome holds. Output a single integer "
    "0-10: 10 = highly typical/expected under this outcome, 5 = neutral, "
    "0 = completely atypical/inconsistent with it. Output ONLY the integer."
)

# Floor added to each typicality score before ratioing, so r=0 doesn't give an
# infinite LLR and the per-evidence magnitude stays bounded (Amin Eq. 61 uses
# r/10; the /10 cancels in the ratio, so we ratio the raw scores + eps).
_TYP_EPS = 0.5


def _elicit_typicality(question, evidence, query, context, positive,
                       model, cache, dry):
    """One typicality score r in [0,10] under outcome=YES (positive) or NO,
    elicited per-state in a separate call (Amin 2601.01522, Eq. 61). Cached.

    `context` is the prior-evidence conditioning block (belief summary, raw
    history, or both); its content distinguishes the cache key per method."""
    state = "YES" if positive else "NO"
    k = cache.key("typ", state, model, question, query or "", evidence,
                  context or "")
    if k in cache.d:
        return cache.d[k]
    if dry:
        cache.calls += 1
        return 5.0
    qline = (f"The agent issued this search query: {query!r}.\n"
             if query is not None else "")
    hblock = (f"Evidence already gathered before this search:\n{context}\n\n"
              if context else "")
    newlabel = "NEW evidence retrieved by this search" if context else "Evidence (retrieved text)"
    incr = (" Judge only what the NEW evidence adds beyond what is already known."
            if context else "")
    prompt = (
        f"Binary question: {question}\n\n{qline}{hblock}"
        f"{newlabel}:\n{evidence}\n\n"
        f"Assume the TRUE outcome is: {state}. How typical/representative is the "
        f"evidence above of a world in which the answer is {state}?{incr} Answer "
        f"with a single integer 0-10 (10 = highly typical, 5 = neutral, "
        f"0 = completely atypical). Output only the integer."
    )
    try:
        text = _llm_text(prompt, _TYP_SYS, model)
    except RuntimeError as e:
        cache.fail(str(e))
        return 5.0  # neutral, not cached
    m = re.search(r"\d+(?:\.\d+)?", text)
    if m is None:
        cache.fail(f"unparseable typicality: {text[:80]!r}")
        return 5.0
    r = max(0.0, min(10.0, float(m.group(0))))
    cache.put(k, r)
    return r


def elicit_llr(question, evidence, query, model, cache, llr_cap, dry, tag,
               context=None, llr_mode="ratio"):
    """Return an LLR in nats (clipped to +-llr_cap), cached on disk.

    llr_mode="ratio": one call eliciting log p(x|s=1)/p(x|s=0) directly.
    llr_mode="per-state": two calls eliciting a typicality score in [0,10] per
        state (Amin Eq. 61), then lambda = log((r1+eps)/(r0+eps)).

    `context` is the prior-evidence conditioning block — the belief summary
    (bcond), the raw history (hcond), or both (hbcond). When present we ask for
    the INCREMENTAL contribution of the new evidence, the chain-rule term
    log p(x_t|s,prior)/..., which discounts redundancy automatically.

    A failed call (all retries exhausted, or no number parseable) is counted in
    cache.errors and returns a neutral 0.0 that is NOT cached — so it shows up
    in the report and is retried on the next run, never masquerading as a real
    'uninformative' 0."""
    if llr_mode == "per-state":
        r1 = _elicit_typicality(question, evidence, query, context, True,
                                model, cache, dry)
        r0 = _elicit_typicality(question, evidence, query, context, False,
                                model, cache, dry)
        if dry:
            return 0.0
        val = math.log((r1 + _TYP_EPS) / (r0 + _TYP_EPS))
        return max(-llr_cap, min(llr_cap, val))

    k = cache.key(tag, model, str(llr_cap), question, query or "", evidence,
                  context or "")
    if k in cache.d:
        return cache.d[k]
    if dry:
        cache.calls += 1
        return 0.0  # placeholder; not stored
    qline = (f"The agent issued this search query: {query!r}.\n"
             if query is not None else "")
    if context:
        prompt = (
            f"Binary question: {question}\n\n"
            f"{qline}"
            f"Evidence ALREADY gathered before this search:\n"
            f"{context}\n\n"
            f"NEW evidence retrieved by this search:\n{evidence}\n\n"
            f"Report the INCREMENTAL log[P(new evidence | YES, given the prior "
            f"evidence) / P(new evidence | NO, given the prior evidence)] in "
            f"nats, in [{-llr_cap}, {llr_cap}]. If the new evidence merely "
            f"repeats what the prior evidence already establishes, report ~0. "
            f"Output only the number."
        )
    else:
        prompt = (
            f"Binary question: {question}\n\n"
            f"{qline}"
            f"Evidence (retrieved text):\n{evidence}\n\n"
            f"Report log[P(evidence|YES)/P(evidence|NO)] in nats, in "
            f"[{-llr_cap}, {llr_cap}]. Output only the number."
        )
    try:
        text = _llm_text(prompt, _LLR_SYS, model)
    except RuntimeError as e:
        cache.fail(str(e))
        return 0.0
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if m is None:  # model returned no number -> a failure, not a real 0
        cache.fail(f"unparseable LLR response: {text[:80]!r}")
        return 0.0
    val = max(-llr_cap, min(llr_cap, float(m.group(0))))
    cache.put(k, val)
    return val


_CLUSTER_SYS = (
    "You group pieces of evidence that report the SAME underlying fact, event, "
    "or claim, so that redundant copies are counted once. Two items are in the "
    "same group if a reasonable analyst would treat them as the same piece of "
    "news rather than independent confirmation."
)


def cluster_docs(question, items, model, cache, dry):
    """Cluster (step_query, doc) items by underlying claim. Returns list[int]
    cluster ids aligned with `items`. Cached. One LLM call per question."""
    gists = [f"[{i}] (q={it['query'][:60]!r}) {it['doc'][:300]}"
             for i, it in enumerate(items)]
    k = cache.key("CLUSTER", model, question, "\n".join(gists))
    if k in cache.d:
        return cache.d[k]
    if dry:
        cache.calls += 1
        return list(range(len(items)))  # placeholder: all distinct
    prompt = (
        f"Binary question: {question}\n\n"
        f"Evidence items:\n" + "\n".join(gists) + "\n\n"
        f"Partition the item indices 0..{len(items)-1} into groups that report "
        f"the same underlying fact/event/claim. Output ONLY a JSON list of "
        f"lists of integers, e.g. [[0,2],[1],[3,4]]. Every index appears once."
    )
    try:
        text = _llm_text(prompt, _CLUSTER_SYS, model)
        clean = re.sub(r"```(?:json)?", "", text).strip()
        groups = json.loads(clean[clean.index("["):clean.rindex("]") + 1])
        ids = [-1] * len(items)
        for cid, grp in enumerate(groups):
            for idx in grp:
                if 0 <= idx < len(items):
                    ids[idx] = cid
        nxt = (max(ids) + 1) if ids else 0  # any unassigned -> own cluster
        for i in range(len(ids)):
            if ids[i] < 0:
                ids[i] = nxt; nxt += 1
    except (RuntimeError, ValueError, json.JSONDecodeError) as e:
        cache.fail(f"cluster failed: {e}")
        ids = list(range(len(items)))  # fall back to no pooling (counted)
    cache.put(k, ids)
    return ids


# ---------------------------------------------------------------------------
# the four posteriors (returns sum-of-LLR, before tempering/prior)
# ---------------------------------------------------------------------------

_SUMM_SYS = (
    "You compress raw web-search results into a short, factual evidence brief for "
    "forecasting a binary question. Keep only the facts, numbers, dates, and "
    "claims that bear on the question; drop boilerplate, navigation text, ads, and "
    "repetition. Be neutral and do not state a probability."
)


def summarize_step(question, query, docs, model, cache, dry):
    """Compress one search step's concatenated results into a short evidence
    brief sigma(x_t), conditioned on the question (context c) and query q_t so the
    summary keeps what is relevant. Cached in its own 'summ' namespace."""
    raw = "\n\n".join(docs)
    k = cache.key("summ", model, question, query or "", raw)
    if k in cache.d:
        return cache.d[k]
    if dry:
        cache.calls += 1
        return ""
    prompt = (
        f"Binary question (the context c): {question}\n\n"
        f"Search query that produced these results: {query}\n\n"
        f"Raw search results:\n{raw}\n\n"
        f"Write a concise evidence brief (at most ~5 sentences) of what these "
        f"results say that bears on the question. Facts, numbers, and dates only; "
        f"no probability."
    )
    try:
        text = _llm_text(prompt, _SUMM_SYS, model)
    except RuntimeError as e:
        cache.fail(str(e))
        return raw[:800]  # fallback: truncated raw, not cached
    cache.put(k, text.strip())
    return text.strip()


def _raw_history(steps, i, cap=12000):
    """Raw prior evidence before step i: the queries+docs of steps 0..i-1,
    capped to the most recent `cap` chars (an offline proxy for q_{1:t},x_{1:t-1})."""
    blocks = [f"Search query: {s['query']}\nResults:\n" + "\n\n".join(s["docs"])
              for s in steps[:i]]
    raw = "\n\n---\n\n".join(blocks)
    return raw[-cap:] if len(raw) > cap else raw


def sum_llr(rec, method, model, cache, llr_cap, dry, llr_mode="ratio"):
    q = rec["question"]
    steps = rec["steps"]
    if method == "uncond":
        return sum(elicit_llr(q, "\n\n".join(s["docs"]), None, model, cache,
                              llr_cap, dry, "uncond", llr_mode=llr_mode)
                   for s in steps)
    if method == "qcond":
        return sum(elicit_llr(q, "\n\n".join(s["docs"]), s["query"], model,
                              cache, llr_cap, dry, "qcond", llr_mode=llr_mode)
                   for s in steps)
    if method == "bcond":  # condition on belief-state SUMMARY b_{t-1}.h
        return sum(elicit_llr(q, "\n\n".join(s["docs"]), s["query"], model,
                              cache, llr_cap, dry, "bcond",
                              context=s.get("history", ""), llr_mode=llr_mode)
                   for s in steps)
    if method == "hcond":  # condition on RAW history q_{1:t}, x_{1:t-1}
        return sum(elicit_llr(q, "\n\n".join(s["docs"]), s["query"], model,
                              cache, llr_cap, dry, "hcond",
                              context=_raw_history(steps, i), llr_mode=llr_mode)
                   for i, s in enumerate(steps))
    if method == "hbcond":  # condition on raw history AND belief summary (~BLF)
        return sum(elicit_llr(
                       q, "\n\n".join(s["docs"]), s["query"], model, cache,
                       llr_cap, dry, "hbcond",
                       context=(_raw_history(steps, i)
                                + (("\n\n--- Running summary ---\n" + s["history"])
                                   if s.get("history") else "")),
                       llr_mode=llr_mode)
                   for i, s in enumerate(steps))
    if method in ("sbcond", "sqbcond", "sqhbcond"):
        # summarized observation sigma(x_t); summary is shared across these three
        # methods (same per-step brief), only the conditioning differs.
        summaries = [summarize_step(q, s["query"], s["docs"], model, cache, dry)
                     for s in steps]
        total = 0.0
        for i, s in enumerate(steps):
            bh = s.get("history", "")
            if method == "sbcond":                       # p(σ(x_t) | s, b.h)
                query, context = None, bh
            elif method == "sqbcond":                    # p(σ(x_t) | s, q_t, b.h)
                query, context = s["query"], bh
            else:                                        # + σ(x_{1:t-1})
                query = s["query"]
                prior = "\n".join(f"- {summaries[j]}" for j in range(i)
                                  if summaries[j])
                context = bh + (("\n\n--- Earlier search summaries ---\n" + prior)
                                if prior else "")
            total += elicit_llr(q, summaries[i], query, model, cache, llr_cap,
                                dry, method, context=context, llr_mode=llr_mode)
        return total
    if method == "pooled":
        items = [{"query": s["query"], "doc": doc}
                 for s in rec["steps"] for doc in s["docs"]]
        ids = cluster_docs(q, items, model, cache, dry)
        total, seen = 0.0, {}
        for it, cid in zip(items, ids):
            seen.setdefault(cid, []).append(it)
        for cid, group in seen.items():
            ev = "\n\n".join(g["doc"] for g in group)
            qy = group[0]["query"]
            total += elicit_llr(q, ev, qy, model, cache, llr_cap, dry, "qcond",
                                llr_mode=llr_mode)
        return total
    raise ValueError(method)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def prior_logit(rec, prior_mode):
    if prior_mode == "market" and rec["market_value"] is not None:
        return logit(rec["market_value"])
    return logit(0.5)


def run_config(config, recs, methods, model, cache, alphas, prior_mode,
               llr_cap, lo, hi, dry, llr_mode="ratio", workers=1):
    # accumulate raw sum-LLR per method per question (indexed by question position)
    sums = {m: [None] * len(recs) for m in methods}

    def do_question(ri):
        return ri, {m: sum_llr(recs[ri], m, model, cache, llr_cap, dry, llr_mode)
                    for m in methods}

    def abort_if_broken(done):
        # fail fast: if the first calls produce only errors and no successes, the
        # API/keys/rate-limits are broken — bail rather than caching garbage.
        if cache.errors >= 15 and cache.calls == 0:
            sys.exit(f"\nAborting: {cache.errors} LLM calls failed, 0 succeeded "
                     f"after {done} questions.\nLast error: {cache.last_error}\n"
                     f"Check OPENROUTER_API_KEY / rate limits / model name "
                     f"({model}), then re-run (cache is intact).")

    if dry or workers <= 1:
        for ri in range(len(recs)):
            _, mvals = do_question(ri)
            for m in methods:
                sums[m][ri] = mvals[m]
            if not dry:
                abort_if_broken(ri + 1)
    else:
        done = 0
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(do_question, ri) for ri in range(len(recs))]
            for fut in cf.as_completed(futs):
                ri, mvals = fut.result()
                for m in methods:
                    sums[m][ri] = mvals[m]
                done += 1
                if done % 10 == 0 or done == len(recs):
                    cache.save()  # incremental, interrupt-safe (atomic replace)
                    print(f"  ... {done}/{len(recs)} questions "
                          f"({cache.calls} calls, {cache.errors} errors)", flush=True)
                abort_if_broken(done)
    if dry:
        return None

    # direct baseline
    direct_b = [brier(r["direct_p"], r["ys"], lo, hi) for r in recs]
    direct_pairs = [(clampp(r["direct_p"], lo, hi), y) for r in recs for y in r["ys"]]
    results = {"direct": {"brier@best": float(np.mean(direct_b)),
                          "alpha*": None,
                          "ece@best": ece([p for p, _ in direct_pairs],
                                          [y for _, y in direct_pairs])}}

    results["direct"]["pairs"] = direct_pairs
    for m in methods:
        prior_l = [prior_logit(r, prior_mode) for r in recs]
        best = None
        best_pairs = None
        curve = []
        for a in alphas:
            ps = [sigmoid(prior_l[i] + a * sums[m][i]) for i in range(len(recs))]
            b = float(np.mean([brier(ps[i], recs[i]["ys"], lo, hi)
                               for i in range(len(recs))]))
            pairs = [(clampp(ps[i], lo, hi), y)
                     for i in range(len(recs)) for y in recs[i]["ys"]]
            e = ece([p for p, _ in pairs], [y for _, y in pairs])
            curve.append((a, b, e))
            if best is None or b < best[1]:
                best = (a, b, e); best_pairs = pairs
        results[m] = {"alpha*": best[0], "brier@best": best[1],
                      "ece@best": best[2], "curve": curve, "pairs": best_pairs,
                      "brier@1.0": next(b for a, b, e in curve if abs(a - 1.0) < 1e-9)
                                    if any(abs(a - 1.0) < 1e-9 for a, _, _ in curve)
                                    else None}
    return results


def report(config, recs, results, methods, alphas, llr_mode="ratio"):
    print(f"\n{'=' * 72}\nCONFIG: {config}   (n={len(recs)} questions, "
          f"{sum(len(r['ys']) for r in recs)} resolution events)   "
          f"[llr={llr_mode}]")
    print(f"{'=' * 72}")
    n_steps = np.mean([len(r["steps"]) for r in recs])
    n_docs = np.mean([sum(len(s["docs"]) for s in r["steps"]) for r in recs])
    print(f"mean web-search steps/q: {n_steps:.1f}   mean docs/q: {n_docs:.1f}")
    print(f"\n{'method':<8} {'alpha*':>7} {'Brier@a*':>9} {'Brier@1.0':>9} "
          f"{'ECE@a*':>7}")
    print("-" * 46)
    d = results["direct"]
    print(f"{'direct':<8} {'-':>7} {d['brier@best']:>9.4f} {'-':>9} "
          f"{d['ece@best']:>7.4f}")
    for m in methods:
        r = results[m]
        b1 = f"{r['brier@1.0']:.4f}" if r["brier@1.0"] is not None else "-"
        print(f"{m:<8} {r['alpha*']:>7.2f} {r['brier@best']:>9.4f} {b1:>9} "
              f"{r['ece@best']:>7.4f}")
    print(f"\n(lower Brier/ECE better; direct = agent's logged forecast baseline. "
          f"alpha*<1 ⇒ accumulated evidence was overconfident / correlated.)")


def _reliability(pairs, n_bins=10):
    """Decile reliability points: (mean predicted, mean outcome, weight)."""
    if not pairs:
        return [], [], []
    ps = np.array([p for p, _ in pairs]); ys = np.array([y for _, y in pairs])
    order = np.argsort(ps)
    xs, os_, ws = [], [], []
    for chunk in np.array_split(order, min(n_bins, len(ps))):
        if len(chunk):
            xs.append(ps[chunk].mean()); os_.append(ys[chunk].mean())
            ws.append(len(chunk))
    return xs, os_, ws


def make_plots(config, results, methods, out_dir, llr_mode="ratio"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    d = results["direct"]
    colors = {"uncond": "tab:orange", "qcond": "tab:green", "bcond": "tab:red",
              "hcond": "tab:purple", "hbcond": "tab:brown", "pooled": "tab:gray",
              "sbcond": "tab:blue", "sqbcond": "tab:cyan", "sqhbcond": "tab:pink"}

    # Panel 1: Brier vs alpha
    for m in methods:
        c = results[m]["curve"]
        ax[0].plot([a for a, _, _ in c], [b for _, b, _ in c], "o-",
                   color=colors.get(m, "gray"), label=m)
        ax[0].scatter([results[m]["alpha*"]], [results[m]["brier@best"]],
                      color=colors.get(m, "gray"), s=90, zorder=5,
                      edgecolor="k", linewidth=0.6)
    ax[0].axhline(d["brier@best"], ls="--", color="k", label="direct (baseline)")
    ax[0].set_xlabel("tempering α"); ax[0].set_ylabel("mean Brier")
    ax[0].set_title("Brier vs α  (★ = best α)"); ax[0].legend(fontsize=8)

    # Panel 2: ECE vs alpha
    for m in methods:
        c = results[m]["curve"]
        ax[1].plot([a for a, _, _ in c], [e for _, _, e in c], "o-",
                   color=colors.get(m, "gray"), label=m)
    ax[1].axhline(d["ece@best"], ls="--", color="k", label="direct")
    ax[1].set_xlabel("tempering α"); ax[1].set_ylabel("ECE")
    ax[1].set_title("Calibration error vs α"); ax[1].legend(fontsize=8)

    # Panel 3: reliability at best α
    ax[2].plot([0, 1], [0, 1], ls=":", color="gray")
    for name, res, col in ([("direct", d, "k")] +
                           [(m, results[m], colors.get(m, "gray")) for m in methods]):
        xs, os_, ws = _reliability(res.get("pairs", []))
        if xs:
            ax[2].plot(xs, os_, "o-", color=col, markersize=4,
                       label=f"{name} (α*={res['alpha*']})" if res["alpha*"] else name)
    ax[2].set_xlabel("mean predicted p"); ax[2].set_ylabel("observed frequency")
    ax[2].set_title("Reliability @ best α"); ax[2].legend(fontsize=8)
    ax[2].set_xlim(0, 1); ax[2].set_ylim(0, 1)

    fig.suptitle(f"Explicit-Bayes replay — {config}  [llr={llr_mode}]", fontsize=12)
    fig.tight_layout()
    suffix = "" if llr_mode == "ratio" else f"_{llr_mode}"
    path = os.path.join(out_dir, f"likelihood_{config}{suffix}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    plot → {path}")


def summary_size_report(recs, model, cache):
    """Print mean raw-vs-summary size of x_t, for any steps whose summary is
    cached (i.e. a summarized method was run)."""
    raw_sizes, sum_sizes = [], []
    for r in recs:
        for s in r["steps"]:
            raw = "\n\n".join(s["docs"])
            k = cache.key("summ", model, r["question"], s["query"] or "", raw)
            if k in cache.d:
                raw_sizes.append(len(raw)); sum_sizes.append(len(cache.d[k]))
    if sum_sizes:
        rm, sm = np.mean(raw_sizes), np.mean(sum_sizes)
        print(f"\nx_t size (summarized methods): raw mean={int(rm)} chars  ->  "
              f"summary mean={int(sm)} chars ({100*sm/rm:.0f}% of raw, "
              f"{len(sum_sizes)} steps).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--exam", default=None)
    ap.add_argument("--n", type=int, default=0, help="subsample N questions (0=all)")
    ap.add_argument("--trial", type=int, default=1,
                    help="which trial's trajectory+searches to replay (multi-trial configs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="uncond,qcond,pooled")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--alphas", default="0.1,0.25,0.4,0.55,0.7,0.85,1.0")
    ap.add_argument("--prior", choices=["half", "market"], default="half",
                    help="starting point for accumulation (0.5 or the market value)")
    ap.add_argument("--clamp", type=float, default=0.05)
    ap.add_argument("--llr-cap", type=float, default=4.0,
                    help="clip each elicited LLR to +-cap nats")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent questions (LLM calls are I/O-bound; chat()'s "
                         "per-provider semaphore throttles to the rate limit)")
    ap.add_argument("--llr", choices=["ratio", "per-state"], default="ratio",
                    help="elicitation: 'ratio' = one call for log p(x|1)/p(x|0); "
                         "'per-state' = two typicality calls (Amin Eq. 61), "
                         "lambda = log((r1+eps)/(r0+eps))")
    ap.add_argument("--max-docs-per-search", type=int, default=6)
    ap.add_argument("--max-doc-chars", type=int, default=1500)
    ap.add_argument("--out", default=os.path.join("experiments", "analysis", "likelihood"))
    ap.add_argument("--dry-run", action="store_true",
                    help="count the paid LLM calls a real run would make, then exit")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    lo, hi = args.clamp, 1.0 - args.clamp
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    alphas = [float(a) for a in args.alphas.split(",")]
    exam_stems = load_exam_stems(args.exam) if args.exam else None
    cache = LLRCache(os.path.join(args.out, "llr_cache.json"))

    for config in args.configs.split(","):
        config = config.strip()
        recs = load_questions(config, exam_stems,
                              args.max_docs_per_search, args.max_doc_chars,
                              trial=args.trial)
        if args.n and len(recs) > args.n:
            random.Random(args.seed).shuffle(recs)
            recs = recs[:args.n]
        if not recs:
            print(f"{config}: no replayable questions (no saved searches?)")
            continue

        cache.calls = 0; cache.errors = 0
        results = run_config(config, recs, methods, args.model, cache, alphas,
                             args.prior, args.llr_cap, lo, hi, args.dry_run,
                             llr_mode=args.llr, workers=args.workers)
        if not args.dry_run and cache.errors:
            print(f"\n⚠ {cache.errors} LLM calls FAILED this run (returned a "
                  f"neutral 0, not cached). Results below are partial/unreliable; "
                  f"re-run to retry. Last error: {cache.last_error}")
        if args.dry_run:
            extra = "; per-state doubles per evidence" if args.llr == "per-state" else ""
            print(f"{config}: n={len(recs)} questions → "
                  f"~{cache.calls} uncached LLM calls "
                  f"({'pooled adds 1 cluster call/q' if 'pooled' in methods else 'no clustering'}{extra}). "
                  f"Re-run without --dry-run to execute (model={args.model}, llr={args.llr}).")
            continue
        cache.save()
        report(config, recs, results, methods, alphas, args.llr)
        summary_size_report(recs, args.model, cache)
        if not args.no_plots:
            make_plots(config, results, methods, args.out, args.llr)
    if not args.dry_run:
        cache.save()


if __name__ == "__main__":
    main()
