"""bayes_update.py — Online explicit-Bayes belief update for the agent loop.

Instead of trusting the probability the LLM emits in `updated_belief`, this
recomputes the belief as a sequential Bayesian update on each new observation:

  1. summarize the observation, task-aware:  sigma_t = sigma(x_t | q_t, c)
  2. elicit a per-state typicality score r^s in [0,10] (Amin 2601.01522, Eq. 61),
     conditioned on the query, the belief summary b_{t-1}.h, and earlier
     summaries sigma_{1:t-1};
  3. lambda_t = log((r^1 + eps)/(r^0 + eps));
  4. logit p_t = logit p_{t-1} + alpha * lambda_t.

The LLM keeps maintaining the linguistic evidence (evidence_for/against, used as
b_{t-1}.h) and choosing actions; only the probability is taken over by the
explicit update. This is the online analogue of the offline `sqhbcond` variant
analysed in the paper appendix on explicit-Bayes updates.
"""

import concurrent.futures as cf
import math
import re

from agent.llm_client import chat

_TYP_EPS = 0.5  # floor on typicality scores so r=0 doesn't give an infinite LLR

_SUMM_SYS = (
    "You compress raw web-search results into a short, factual evidence brief for "
    "forecasting a binary question. Keep only the facts, numbers, dates, and "
    "claims that bear on the question; drop boilerplate, navigation text, ads, and "
    "repetition. Be neutral and do not state a probability."
)

_TYP_SYS = (
    "You estimate how typical an observation is under a hypothesized outcome of a "
    "binary forecasting question. Use forward simulation: assume the stated "
    "outcome is the truth, then judge how typical/representative the evidence "
    "would be of a world in which that outcome holds. Output a single integer "
    "0-10: 10 = highly typical/expected under this outcome, 5 = neutral, "
    "0 = completely atypical/inconsistent with it. Output ONLY the integer."
)


def _logit(p):
    p = min(0.98, max(0.02, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def belief_summary(state):
    """Format a belief state's evidence as b.h (probability excluded)."""
    parts = []
    if state.evidence_for:
        parts.append("Evidence FOR: " + "; ".join(state.evidence_for))
    if state.evidence_against:
        parts.append("Evidence AGAINST: " + "; ".join(state.evidence_against))
    if state.key_uncertainties:
        parts.append("Open questions: " + "; ".join(state.key_uncertainties))
    return "\n".join(parts)


def summarize_observation(question, query, text, model, max_chars=9000):
    """sigma(x_t | q_t, c): a task-aware compression of the observation."""
    raw = (text or "")[:max_chars]
    prompt = (
        f"Binary question (the context c): {question}\n\n"
        f"Search query that produced these results: {query}\n\n"
        f"Raw search results:\n{raw}\n\n"
        f"Write a concise evidence brief (at most ~5 sentences) of what these "
        f"results say that bears on the question. Facts, numbers, and dates only; "
        f"no probability."
    )
    try:
        out, *_ = chat(prompt, model=model, system=_SUMM_SYS, max_tokens=2000)
        return (out or "").strip() or raw[:800]
    except Exception:
        return raw[:800]


def _typicality(question, query, summary, context, positive, model):
    state = "YES" if positive else "NO"
    qline = f"The agent issued this search query: {query!r}.\n" if query else ""
    hblock = (f"Evidence already gathered before this search:\n{context}\n\n"
              if context else "")
    prompt = (
        f"Binary question: {question}\n\n{qline}{hblock}"
        f"NEW evidence retrieved by this search:\n{summary}\n\n"
        f"Assume the TRUE outcome is: {state}. How typical/representative is the "
        f"evidence above of a world in which the answer is {state}? Judge only "
        f"what the NEW evidence adds beyond what is already known. Answer with a "
        f"single integer 0-10. Output only the integer."
    )
    try:
        out, *_ = chat(prompt, model=model, system=_TYP_SYS, max_tokens=500)
        m = re.search(r"\d+(?:\.\d+)?", out or "")
        if m:
            return max(0.0, min(10.0, float(m.group(0))))
    except Exception:
        pass
    return 5.0


def bayes_step(prior_p, question, query, observation, prior_h,
               prior_summaries, model, alpha):
    """One online update. Returns (new_p, sigma_t, lambda_t, r1, r0)."""
    sigma = summarize_observation(question, query, observation, model)
    context = prior_h or ""
    if prior_summaries:
        context += "\n\n--- Earlier search summaries ---\n" + "\n".join(
            f"- {s}" for s in prior_summaries)
    # the two per-state typicality calls are independent -> run them concurrently
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(_typicality, question, query, sigma, context, True, model)
        f0 = ex.submit(_typicality, question, query, sigma, context, False, model)
        r1, r0 = f1.result(), f0.result()
    lam = math.log((r1 + _TYP_EPS) / (r0 + _TYP_EPS))
    new_p = _sigmoid(_logit(prior_p) + alpha * lam)
    return max(0.02, min(0.98, new_p)), sigma, lam, r1, r0
