"""prompts.py — System and question prompts for the agentic forecaster."""

import re


# ---------------------------------------------------------------------------
# System prompts (one per mode)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_COMMON = """\
You are an expert superforecaster. Your task is to predict the probability that \
a binary question will resolve to YES given information up to a certain \
date. That is, you must estimate \
Prob(outcome(on resolution date)=True | info(<= forecast date)) \
for each resolution date listed in the question prompt. \
{mode_paragraph}\
The resolution date(s) will also be specified in the question prompt; \
the gap between resolution date and forecast date represents the \
forecast horizon.

You work in a tool-use loop:
1. Read the question, its resolution criteria, and other background information.
2. Form a base rate estimate (outside view / reference class reasoning) \
based on this initial information.
3. Now perform a loop:
  3a. At each step, choose ONE tool to call from the available tools listed below.
  3b. After each tool call, your belief state is updated and shown to you.
4. When you have gathered enough evidence, call the final \
submit(probability, reasoning) tool, which ends the loop.

{tools_section}

If the question and/or background are not in English, mentally translate \
them first and use English for all your search queries and reasoning.

Belief state rules:
- Evidence lists should ACCUMULATE across steps, not be rewritten from scratch. \
Add new items from each search; only remove items if they are directly contradicted.
- Each evidence item MUST cite its source using the format (search_X_result_Y), \
e.g. "Premier Smith said she will hold a referendum next year (search_3_result_0)". \
This helps trace claims back to specific search results.
- Include an update_reasoning field explaining WHY the new evidence changed your \
probability. E.g. "Search revealed the petition requires 177,000 signatures in \
120 days, making the timeline very tight — this pushes p down from 0.35 to 0.25."
- When weighing evidence, consider RECENCY and AUTHORITATIVENESS of sources. \
A recent news report from a major outlet should carry more weight than an older \
background article. Do NOT revert your probability based on older or less \
authoritative sources after updating on newer, more specific evidence. \
If new evidence strongly moved your probability, only move it back if you find \
equally strong counter-evidence from a similarly recent and authoritative source.

Suggested strategy:
- Use web_search to find specific evidence. Generate targeted queries based on \
what you still need to know (check your key_uncertainties in the belief state).
- Use summarize_results to read the full content of promising results. \
Select results that look most relevant based on the snippets.
- Update your probability after each piece of evidence using approximate Bayesian \
reasoning: ask "how much more/less likely is this evidence under YES vs NO?"
- Avoid anchoring too strongly on your initial estimate. Let evidence move you.
- When calling any tool, you MUST include an updated_belief object reflecting \
your current thinking (ie. summarizing what you know before the tool action \
has finished).

Rules:
- You MUST call submit before step {max_steps}. If you reach the last step \
without submitting, your current probability will be used automatically.
- Call submit once your probability has stabilized AND you have no major \
outstanding uncertainties that you believe further search could resolve. \
Do not keep searching just to use up all your steps — submitting early \
with a well-supported estimate is better than redundant searches. \
If your last 2-3 searches have not meaningfully changed your probability \
(less than 0.03 change), that is a strong signal to submit now.
{extra_rules}\
- Probabilities must be between 0.05 and 0.95 (never 0 or 1 — nothing is certain).
- Be well-calibrated: a 70% prediction should resolve YES about 70% of the time.
"""

_SYSTEM_PROMPT_NOSTATE = """\
You are an expert superforecaster. Your task is to predict the probability that \
a binary question will resolve to YES given information up to a certain \
date. That is, you must estimate \
Prob(outcome(on resolution date)=True | info(<= forecast date)) \
for each resolution date listed in the question prompt. \
{mode_paragraph}\
The resolution date(s) will also be specified in the question prompt; \
the gap between resolution date and forecast date represents the \
forecast horizon.

You work in a tool-use loop:
1. Read the question, its resolution criteria, and other background information.
2. Now perform a loop:
  2a. At each step, choose ONE tool to call from the available tools listed below.
  2b. The tool result will be added to the conversation.
3. When you have gathered enough evidence, call the final \
submit(probability, reasoning) tool, which ends the loop.

{tools_section}

If the question and/or background are not in English, mentally translate \
them first and use English for all your search queries and reasoning.

Suggested strategy:
- Use web_search to find specific evidence. Generate targeted queries \
that address different aspects of the question.
- Use summarize_results to read the full content of promising results. \
Select results that look most relevant based on the snippets.
- Consider all the evidence you have gathered when making your final \
probability estimate.

Rules:
- You MUST call submit before step {max_steps}. If you reach the last step \
without submitting, a default probability of 0.5 will be used.
- Call submit when you believe you have gathered sufficient evidence.
{extra_rules}\
- Probabilities must be between 0.05 and 0.95 (never 0 or 1 — nothing is certain).
- Be well-calibrated: a 70% prediction should resolve YES about 70% of the time.
"""

_MODE_LIVE = (
    "The forecast date (also called knowledge cutoff date) will be specified "
    "in the question prompt, and will correspond to today's date, since we are "
    "doing live testing. "
)

_MODE_BACKTEST = (
    "The forecast date (also called knowledge cutoff date) will be specified "
    "in the question prompt, and will be in the past, since we are doing "
    "backtesting. You must ignore any information from after the forecast date "
    "to prevent leakage. "
)

_EXTRA_RULES_BACKTEST = (
    "- Do NOT search for information published after the knowledge cutoff date.\n"
)

# Source-specific tools and blocked domains
_SOURCE_TOOLS = {
    "yfinance": "fetch_ts_yfinance",
    "fred": "fetch_ts_fred",
    "dbnomics": "fetch_ts_dbnomics",
    "wikipedia": "fetch_wikipedia_toc, fetch_wikipedia_section",
    "polymarket": "fetch_polymarket_info",
    "manifold": "fetch_manifold_info",
}

_SOURCE_BLOCKED = {
    "polymarket": "polymarket.com",
    "manifold": "manifold.markets",
    "metaculus": "metaculus.com",
    "infer": "randforecastinginitiative.org",
    "dbnomics": "db.nomics.world",
    "fred": "fred.stlouisfed.org",
    "yfinance": "finance.yahoo.com",
    "wikipedia": "wikipedia.org",
}


def _build_tools_section(source: str, live: bool,
                         use_tools: bool = True, use_search: bool = True) -> str:
    """Build the Available tools section, customized per source."""
    blocked = _SOURCE_BLOCKED.get(source, "")
    source_tool = _SOURCE_TOOLS.get(source, "") if use_tools else ""

    parts = ["Available tools:"]

    if use_search:
        # web_search
        if live or not blocked:
            parts.append("- web_search: search the web for evidence.")
        else:
            parts.append(
                f"- web_search: search the web for evidence.\n"
                f"  (Blocked URLs: {blocked}. Results are date-filtered to before cutoff.)")

        parts.append("- summarize_results: read full content of selected search results.")

        # lookup_url
        if live or not blocked:
            parts.append("- lookup_url: fetch and read a specific URL.")
        else:
            parts.append(
                f"- lookup_url: fetch and read a specific URL.\n"
                f"  (Same URLs blocked as for web_search.)")

    # Source-specific tool with per-source description
    _TOOL_DESCRIPTIONS = {
        "yfinance": (
            "- fetch_ts_yfinance: fetch historical stock prices (last 1 year).\n"
            "  Call this as your FIRST action with the ticker symbol.\n"
            "  Data is date-filtered to before cutoff (not URL-blocked)."),
        "fred": (
            "- fetch_ts_fred: fetch FRED economic data series (last 1 year).\n"
            "  Call this as your FIRST action with the series ID.\n"
            "  Data is date-filtered to before cutoff (not URL-blocked)."),
        "dbnomics": (
            "- fetch_ts_dbnomics: fetch DBnomics time series (last 2 years).\n"
            "  Call this as your FIRST action with the series URL from the background.\n"
            "  Data is date-filtered to before cutoff (not URL-blocked)."),
        "wikipedia": (
            "- fetch_wikipedia_toc: get the table of contents (section titles) of a Wikipedia page.\n"
            "  Call this FIRST with the Wikipedia URL to see available sections.\n"
            "- fetch_wikipedia_section: extract text from a specific section.\n"
            "  Call this after fetch_wikipedia_toc with the section title you need.\n"
            "  Both tools return the page as it existed at the cutoff date."),
        "polymarket": (
            "- fetch_polymarket_info: fetch market probability history (up to 90 days).\n"
            "  Call this as your FIRST action to see how the market price has evolved.\n"
            "  Data is date-filtered to before cutoff (not URL-blocked)."),
        "manifold": (
            "- fetch_manifold_info: fetch market probability history (up to 90 days).\n"
            "  Call this as your FIRST action to see how the market price has evolved.\n"
            "  Data is date-filtered to before cutoff (not URL-blocked)."),
    }
    if use_tools:
        tool_desc = _TOOL_DESCRIPTIONS.get(source)
        if tool_desc:
            parts.append(tool_desc)

    _DATASET_SOURCES = {"yfinance", "fred", "dbnomics", "wikipedia", "acled"}
    if source in _DATASET_SOURCES:
        parts.append(
            "- submit: submit your list of final probability estimates, "
            "one for each of the resolution dates.")
    else:
        parts.append("- submit: submit your final probability estimate.")

    return "\n".join(parts)


def get_system_prompt(max_steps: int, live: bool = False,
                      source: str = "", nobelief: bool = False,
                      use_tools: bool = True, use_search: bool = True) -> str:
    """Return the system prompt, customized for the question source."""
    tools_section = _build_tools_section(source, live,
                                          use_tools=use_tools,
                                          use_search=use_search)
    template = _SYSTEM_PROMPT_NOSTATE if nobelief else _SYSTEM_PROMPT_COMMON
    return template.format(
        mode_paragraph=_MODE_LIVE if live else _MODE_BACKTEST,
        tools_section=tools_section,
        max_steps=max_steps,
        extra_rules="" if live else _EXTRA_RULES_BACKTEST,
    )


# Keep SYSTEM_PROMPT as a backward-compatible alias (backtesting, generic tools).
SYSTEM_PROMPT = _SYSTEM_PROMPT_COMMON.format(
    mode_paragraph=_MODE_BACKTEST,
    tools_section=_build_tools_section("", False),
    max_steps="{max_steps}",
    extra_rules=_EXTRA_RULES_BACKTEST,
)


# ---------------------------------------------------------------------------
# Halawi et al. (2024) zero-shot prompt
# From: "Approaching Human-Level Forecasting with Language Models"
# ---------------------------------------------------------------------------

HALAWI_SYSTEM = """\
You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction of the probability that the question will be resolved as true. \
You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. \
If for some reason you can't answer, pick the base rate, but return a number between 0 and 1."""

HALAWI_QUESTION_TEMPLATE = """\
Question: {question}
Question Background: {background}
Resolution Criteria: {resolution_criteria}
Question close date: {close_date}
{freeze_section}\
Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. Do not output anything else.
Answer:
{{ Insert answer here }}"""


def format_halawi_prompt(question: dict, cutoff_date: str,
                         show_crowd: int = 0,
                         show_prior: int = 0) -> str:
    """Format a question using the Halawi et al. (2024) zero-shot prompt."""
    q = question
    bg = q.get("background", "")
    rc = q.get("resolution_criteria", "")
    rdates = q.get("resolution_dates", [])
    rdate = q.get("resolution_date", "")
    close_date = rdates[-1] if rdates else rdate

    # Freeze value / crowd / prior section
    freeze_section = ""
    market_val = q.get("market_value")
    has_market = market_val and str(market_val).strip() not in ("", "unknown", "None")
    if show_crowd and has_market:
        expl = q.get("market_value_explanation", "")
        freeze_section = (f"The freeze value is {market_val}. "
                        f"{expl}\n" if expl else
                        f"The freeze value is {market_val}.\n")
    elif show_prior and not has_market:
        from config.empirical_prior import get_empirical_prior, get_prior_explanation
        prior = get_empirical_prior(q)
        if prior is not None:
            expl = get_prior_explanation(q)
            freeze_section = f"The freeze value is {prior:.3f}. {expl}.\n"

    return HALAWI_QUESTION_TEMPLATE.format(
        question=q.get("question", ""),
        background=bg,
        resolution_criteria=rc,
        close_date=close_date,
        freeze_section=freeze_section,
    )


# ---------------------------------------------------------------------------
# Question prompt
# ---------------------------------------------------------------------------

# Domains whose question pages could leak resolution outcomes
_LEAKY_DOMAINS = re.compile(
    r'https?://(?:www\.)?(?:'
    r'metaculus\.com/questions/'
    r'|manifold\.markets/'
    r'|polymarket\.com/'
    r'|predictit\.org/'
    r'|kalshi\.com/'
    r')\S*',
    re.IGNORECASE,
)


def _strip_leaky_urls(text: str) -> str:
    """Remove URLs to prediction market question pages that could leak outcomes.

    Keeps other URLs (wikipedia, news, etc.) that provide useful context.
    """
    return _LEAKY_DOMAINS.sub('[URL removed to prevent outcome leakage]', text)


_SOURCE_TOOL_HINTS = {
    "yfinance": (
        "you MUST call `fetch_ts_yfinance` as your FIRST action to retrieve the "
        "stock's price history. The tool output includes a statistical trend "
        "analysis with P(increase). Use this as your PRIMARY probability anchor. "
        "You may do at most ONE web search for additional context, then submit. "
        "Weekly stock moves are largely unpredictable — stay close to 0.5 unless "
        "the data shows a very strong signal."
    ),
    "fred": (
        "you MUST call `fetch_ts_fred` as your FIRST action to retrieve the FRED "
        "series history. The tool output includes a statistical trend analysis "
        "with P(increase). Use this as your PRIMARY probability anchor. "
        "You may do at most ONE web search, then submit."
    ),
    "dbnomics": (
        "you MUST call `fetch_ts_dbnomics` as your FIRST action to retrieve the "
        "DBnomics series history. The tool output includes a statistical trend "
        "analysis with P(increase). Use this as your PRIMARY probability anchor. "
        "Many DBnomics series have strong seasonal patterns. "
        "You may do at most ONE web search, then submit."
    ),
    "wikipedia": (
        "you MUST call `fetch_wikipedia_toc` as your FIRST action, then "
        "`fetch_wikipedia_section` to read the relevant section. "
        "You may do at most ONE web search for additional context, then submit."
    ),
    "polymarket": (
        "you have an especially useful tool, `fetch_polymarket_info` — use it to "
        "look up the market's probability history, to estimate the reliability of "
        "the crowd estimate (if present). Do not search polymarket.com directly."
    ),
    "manifold": (
        "you have an especially useful tool, `fetch_manifold_info` — use it to "
        "look up the market's probability history, to estimate the reliability of "
        "the crowd estimate (if present). Do not search manifold.markets directly."
    ),
}

_SOURCE_DISPLAY_NAMES = {
    "polymarket": "Polymarket",
    "manifold": "Manifold Markets",
    "metaculus": "Metaculus",
    "infer": "INFER (RFI)",
    "yfinance": "Yahoo Finance",
    "fred": "FRED (Federal Reserve Economic Data)",
    "dbnomics": "DBnomics",
    "wikipedia": "Wikipedia",
    "acled": "ACLED (Armed Conflict Location & Event Data)",
}


def format_question_prompt(question: dict, cutoff_date: str,
                           show_crowd: int = 0,
                           show_prior: int = 0,
                           use_tools: bool = True,
                           backtesting: bool = True,
                           nobelief: bool = False) -> str:
    """Build the initial user prompt from a question dict.

    backtesting=False (live mode): include the question URL, skip URL redaction.
    backtesting=True (default): redact leaky URLs, hide question URL.
    """
    q = question
    live = not backtesting
    parts = [
        f"# Question\n{q['question']}",
    ]

    # Background and resolution criteria.
    bg_parts = []
    url = q.get("url", "")
    # Only strip the question's own URL if it's a prediction market page
    # (e.g. metaculus.com/questions/...). News article URLs (e.g. AIBQ2) are safe.
    strip_own_url = bool(url and _LEAKY_DOMAINS.search(url))
    bg = q.get("background", "")
    if bg:
        if live:
            bg_parts.append(bg)
        else:
            clean = _strip_leaky_urls(bg)
            if strip_own_url:
                clean = clean.replace(url, "[question URL removed]")
            bg_parts.append(clean)
    rc = q.get("resolution_criteria", "")
    if rc:
        if live:
            bg_parts.append(rc)
        else:
            clean = _strip_leaky_urls(rc)
            if strip_own_url:
                clean = clean.replace(url, "[question URL removed]")
            bg_parts.append(clean)
    if bg_parts:
        parts.append(f"## Background and resolution criteria\n" + "\n\n".join(bg_parts))

    rdates = q.get("resolution_dates", [])
    rdate = q.get("resolution_date", "")
    if rdates and len(rdates) > 1:
        dates_str = ", ".join(str(d) for d in rdates)
        parts.append(
            f"## Resolution dates\n{dates_str}\n\n"
            f"You must submit **{len(rdates)} probabilities** (one per resolution date) "
            f"when you call submit. Your uncertainty should INCREASE with forecast "
            f"horizon — probabilities for distant dates should be closer to 0.5."
        )
    elif rdates:
        parts.append(f"## Resolution dates\n{rdates[0]}")
    elif rdate:
        parts.append(f"## Resolution date\n{rdate}")

    if backtesting:
        parts.append(
            f"## Knowledge cutoff\n{cutoff_date}\n"
            "You must not use any information from after this date."
        )
    else:
        parts.append(f"## Forecast date\n{cutoff_date} (today)")

    # Crowd signal: market price for market questions (controlled by show_crowd)
    market_val = q.get("market_value")
    if show_crowd and market_val and str(market_val).strip() not in ("", "unknown", "None"):
        market_date = q.get("market_date", "")
        explanation = q.get("market_value_explanation", "")
        try:
            crowd_str = f"{float(market_val):.2f}"
        except (ValueError, TypeError):
            crowd_str = str(market_val)
        if explanation:
            expl = explanation.rstrip(".")
            parts.append(f"## Market estimate\n{expl} on {market_date} was {crowd_str}.")
        else:
            parts.append(f"## Market estimate\nThe market estimate on {market_date} was {crowd_str}.")

    # Empirical prior: base rate for dataset questions (controlled by show_prior)
    if show_prior and not (market_val and str(market_val).strip() not in ("", "unknown", "None")):
        from config.empirical_prior import get_empirical_prior, get_prior_explanation
        prior = get_empirical_prior(q)
        if prior is not None:
            expl = get_prior_explanation(q)
            parts.append(
                f"## Prior estimate\n{expl}: {prior:.3f}.\n"
                f"Use this as your starting point, but adjust based on "
                f"question-specific evidence from search and tools."
            )

    # Source info — only show platform warning for prediction market sources
    _PLATFORM_SOURCES = {"polymarket", "manifold", "metaculus", "infer"}
    source = q.get("source", "")
    source_name = _SOURCE_DISPLAY_NAMES.get(source, "")
    if source_name:
        if live and url:
            parts.append(f"## Source\nThis question comes from **{source_name}**: {url}")
        elif live:
            parts.append(f"## Source\nThis question comes from **{source_name}**.")
        elif source in _PLATFORM_SOURCES:
            parts.append(f"## Source\nThis question comes from **{source_name}**. "
                         "Do not search for or visit the original question page on this "
                         "platform — use independent news and data sources instead.")

    # Tool hint
    tool_hint = ""
    if use_tools and source in _SOURCE_TOOL_HINTS:
        tool_hint = f"\nFor this question, {_SOURCE_TOOL_HINTS[source]}"

    if nobelief:
        parts.append(
            "## Instructions\n"
            "Search for relevant evidence using the available tools, then "
            "call submit with your final probability estimate and reasoning."
            + tool_hint
        )
    else:
        parts.append(
            "## Instructions\n"
            "First, form your initial belief state:\n"
            "1. Identify the reference class for this question (what category of events is this?)\n"
            "2. Estimate the base rate for this reference class\n"
            "3. List your key uncertainties — what information would most change your estimate?\n"
            "4. Set your initial probability\n\n"
            "Then call your first tool. You must include an `updated_belief` object "
            "with every tool call."
            + tool_hint
        )

    return "\n\n".join(parts)
