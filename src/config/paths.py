"""paths.py — Centralized directory paths for the project.

All path constants in one place. Import from here instead of hardcoding
"experiments/forecasts" or "experiments/exams" in individual modules.

Directory layout (v6):
    data/
        questions/{source}/{id}.json        — downloaded question files
        exams/{name}/mixture.json           — exam definitions
        exams/{name}/indices.json           — materialized exam indices
        exams/{name}/meta.json              — exam metadata
        tags_{version}/{source}/{id}.json   — LLM-classified tags
        fb_cache/                           — cached ForecastBench downloads
    experiments/
        xids/{xid}.json                     — experiment definitions
        forecasts/{config}/{source}/{id}.json — prediction outputs
        eval/{xid}/                         — evaluation outputs
        calibration_models/{name}/          — saved calibration models
        generated_prompts/                  — dumped prompts for inspection
        progress/                           — live progress monitoring
"""

import os

# ---------------------------------------------------------------------------
# Data (inputs)
# ---------------------------------------------------------------------------

QUESTIONS_DIR = os.path.join("data", "questions")
EXAMS_DIR = os.path.join("data", "exams")
TAGS_DIR_PREFIX = os.path.join("data", "tags_")  # + version name
FB_CACHE_DIR = os.path.join("data", "fb_cache")

# ---------------------------------------------------------------------------
# Experiments (outputs + definitions)
# ---------------------------------------------------------------------------

XIDS_DIR = os.path.join("experiments", "xids")
FORECASTS_DIR = os.path.join("experiments", "forecasts")
EVAL_DIR = os.path.join("experiments", "eval")
CALIBRATION_DIR = os.path.join("experiments", "calibration_models")
PROMPTS_DIR = os.path.join("experiments", "generated_prompts")
PROGRESS_DIR = os.path.join("experiments", "progress")

# Legacy (v5 compat — used by resolve_config for old results)
LEGACY_CONFIGS_DIR = os.path.join("experiments", "configs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def forecast_dir(config_name: str) -> str:
    """Return the forecast directory for a config: experiments/forecasts/{config}/"""
    return os.path.join(FORECASTS_DIR, config_name)


def forecast_path(config_name: str, source: str, qid: str) -> str:
    """Return path to a specific forecast file."""
    import re
    safe_id = re.sub(r'[/\\:]', '_', str(qid))
    return os.path.join(FORECASTS_DIR, config_name, source, f"{safe_id}.json")


def eval_dir(xid: str) -> str:
    """Return the eval output directory for an xid."""
    return os.path.join(EVAL_DIR, xid)


def exam_dir(exam_name: str) -> str:
    """Return the exam directory: data/exams/{name}/"""
    return os.path.join(EXAMS_DIR, exam_name)


def xid_path(xid: str) -> str:
    """Return path to an xid JSON file."""
    return os.path.join(XIDS_DIR, f"{xid}.json")


def tags_dir(version: str) -> str:
    """Return the tags directory for a version: data/tags_{version}/"""
    return f"{TAGS_DIR_PREFIX}{version}"
