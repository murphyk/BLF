"""
knowledge_cutoffs.py — Training data cutoff dates for LLM models.

Used by make_fb_data.py to filter benchmark questions that fall within
a model's training window.
"""

KNOWLEDGE_CUTOFFS = {
    "anthropic/claude-sonnet-4-6":      "2025-08-31",
    "anthropic/claude-opus-4-6":        "2025-05-31",
    "google/gemini-3.1-pro-preview":    "2025-01-31",
    "google/gemini-3-flash-preview":    "2025-01-31",
    "openai/gpt-5.4":                   "2025-08-31",
    "x-ai/grok-4-fast":                 "2025-07-31",
    "x-ai/grok-4.1-fast":              "2025-07-31",
    "x-ai/grok-4":                      "2025-07-31",
    "moonshotai/kimi-k2.5":            "2025-06-30",
    "moonshotai/kimi-k2":              "2025-03-31",
    "openai/gpt-oss-120b":             "2024-06-30",
}
