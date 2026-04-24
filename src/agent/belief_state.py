"""belief_state.py — Explicit belief state maintained across agent steps."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BeliefState:
    p: float = 0.5
    base_rate_anchor: str = ""
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    key_uncertainties: list[str] = field(default_factory=list)
    confidence: str = "low"  # low / medium / high
    update_reasoning: str = ""  # why p changed at this step
    searches_tried: list[str] = field(default_factory=list)
    step: int = 0

    def to_prompt_str(self, max_steps: int = 10) -> str:
        """Format for injection into the LLM prompt."""
        lines = [
            f"Current belief state (step {self.step}/{max_steps}):",
            f"  Probability: {self.p:.3f}",
        ]
        if self.base_rate_anchor:
            lines.append(f"  Base rate anchor: {self.base_rate_anchor}")
        if self.evidence_for:
            lines.append("  Evidence FOR resolution (pushes p up):")
            for e in self.evidence_for:
                lines.append(f"    - {e}")
        if self.evidence_against:
            lines.append("  Evidence AGAINST resolution (pushes p down):")
            for e in self.evidence_against:
                lines.append(f"    - {e}")
        if self.key_uncertainties:
            lines.append("  Key uncertainties:")
            for u in self.key_uncertainties:
                lines.append(f"    - {u}")
        lines.append(f"  Confidence: {self.confidence}")
        if self.searches_tried:
            lines.append(f"  Searches tried: {self.searches_tried}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "p": self.p,
            "base_rate_anchor": self.base_rate_anchor,
            "evidence_for": list(self.evidence_for),
            "evidence_against": list(self.evidence_against),
            "key_uncertainties": list(self.key_uncertainties),
            "confidence": self.confidence,
            "searches_tried": list(self.searches_tried),
            "step": self.step,
        }
        if self.update_reasoning:
            d["update_reasoning"] = self.update_reasoning
        return d

    def evidence_char_count(self) -> int:
        """Total characters across all evidence and uncertainty lists."""
        return (
            sum(len(e) for e in self.evidence_for)
            + sum(len(e) for e in self.evidence_against)
            + sum(len(u) for u in self.key_uncertainties)
        )

    @classmethod
    def from_dict(cls, d: dict) -> BeliefState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Automatic compaction
# ---------------------------------------------------------------------------

_COMPACT_LLM = "openrouter/google/gemini-3-flash-preview"

# Compaction threshold: total chars across evidence_for + evidence_against + key_uncertainties
_COMPACT_THRESHOLD = 2000


def compact_belief(state: BeliefState, config=None) -> BeliefState:
    """Compact the belief state if evidence lists are getting too long.

    Uses an LLM to merge/deduplicate evidence items into a shorter list,
    preserving the most important points. Only triggers when total evidence
    chars exceed the threshold.
    """
    threshold = _COMPACT_THRESHOLD
    if config is not None:
        threshold = getattr(config, "compact_threshold", _COMPACT_THRESHOLD)
    if state.evidence_char_count() < threshold:
        return state

    from agent.llm_client import chat  # lazy import to avoid circular dependency

    prompt = (
        "You are compacting a forecaster's belief state to save context space.\n"
        "Merge, deduplicate, and condense the following evidence lists while "
        "preserving all important information. Each item should be one concise sentence.\n"
        "Aim for at most 3-5 items per list.\n\n"
        f"Evidence FOR (pushes probability up):\n"
        + "\n".join(f"- {e}" for e in state.evidence_for) + "\n\n"
        f"Evidence AGAINST (pushes probability down):\n"
        + "\n".join(f"- {e}" for e in state.evidence_against) + "\n\n"
        f"Key uncertainties:\n"
        + "\n".join(f"- {u}" for u in state.key_uncertainties) + "\n\n"
        "Output format (strict):\n"
        "FOR:\n- item\n- item\n"
        "AGAINST:\n- item\n- item\n"
        "UNCERTAINTIES:\n- item\n- item"
    )

    try:
        text, _, _, _ = chat(prompt, model=_COMPACT_LLM, max_tokens=1000)
        evidence_for, evidence_against, uncertainties = _parse_compact_response(text)
        return BeliefState(
            p=state.p,
            base_rate_anchor=state.base_rate_anchor,
            evidence_for=evidence_for or state.evidence_for,
            evidence_against=evidence_against or state.evidence_against,
            key_uncertainties=uncertainties or state.key_uncertainties,
            confidence=state.confidence,
            searches_tried=state.searches_tried,
            step=state.step,
        )
    except Exception:
        return state  # compaction failed, keep original


def _parse_compact_response(text: str) -> tuple[list[str], list[str], list[str]]:
    """Parse the compacted evidence from LLM response."""
    evidence_for = []
    evidence_against = []
    uncertainties = []
    current = None

    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("FOR"):
            current = evidence_for
        elif upper.startswith("AGAINST"):
            current = evidence_against
        elif upper.startswith("UNCERTAINT"):
            current = uncertainties
        elif line.startswith("- ") and current is not None:
            current.append(line[2:].strip())

    return evidence_for, evidence_against, uncertainties
