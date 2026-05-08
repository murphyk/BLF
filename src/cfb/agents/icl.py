"""ICL wrapper: adds an in-context-learning memory to a base zero-shot LLM
agent. The agent stores past (question_text, outcome) pairs and prepends
the most recent ones to each new query as few-shot examples.

This is the *stateful* baseline the CFB leaderboard wants to beat: a frozen
LLM whose only learning channel is its own context window.

For now, memory is a simple FIFO of (forecast_date, question_text, o). When
prompt budget is tight, we keep the most recent K examples; smarter
compression schemes (clustering, retrieval, summarisation) are future work.
"""

from __future__ import annotations
import concurrent.futures as cf
from collections import deque
from datetime import date

from ..schema import Question, Resolution
from ._llm import chat, parse_probability
from .flash_zs import render_halawi


SYSTEM_ICL = (
    "You are a careful binary forecaster. You will be shown a list of past "
    "questions and their actual yes/no outcomes, then a NEW question. Use the "
    "past examples to calibrate your estimate. Reply with a single line of "
    "the form\n"
    "  probability=<float in [0,1]>\n"
    "with no additional text, no explanation, no caveats."
)


def _format_example(text: str, o: int) -> str:
    return f"  - Q: {text.strip()[:240]}\n    Outcome: {'YES' if o == 1 else 'NO'}"


class ICLAgent:
    """Wraps a frozen LLM. Maintains a FIFO of resolved (question, outcome)
    pairs and prepends the most recent K to the prompt for each new query."""

    DEFAULT_MODEL = "openrouter/google/gemini-3-flash-preview"

    def __init__(self, model: str = None, crowd: bool = False,
                 max_tokens: int = 256, max_workers: int = 8,
                 memory_k: int = 50, example_chars: int = 240):
        self.model = model or self.DEFAULT_MODEL
        self.crowd = crowd
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.memory_k = memory_k
        self.example_chars = example_chars
        # Store the question text indexed by u so observe() can look it up
        # when the resolution arrives.
        self._q_text: dict[str, str] = {}
        # Resolved memory: list of (resolution_date, text, o) — newest last.
        self._memory: deque[tuple[date, str, int]] = deque()

    # --- prompting ---

    def _examples_block(self) -> str:
        if not self._memory:
            return ""
        recent = list(self._memory)[-self.memory_k:]
        lines = [_format_example(t, o) for _, t, o in recent]
        return ("Past resolved questions (most-recent first):\n" +
                "\n".join(reversed(lines)) + "\n\n---\n\n")

    def _forecast_one(self, q: Question) -> tuple[str, float]:
        prompt = self._examples_block() + "NEW question:\n\n" + \
            render_halawi(q, crowd=self.crowd)
        try:
            text = chat(prompt, model=self.model, system=SYSTEM_ICL,
                        max_tokens=self.max_tokens, temperature=0.0)
        except Exception:
            return q.u, 0.5
        return q.u, parse_probability(text, default=0.5)

    # --- env-facing API ---

    def act(self, questions: list[Question]) -> dict[str, float]:
        if not questions:
            return {}
        for q in questions:
            self._q_text[q.u] = q.text
        out: dict[str, float] = {}
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for u, p in ex.map(self._forecast_one, questions):
                out[u] = p
        return out

    def observe(self,
                questions: list[Question],
                forecasts: dict[str, float],
                resolutions: list[Resolution]) -> None:
        for r in resolutions:
            text = self._q_text.pop(r.u, None)
            if text is None:
                continue
            self._memory.append((r.r, text, int(r.o)))
            # FIFO bound — keep ~3x memory_k so we always have memory_k to draw from
            while len(self._memory) > 3 * self.memory_k:
                self._memory.popleft()

    def state(self) -> dict:
        return {"memory_size": len(self._memory),
                "in_flight": len(self._q_text)}
