"""
Judge LLM backend.

LLM-as-judge metrics need a model to score answers. Production uses Anthropic's
Claude. A deterministic stub judge is the default when no API key is present,
so the framework is fully testable offline and in CI.

The judge always returns a float score in [0, 1] plus a short rationale. The
prompt asks the model to return strict JSON, and the parser is defensive about
models that wrap JSON in prose or code fences.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class JudgeVerdict:
    score: float       # 0.0 to 1.0
    rationale: str


class JudgeLLM(ABC):
    @abstractmethod
    def judge(self, prompt: str) -> JudgeVerdict:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


def _parse_json_verdict(text: str) -> JudgeVerdict:
    """Pull {"score": ..., "rationale": ...} out of a possibly messy response."""
    # Strip code fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    # Find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            score = float(obj.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            return JudgeVerdict(score, str(obj.get("rationale", "")))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    # Fallback, look for a bare number
    num = re.search(r"(\d*\.?\d+)", cleaned)
    if num:
        try:
            score = max(0.0, min(1.0, float(num.group(1))))
            return JudgeVerdict(score, cleaned[:200])
        except ValueError:
            pass
    return JudgeVerdict(0.0, f"could not parse: {cleaned[:120]}")


class AnthropicJudge(JudgeLLM):
    def __init__(self, model: str = "claude-sonnet-4-6",
                 api_key: Optional[str] = None):
        import anthropic
        self._model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def judge(self, prompt: str) -> JudgeVerdict:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=("You are a strict evaluator. Respond only with JSON of the "
                    "form {\"score\": <float 0..1>, \"rationale\": <short string>}. "
                    "No prose outside the JSON."),
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return _parse_json_verdict(text)

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"


class StubJudge(JudgeLLM):
    """
    Deterministic offline judge.

    Heuristics, not intelligence. It computes lexical overlap between the parts
    of the prompt it can identify (answer, context, question) and maps that to a
    score. Good enough to exercise the pipeline and produce non-degenerate
    numbers in tests and offline demos.
    """

    def judge(self, prompt: str) -> JudgeVerdict:
        # Pull labeled sections out of the judge prompt
        sections = {}
        for label in ("QUESTION", "ANSWER", "CONTEXT", "REFERENCE"):
            m = re.search(rf"{label}:\s*(.*?)(?=\n[A-Z]+:|\Z)", prompt, re.S)
            if m:
                sections[label] = m.group(1).strip()

        def words(s):
            return set(re.findall(r"\w{4,}", s.lower()))

        # context_precision style: QUESTION + CONTEXT, no ANSWER. Score the
        # overlap between the question and the retrieved context.
        if "ANSWER" not in sections and "CONTEXT" in sections \
                and "QUESTION" in sections:
            q_w = words(sections["QUESTION"])
            ctx_w = words(sections["CONTEXT"])
            if q_w and ctx_w:
                overlap = len(q_w & ctx_w) / len(q_w)
                score = min(1.0, 0.3 + 0.7 * overlap)
                return JudgeVerdict(round(score, 3),
                                    f"stub q-context overlap {overlap:.2f}")
            return JudgeVerdict(0.3, "stub no overlap")

        answer_w = words(sections.get("ANSWER", ""))
        if not answer_w:
            return JudgeVerdict(0.0, "empty answer")

        # Faithfulness style: overlap of answer with context
        if "CONTEXT" in sections:
            ctx_w = words(sections["CONTEXT"])
            if ctx_w:
                overlap = len(answer_w & ctx_w) / len(answer_w)
                score = min(1.0, 0.4 + 0.6 * overlap)
                return JudgeVerdict(round(score, 3),
                                    f"stub answer-context overlap {overlap:.2f}")

        # Relevance/correctness style: overlap of answer with question/reference
        ref = sections.get("REFERENCE") or sections.get("QUESTION", "")
        ref_w = words(ref)
        if ref_w:
            overlap = len(answer_w & ref_w) / max(len(ref_w), 1)
            score = min(1.0, 0.3 + 0.7 * overlap)
            return JudgeVerdict(round(score, 3),
                                f"stub answer-ref overlap {overlap:.2f}")

        return JudgeVerdict(0.5, "stub default")

    @property
    def name(self) -> str:
        return "stub"


def build_judge(prefer: str = "anthropic",
                api_key: Optional[str] = None) -> JudgeLLM:
    if prefer == "stub":
        return StubJudge()
    if prefer == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("warning, ANTHROPIC_API_KEY not set, using stub judge")
            return StubJudge()
        try:
            return AnthropicJudge(api_key=key)
        except Exception as exc:
            print(f"warning, anthropic judge failed ({exc}), using stub")
            return StubJudge()
    raise ValueError(f"unknown judge preference: {prefer}")
