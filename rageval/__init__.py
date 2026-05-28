"""rageval, a lightweight framework for evaluating RAG systems."""

from .types import (EvalResult, EvalSummary, RAGOutput, RetrievedChunk,
                    TestCase)
from .judge_llm import (AnthropicJudge, JudgeLLM, JudgeVerdict, StubJudge,
                        build_judge)
from .evaluator import DEFAULT_METRICS, Evaluator, compare
from .adapters.base import CallableAdapter, RAGAdapter
from .report import save_report, to_html, to_markdown

__all__ = [
    "TestCase", "RAGOutput", "RetrievedChunk", "EvalResult", "EvalSummary",
    "JudgeLLM", "AnthropicJudge", "StubJudge", "JudgeVerdict", "build_judge",
    "Evaluator", "compare", "DEFAULT_METRICS",
    "RAGAdapter", "CallableAdapter",
    "save_report", "to_html", "to_markdown",
]

__version__ = "0.1.0"
