"""
Core data types shared across the eval framework.

A TestCase is one question, optionally with ground-truth relevant document ids
and a reference answer. A RAGOutput is what a RAG system returns for a question,
the answer text plus the retrieved chunk ids and the latency. An EvalResult
pairs a TestCase with its RAGOutput and the metric scores computed for it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TestCase:
    __test__ = False  # tell pytest this dataclass is not a test class
    question: str
    # ids of the documents/chunks that should be retrieved for this question
    relevant_ids: List[str] = field(default_factory=list)
    # an optional gold answer for correctness scoring
    reference_answer: Optional[str] = None
    # free-form metadata (category, difficulty, source doc, etc.)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str = ""


@dataclass
class RAGOutput:
    answer: str
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    @property
    def retrieved_ids(self) -> List[str]:
        return [c.chunk_id for c in self.retrieved]

    @property
    def context_texts(self) -> List[str]:
        return [c.text for c in self.retrieved]


@dataclass
class EvalResult:
    test_case: TestCase
    output: RAGOutput
    scores: Dict[str, float] = field(default_factory=dict)
    # judge rationales keyed by metric name, useful for debugging
    rationales: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Aggregate of many EvalResults."""
    n_cases: int
    metric_means: Dict[str, float]
    metric_stds: Dict[str, float]
    latency_p50_ms: float
    latency_p95_ms: float
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    results: List[EvalResult] = field(default_factory=list)
    config: Dict[str, str] = field(default_factory=dict)
