"""
LLM-as-judge quality metrics.

These do not need ground-truth relevant ids. They ask a judge model to score
some aspect of the RAG output. Three are implemented.

  - faithfulness        is every claim in the answer supported by the retrieved
                        context? penalizes hallucination.
  - answer_relevance    does the answer actually address the question, or is it
                        evasive / off-topic?
  - context_precision   how much of the retrieved context was actually relevant
                        to answering the question? penalizes noisy retrieval.

Two more need a reference answer.

  - answer_correctness  does the answer match the gold reference answer?

Each builds a focused judge prompt and returns the verdict from the judge LLM.
"""

from typing import List, Optional

from ..judge_llm import JudgeLLM, JudgeVerdict
from ..types import RAGOutput, TestCase


FAITHFULNESS_PROMPT = """\
Evaluate whether the ANSWER is fully supported by the CONTEXT. An answer is
faithful if every factual claim it makes can be traced to the context. If the
answer introduces facts not present in the context, it is unfaithful. If the
answer correctly says it does not know because the context lacks the info, that
is faithful.

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}

Score 1.0 if fully faithful, 0.0 if it hallucinates, partial otherwise.
"""

RELEVANCE_PROMPT = """\
Evaluate whether the ANSWER directly addresses the QUESTION. Ignore whether the
answer is factually correct, only judge relevance. An evasive, off-topic, or
overly generic answer scores low. A focused answer that engages the question
scores high.

QUESTION: {question}

ANSWER: {answer}

Score 1.0 if the answer is fully on point, 0.0 if it ignores the question.
"""

CONTEXT_PRECISION_PROMPT = """\
Evaluate how relevant the retrieved CONTEXT is to answering the QUESTION.
High precision means most of the context is useful for answering. Low precision
means the context is full of unrelated material.

QUESTION: {question}

CONTEXT:
{context}

Score 1.0 if all context is relevant, 0.0 if none is, partial otherwise.
"""

CORRECTNESS_PROMPT = """\
Evaluate whether the ANSWER is correct relative to the REFERENCE answer. They
do not need identical wording, only the same factual content. Missing or
contradicting facts lower the score.

QUESTION: {question}

REFERENCE: {reference}

ANSWER: {answer}

Score 1.0 if factually equivalent to the reference, 0.0 if wrong.
"""


def _format_context(output: RAGOutput, max_chunks: int = 8) -> str:
    parts = []
    for i, chunk in enumerate(output.retrieved[:max_chunks], 1):
        parts.append(f"[{i}] {chunk.text.strip()}")
    return "\n\n".join(parts) if parts else "(no context retrieved)"


def faithfulness(judge: JudgeLLM, case: TestCase,
                 output: RAGOutput) -> JudgeVerdict:
    prompt = FAITHFULNESS_PROMPT.format(
        question=case.question,
        context=_format_context(output),
        answer=output.answer)
    return judge.judge(prompt)


def answer_relevance(judge: JudgeLLM, case: TestCase,
                     output: RAGOutput) -> JudgeVerdict:
    prompt = RELEVANCE_PROMPT.format(
        question=case.question, answer=output.answer)
    return judge.judge(prompt)


def context_precision(judge: JudgeLLM, case: TestCase,
                      output: RAGOutput) -> JudgeVerdict:
    prompt = CONTEXT_PRECISION_PROMPT.format(
        question=case.question, context=_format_context(output))
    return judge.judge(prompt)


def answer_correctness(judge: JudgeLLM, case: TestCase,
                       output: RAGOutput) -> Optional[JudgeVerdict]:
    if not case.reference_answer:
        return None
    prompt = CORRECTNESS_PROMPT.format(
        question=case.question,
        reference=case.reference_answer,
        answer=output.answer)
    return judge.judge(prompt)


JUDGE_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevance": answer_relevance,
    "context_precision": context_precision,
    "answer_correctness": answer_correctness,
}


def is_judge_metric(name: str) -> bool:
    return name in JUDGE_METRICS
