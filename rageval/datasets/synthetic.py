"""
Test set loading and synthetic generation.

A test set is a list of TestCase. You can load one from JSON, or generate one
synthetically from a corpus using an LLM (or the stub, which produces simple
template questions for offline use).

JSON format:
[
  {
    "question": "What was Q3 revenue?",
    "relevant_ids": ["acme_q3.md:0"],
    "reference_answer": "$42 million",
    "metadata": {"category": "financials"}
  },
  ...
]
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from ..judge_llm import JudgeLLM, build_judge
from ..types import TestCase


def load_testset(path: str) -> List[TestCase]:
    data = json.loads(Path(path).read_text())
    cases = []
    for item in data:
        cases.append(TestCase(
            question=item["question"],
            relevant_ids=item.get("relevant_ids", []),
            reference_answer=item.get("reference_answer"),
            metadata=item.get("metadata", {}),
        ))
    return cases


def save_testset(cases: List[TestCase], path: str) -> None:
    data = []
    for c in cases:
        data.append({
            "question": c.question,
            "relevant_ids": c.relevant_ids,
            "reference_answer": c.reference_answer,
            "metadata": c.metadata,
        })
    Path(path).write_text(json.dumps(data, indent=2))


GEN_PROMPT = """\
You are creating a QA test set from a document chunk. Read the chunk and write
{n} question and answer pairs that can be answered using only this chunk. The
questions should be specific and varied. Return strict JSON, a list of objects
each with "question" and "answer" keys, nothing else.

CHUNK (id: {chunk_id}):
{text}
"""


def generate_from_chunks(chunks: List[dict], judge: Optional[JudgeLLM] = None,
                         questions_per_chunk: int = 2,
                         backend: str = "anthropic") -> List[TestCase]:
    """
    Generate a synthetic test set from chunks.

    Each chunk is a dict with keys: chunk_id, text, source. For each chunk we
    ask the LLM to write QA pairs, then build TestCases whose relevant_ids point
    back at the source chunk.

    With the stub backend, this produces simple deterministic questions so the
    flow is testable offline without an API key.
    """
    llm = judge or build_judge(backend)
    cases: List[TestCase] = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if llm.name == "stub":
            # Deterministic offline path, derive a question from the first
            # sentence of the chunk
            first = re.split(r"(?<=[.!?])\s", chunk["text"].strip())[0]
            q = f"What does the document say about: {first[:60]}?"
            cases.append(TestCase(
                question=q,
                relevant_ids=[chunk_id],
                reference_answer=first[:200],
                metadata={"source": chunk.get("source", ""), "synthetic": "stub"},
            ))
            continue

        prompt = GEN_PROMPT.format(
            n=questions_per_chunk, chunk_id=chunk_id, text=chunk["text"][:2000])
        verdict = llm.judge(prompt)  # reuse the judge LLM call surface
        pairs = _parse_qa_pairs(verdict.rationale or "")
        for qa in pairs:
            cases.append(TestCase(
                question=qa["question"],
                relevant_ids=[chunk_id],
                reference_answer=qa.get("answer"),
                metadata={"source": chunk.get("source", ""), "synthetic": "llm"},
            ))

    return cases


def _parse_qa_pairs(text: str) -> List[dict]:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\[.*\]", cleaned, re.S)
    if not match:
        return []
    try:
        arr = json.loads(match.group(0))
        return [{"question": x["question"], "answer": x.get("answer")}
                for x in arr if "question" in x]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
