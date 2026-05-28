"""
Adapter for the rag-document-qa system.

This bridges the rag-document-qa RAGSystem into the eval framework. Chunk ids
are assigned as "{source}:{chunk_id}" so they are stable across runs and can be
referenced in a test set's relevant_ids.

Usage:
    from ragqa import RAGSystem, RAGConfig
    from rageval.adapters import RagqaAdapter

    rag = RAGSystem(config=RAGConfig(embedder="tfidf", llm="stub"))
    rag.ingest_directory("docs")
    adapter = RagqaAdapter(rag)

The import of ragqa is done lazily so the eval framework does not hard-depend on
rag-document-qa being installed. If you are evaluating a different system, you
never touch this file.
"""

from ..types import RAGOutput, RetrievedChunk
from .base import RAGAdapter


class RagqaAdapter(RAGAdapter):
    def __init__(self, rag_system):
        self._rag = rag_system

    def query(self, question: str, session_id: str = "eval") -> RAGOutput:
        result = self._rag.query(question, session_id=session_id)

        retrieved = []
        for chunk, score in zip(result.sources, result.scores):
            chunk_id = f"{chunk.source}:{chunk.chunk_id}"
            retrieved.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk.text,
                score=float(score),
                source=chunk.source,
            ))

        prompt_tokens = None
        completion_tokens = None
        if result.llm_response is not None:
            prompt_tokens = result.llm_response.prompt_tokens
            completion_tokens = result.llm_response.completion_tokens

        return RAGOutput(
            answer=result.answer,
            retrieved=retrieved,
            latency_ms=result.latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @property
    def name(self) -> str:
        return "ragqa"
