"""
RAG system adapter interface.

To evaluate any RAG system, wrap it in an adapter that implements query(). The
adapter is responsible for translating the system's native output into the
framework's RAGOutput type, including assigning stable chunk ids that match the
ids used in your test set's relevant_ids.

This is the single integration point. Everything else in the framework works
against RAGOutput, so it does not care what RAG system produced it.
"""

from abc import ABC, abstractmethod

from ..types import RAGOutput


class RAGAdapter(ABC):
    @abstractmethod
    def query(self, question: str, session_id: str = "eval") -> RAGOutput:
        """Run one query through the wrapped RAG system and return a RAGOutput."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


class CallableAdapter(RAGAdapter):
    """
    Wrap a plain function `fn(question) -> RAGOutput` as an adapter. Handy for
    quick experiments without writing a class.
    """

    def __init__(self, fn, name: str = "callable"):
        self._fn = fn
        self._name = name

    def query(self, question: str, session_id: str = "eval") -> RAGOutput:
        return self._fn(question)

    @property
    def name(self) -> str:
        return self._name
