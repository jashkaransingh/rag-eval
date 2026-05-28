from .base import CallableAdapter, RAGAdapter

__all__ = ["RAGAdapter", "CallableAdapter"]

# RagqaAdapter is imported lazily to avoid a hard dependency on rag-document-qa
def __getattr__(name):
    if name == "RagqaAdapter":
        from .ragqa_adapter import RagqaAdapter
        return RagqaAdapter
    raise AttributeError(name)
