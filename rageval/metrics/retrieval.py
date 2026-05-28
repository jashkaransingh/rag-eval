"""
Retrieval metrics.

These measure whether the retriever surfaced the right chunks, independent of
what the LLM did with them. They all need ground-truth relevant chunk ids in
the TestCase. If a test case has no relevant_ids, these metrics are skipped for
that case.

Implemented:
  - hit@k        did we retrieve at least one relevant chunk in the top k
  - recall@k     fraction of all relevant chunks that appear in the top k
  - precision@k  fraction of the top k that are relevant
  - mrr          reciprocal rank of the first relevant chunk
  - ndcg@k       normalized discounted cumulative gain, rewards ranking
                 relevant chunks higher
"""

import math
from typing import List


def hit_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    top_k = retrieved_ids[:k]
    return 1.0 if any(r in relevant_ids for r in top_k) else 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    hits = len(top_k & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not relevant_ids or k == 0:
        return float("nan")
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    if not relevant_ids:
        return float("nan")
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Binary-relevance NDCG. Each retrieved chunk is relevant (gain 1) or not
    (gain 0). DCG discounts gains by log2 of position. IDCG is the best
    possible DCG given the number of relevant chunks.
    """
    if not relevant_ids:
        return float("nan")
    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]

    dcg = 0.0
    for i, rid in enumerate(top_k, start=1):
        if rid in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# Registry so the evaluator can request metrics by name like "recall@5"

def parse_at_k(name: str, default_k: int = 5):
    """Split a metric name like 'recall@5' into ('recall', 5)."""
    if "@" in name:
        base, k_str = name.split("@", 1)
        return base, int(k_str)
    return name, default_k


def compute_retrieval_metric(name: str, retrieved_ids: List[str],
                             relevant_ids: List[str], default_k: int = 5) -> float:
    base, k = parse_at_k(name, default_k)
    if base == "hit":
        return hit_at_k(retrieved_ids, relevant_ids, k)
    if base == "recall":
        return recall_at_k(retrieved_ids, relevant_ids, k)
    if base == "precision":
        return precision_at_k(retrieved_ids, relevant_ids, k)
    if base == "mrr":
        return mrr(retrieved_ids, relevant_ids)
    if base == "ndcg":
        return ndcg_at_k(retrieved_ids, relevant_ids, k)
    raise ValueError(f"unknown retrieval metric: {name}")


RETRIEVAL_METRICS = {"hit", "recall", "precision", "mrr", "ndcg"}


def is_retrieval_metric(name: str) -> bool:
    base, _ = parse_at_k(name)
    return base in RETRIEVAL_METRICS
