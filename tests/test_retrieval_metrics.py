import math

from rageval.metrics import retrieval as R


def test_hit_at_k():
    assert R.hit_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
    assert R.hit_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
    assert R.hit_at_k(["a", "b"], ["z"], k=2) == 0.0


def test_recall_at_k():
    assert R.recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0
    assert R.recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5
    assert R.recall_at_k(["x", "y"], ["a"], k=2) == 0.0


def test_precision_at_k():
    assert R.precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0
    assert abs(R.precision_at_k(["a", "x", "y"], ["a"], k=3) - 1 / 3) < 1e-9


def test_mrr():
    assert R.mrr(["a", "b", "c"], ["a"]) == 1.0
    assert R.mrr(["x", "a", "c"], ["a"]) == 0.5
    assert abs(R.mrr(["x", "y", "a"], ["a"]) - 1 / 3) < 1e-9
    assert R.mrr(["x", "y"], ["a"]) == 0.0


def test_ndcg_perfect_ranking_is_one():
    # Two relevant docs ranked first two = perfect
    score = R.ndcg_at_k(["a", "b", "c", "d"], ["a", "b"], k=4)
    assert abs(score - 1.0) < 1e-9


def test_ndcg_worse_ranking_is_lower():
    good = R.ndcg_at_k(["a", "b", "x", "y"], ["a", "b"], k=4)
    bad = R.ndcg_at_k(["x", "y", "a", "b"], ["a", "b"], k=4)
    assert good > bad


def test_metrics_return_nan_without_relevant_ids():
    assert math.isnan(R.recall_at_k(["a"], [], k=3))
    assert math.isnan(R.mrr(["a"], []))
    assert math.isnan(R.ndcg_at_k(["a"], [], k=3))


def test_parse_at_k():
    assert R.parse_at_k("recall@10") == ("recall", 10)
    assert R.parse_at_k("mrr", default_k=5) == ("mrr", 5)


def test_compute_retrieval_metric_dispatch():
    assert R.compute_retrieval_metric("hit@2", ["a", "b"], ["b"]) == 1.0
    assert R.compute_retrieval_metric("recall@2", ["a", "b"], ["a", "b"]) == 1.0


def test_is_retrieval_metric():
    assert R.is_retrieval_metric("recall@5")
    assert R.is_retrieval_metric("mrr")
    assert not R.is_retrieval_metric("faithfulness")
