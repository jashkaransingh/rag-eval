from rageval import (CallableAdapter, Evaluator, RAGOutput, RetrievedChunk,
                     TestCase)
from rageval.report import to_html, to_markdown


def fake_rag(question: str) -> RAGOutput:
    """A deterministic fake RAG for testing the evaluator wiring."""
    if "revenue" in question.lower():
        return RAGOutput(
            answer="Revenue was 42 million dollars.",
            retrieved=[
                RetrievedChunk("doc:0", "Revenue was 42 million dollars.", 0.9),
                RetrievedChunk("doc:9", "unrelated text about weather", 0.3),
            ],
            latency_ms=10.0)
    return RAGOutput(
        answer="I do not know.",
        retrieved=[RetrievedChunk("doc:5", "some other content", 0.5)],
        latency_ms=20.0)


def test_evaluator_runs_retrieval_metrics():
    adapter = CallableAdapter(fake_rag, name="fake")
    cases = [
        TestCase(question="What was the revenue?",
                 relevant_ids=["doc:0"]),
        TestCase(question="What is the weather?",
                 relevant_ids=["doc:5"]),
    ]
    ev = Evaluator(metrics=["hit@2", "recall@2", "mrr"], verbose=False)
    summary = ev.run(adapter, cases)

    assert summary.n_cases == 2
    assert "hit@2" in summary.metric_means
    # both cases should hit their relevant doc
    assert summary.metric_means["hit@2"] == 1.0


def test_evaluator_with_judge_metrics():
    adapter = CallableAdapter(fake_rag, name="fake")
    cases = [TestCase(question="What was the revenue?",
                      relevant_ids=["doc:0"],
                      reference_answer="42 million dollars")]
    ev = Evaluator(metrics=["faithfulness", "answer_correctness"],
                   judge_backend="stub", verbose=False)
    summary = ev.run(adapter, cases)
    assert "faithfulness" in summary.metric_means
    assert "answer_correctness" in summary.metric_means


def test_evaluator_latency_percentiles():
    adapter = CallableAdapter(fake_rag, name="fake")
    cases = [TestCase(question="What was the revenue?", relevant_ids=["doc:0"]),
             TestCase(question="something else", relevant_ids=["doc:5"])]
    ev = Evaluator(metrics=["hit@2"], verbose=False)
    summary = ev.run(adapter, cases)
    assert summary.latency_p50_ms > 0
    assert summary.latency_p95_ms >= summary.latency_p50_ms


def test_no_judge_built_when_only_retrieval_metrics():
    ev = Evaluator(metrics=["recall@5"], verbose=False)
    assert ev.judge is None


def test_report_markdown_renders():
    adapter = CallableAdapter(fake_rag, name="fake")
    cases = [TestCase(question="What was the revenue?", relevant_ids=["doc:0"])]
    ev = Evaluator(metrics=["hit@2", "mrr"], verbose=False)
    summary = ev.run(adapter, cases)
    md = to_markdown(summary)
    assert "# RAG evaluation report" in md
    assert "hit@2" in md


def test_report_html_renders():
    adapter = CallableAdapter(fake_rag, name="fake")
    cases = [TestCase(question="What was the revenue?", relevant_ids=["doc:0"])]
    ev = Evaluator(metrics=["hit@2"], verbose=False)
    summary = ev.run(adapter, cases)
    html_out = to_html(summary)
    assert "<!DOCTYPE html>" in html_out
    assert "RAG evaluation report" in html_out
