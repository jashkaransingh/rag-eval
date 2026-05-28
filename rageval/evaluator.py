"""
The Evaluator runs a set of metrics over a set of test cases against a RAG
adapter and aggregates the results.

Flow per test case:
  1. call adapter.query(question) to get a RAGOutput
  2. for each requested retrieval metric, compute it from retrieved_ids vs
     relevant_ids
  3. for each requested judge metric, call the judge LLM
  4. record per-case scores and rationales

Then aggregate into an EvalSummary with metric means, stds, and latency
percentiles.

NaN handling: retrieval metrics return NaN for cases without ground-truth
relevant_ids. Those NaNs are excluded from the mean so a partially-labeled test
set still produces meaningful retrieval numbers for the labeled subset.
"""

import statistics
from typing import Dict, List, Optional

from .adapters.base import RAGAdapter
from .judge_llm import JudgeLLM, build_judge
from .metrics import judges as judge_metrics
from .metrics import retrieval as retrieval_metrics
from .types import EvalResult, EvalSummary, TestCase


DEFAULT_METRICS = [
    "recall@5", "precision@5", "mrr", "ndcg@5",
    "faithfulness", "answer_relevance", "context_precision",
]


class Evaluator:
    def __init__(self, metrics: Optional[List[str]] = None,
                 judge: Optional[JudgeLLM] = None,
                 judge_backend: str = "anthropic",
                 default_k: int = 5,
                 verbose: bool = True):
        self.metrics = metrics or list(DEFAULT_METRICS)
        self.default_k = default_k
        self.verbose = verbose

        # Only build a judge if a judge metric was requested
        needs_judge = any(judge_metrics.is_judge_metric(m) for m in self.metrics)
        if needs_judge:
            self.judge = judge or build_judge(judge_backend)
        else:
            self.judge = None

    def evaluate_case(self, adapter: RAGAdapter, case: TestCase) -> EvalResult:
        output = adapter.query(case.question)
        scores: Dict[str, float] = {}
        rationales: Dict[str, str] = {}

        for metric in self.metrics:
            if retrieval_metrics.is_retrieval_metric(metric):
                scores[metric] = retrieval_metrics.compute_retrieval_metric(
                    metric, output.retrieved_ids, case.relevant_ids,
                    self.default_k)
            elif judge_metrics.is_judge_metric(metric):
                fn = judge_metrics.JUDGE_METRICS[metric]
                verdict = fn(self.judge, case, output)
                if verdict is None:
                    # e.g. correctness with no reference answer, skip
                    continue
                scores[metric] = verdict.score
                rationales[metric] = verdict.rationale
            else:
                raise ValueError(f"unknown metric: {metric}")

        return EvalResult(test_case=case, output=output,
                          scores=scores, rationales=rationales)

    def run(self, adapter: RAGAdapter,
            test_cases: List[TestCase]) -> EvalSummary:
        results: List[EvalResult] = []
        for i, case in enumerate(test_cases, 1):
            if self.verbose:
                print(f"  [{i}/{len(test_cases)}] {case.question[:60]}")
            results.append(self.evaluate_case(adapter, case))

        return self._summarize(results, adapter)

    def _summarize(self, results: List[EvalResult],
                   adapter: RAGAdapter) -> EvalSummary:
        # Collect per-metric score lists, dropping NaNs
        metric_values: Dict[str, List[float]] = {m: [] for m in self.metrics}
        for r in results:
            for m, v in r.scores.items():
                if v == v:  # not NaN
                    metric_values.setdefault(m, []).append(v)

        means = {}
        stds = {}
        for m, vals in metric_values.items():
            if vals:
                means[m] = statistics.fmean(vals)
                stds[m] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

        latencies = [r.output.latency_ms for r in results]
        latencies_sorted = sorted(latencies)

        def percentile(data, p):
            if not data:
                return 0.0
            idx = min(len(data) - 1, int(round(p / 100 * (len(data) - 1))))
            return data[idx]

        total_prompt = sum(r.output.prompt_tokens or 0 for r in results)
        total_completion = sum(r.output.completion_tokens or 0 for r in results)

        return EvalSummary(
            n_cases=len(results),
            metric_means=means,
            metric_stds=stds,
            latency_p50_ms=percentile(latencies_sorted, 50),
            latency_p95_ms=percentile(latencies_sorted, 95),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            results=results,
            config={
                "adapter": adapter.name,
                "judge": self.judge.name if self.judge else "none",
                "metrics": ", ".join(self.metrics),
            },
        )


def compare(adapter, test_cases, configs: Dict[str, List[str]],
            judge_backend: str = "anthropic") -> Dict[str, EvalSummary]:
    """
    Convenience for A/B comparisons. configs maps a label to a metric list.
    Returns label -> EvalSummary. Useful for comparing the same system under
    different retrieval settings, but here it just reuses one adapter.
    """
    out = {}
    for label, metrics in configs.items():
        ev = Evaluator(metrics=metrics, judge_backend=judge_backend,
                       verbose=False)
        out[label] = ev.run(adapter, test_cases)
    return out
