"""
Command line interface for running an evaluation.

Example:
    python3 -m scripts.eval \\
        --docs ../rag-document-qa/data/sample_docs \\
        --testset examples/testset.json \\
        --report out/report.html \\
        --judge stub

This wires up a rag-document-qa system as the target, but the --module/--factory
options let you point at any adapter factory you write.
"""

import argparse
import importlib
import os
import sys

from rageval import Evaluator
from rageval.datasets import load_testset
from rageval.report import save_report


def build_ragqa_adapter(docs_dir, embedder, llm, retrieval, k,
                        chunk_size, chunk_overlap):
    """Default adapter factory, builds and populates a rag-document-qa system."""
    from ragqa import RAGConfig, RAGSystem
    from rageval.adapters import RagqaAdapter

    rag = RAGSystem(config=RAGConfig(
        embedder=embedder, llm=llm, retrieval=retrieval, k=k,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    n = rag.ingest_directory(docs_dir)
    print(f"ingested {n} chunks from {docs_dir}")
    return RagqaAdapter(rag)


def main():
    p = argparse.ArgumentParser(description="Run a RAG evaluation")
    p.add_argument("--testset", required=True, help="path to testset json")
    p.add_argument("--docs", help="docs dir to ingest into the default ragqa adapter")
    p.add_argument("--report", help="output report path (.html or .md)")
    p.add_argument("--judge", default="anthropic", choices=["anthropic", "stub"])
    p.add_argument("--metrics", nargs="+", default=None,
                   help="metrics to compute, defaults to a standard set")
    # ragqa adapter knobs
    p.add_argument("--embedder", default="tfidf",
                   choices=["sentence-transformers", "tfidf"])
    p.add_argument("--llm", default="stub", choices=["anthropic", "stub"])
    p.add_argument("--retrieval", default="mmr", choices=["mmr", "topk"])
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=600)
    p.add_argument("--chunk-overlap", type=int, default=80)
    # custom adapter
    p.add_argument("--module", help="import path of a module with a factory")
    p.add_argument("--factory", default="build_adapter",
                   help="name of a zero-arg factory function in --module")
    args = p.parse_args()

    if args.module:
        mod = importlib.import_module(args.module)
        adapter = getattr(mod, args.factory)()
    elif args.docs:
        adapter = build_ragqa_adapter(
            args.docs, args.embedder, args.llm, args.retrieval, args.k,
            args.chunk_size, args.chunk_overlap)
    else:
        p.error("provide either --docs (default ragqa adapter) or --module/--factory")

    cases = load_testset(args.testset)
    print(f"loaded {len(cases)} test cases")

    metrics = args.metrics or [
        "hit@3", "recall@3", "precision@3", "mrr", "ndcg@3",
        "faithfulness", "answer_relevance", "context_precision",
        "answer_correctness",
    ]
    ev = Evaluator(metrics=metrics, judge_backend=args.judge)
    summary = ev.run(adapter, cases)

    print("\nmetrics")
    for m, mean in summary.metric_means.items():
        print(f"  {m:22s} {mean:.3f}")
    print(f"\nlatency p50 {summary.latency_p50_ms:.1f}ms "
          f"p95 {summary.latency_p95_ms:.1f}ms")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        save_report(summary, args.report)
        print(f"\nwrote report to {args.report}")


if __name__ == "__main__":
    main()
