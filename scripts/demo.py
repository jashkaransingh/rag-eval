"""
Demo: evaluate the rag-document-qa system and compare top-k vs MMR retrieval.

Builds two configurations of the same RAG system, runs the eval suite against
both, and prints a side-by-side comparison plus saves an HTML report.

Run with the stub backend (offline) or anthropic (real judge + LLM):
    python3 scripts/demo.py --judge stub --llm stub
    python3 scripts/demo.py --judge anthropic --llm anthropic
"""

import argparse
import sys

from rageval import Evaluator
from rageval.datasets import load_testset
from rageval.report import save_report


METRICS = ["hit@3", "recall@3", "precision@3", "mrr", "ndcg@3",
           "faithfulness", "answer_relevance", "context_precision",
           "answer_correctness"]


def build_adapter(retrieval, embedder, llm, docs_dir):
    from ragqa import RAGConfig, RAGSystem
    from rageval.adapters import RagqaAdapter
    rag = RAGSystem(config=RAGConfig(
        embedder=embedder, llm=llm, retrieval=retrieval, k=3,
        chunk_size=600, chunk_overlap=80))
    rag.ingest_directory(docs_dir)
    return RagqaAdapter(rag)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="../rag-document-qa/data/sample_docs")
    p.add_argument("--testset", default="examples/testset.json")
    p.add_argument("--judge", default="stub", choices=["anthropic", "stub"])
    p.add_argument("--llm", default="stub", choices=["anthropic", "stub"])
    p.add_argument("--report", default="samples/report.html")
    args = p.parse_args()

    cases = load_testset(args.testset)
    print(f"loaded {len(cases)} test cases\n")

    summaries = {}
    for retrieval in ["topk", "mmr"]:
        print(f"=== evaluating retrieval={retrieval} ===")
        adapter = build_adapter(retrieval, "tfidf", args.llm, args.docs)
        ev = Evaluator(metrics=METRICS, judge_backend=args.judge, verbose=False)
        summaries[retrieval] = ev.run(adapter, cases)
        print()

    # Side-by-side table
    print("comparison (top-k vs mmr)")
    print(f"{'metric':<22}{'topk':>8}{'mmr':>8}")
    print("-" * 38)
    all_metrics = list(summaries["mmr"].metric_means.keys())
    for m in all_metrics:
        t = summaries["topk"].metric_means.get(m, float("nan"))
        v = summaries["mmr"].metric_means.get(m, float("nan"))
        print(f"{m:<22}{t:>8.3f}{v:>8.3f}")

    # Save the MMR report as the headline artifact
    save_report(summaries["mmr"], args.report)
    print(f"\nsaved report to {args.report}")
    if args.report.endswith(".html"):
        md_path = args.report.rsplit(".", 1)[0] + ".md"
        save_report(summaries["mmr"], md_path)
        print(f"saved report to {md_path}")


if __name__ == "__main__":
    main()
