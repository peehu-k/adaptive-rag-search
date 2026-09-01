"""Evaluate BM25 / dense / hybrid on a split and significance-test the gaps."""

from __future__ import annotations

import argparse

from ragsearch.benchmark import doc_text, load_benchmark
from ragsearch.eval.harness import compare, evaluate_retrieval, format_table
from ragsearch.index.cache import cached_doc_embeddings
from ragsearch.index.embedder import DenseEmbedder
from ragsearch.index.inverted import InvertedIndex
from ragsearch.retrieve.bm25 import BM25Retriever
from ragsearch.retrieve.dense import DenseRetriever
from ragsearch.retrieve.fusion import HybridRetriever

METRICS = ["ndcg@10", "recall@10", "recall@100", "mrr@10", "map@100"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="optimize", choices=["train", "optimize", "test"])
    ap.add_argument("--method", default="randomization",
                    choices=["randomization", "bootstrap"])
    args = ap.parse_args()

    bench = load_benchmark()
    doc_ids = list(bench.corpus)
    texts = [doc_text(bench.corpus[d]) for d in doc_ids]
    corpus_map = dict(zip(doc_ids, texts))
    queries = bench.subset(args.split)

    bm25 = BM25Retriever(InvertedIndex.build(corpus_map))
    embedder = DenseEmbedder()
    vecs = cached_doc_embeddings(embedder, doc_ids, texts)
    dense = DenseRetriever.build(
        {d: "" for d in doc_ids}, embedder=embedder, precomputed=vecs
    )
    hybrid = HybridRetriever(
        {"bm25": bm25, "dense": dense}, method="weighted",
        weights={"bm25": 1.0, "dense": 1.0}, candidate_k=200,
    )

    runs = [
        evaluate_retrieval("bm25", bm25, queries, bench.qrels, depth=100),
        evaluate_retrieval("dense", dense, queries, bench.qrels, depth=100),
        evaluate_retrieval("hybrid-weighted", hybrid, queries, bench.qrels, depth=100),
    ]

    print(f"\nsplit={args.split}  scored_queries={len(runs[0].per_query)}\n")
    print(format_table(runs, METRICS))

    print("\nsignificance vs bm25:")
    for cand in runs[1:]:
        for metric in ["ndcg@10", "recall@100"]:
            print("  " + str(compare(runs[0], cand, metric, method=args.method)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
