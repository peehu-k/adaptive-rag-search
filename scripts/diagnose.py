"""Cluster the failures of the weighted-hybrid pipeline on the train split.

Diagnosis is a train-time activity: it looks at *why* queries miss so the
mutation proposer has something to act on. The optimize and test splits are
left alone here.
"""

from __future__ import annotations

import argparse

from ragsearch.benchmark import doc_text, load_benchmark
from ragsearch.diagnose.cluster import diagnose_hybrid
from ragsearch.diagnose.failures import FAILURE_MODES
from ragsearch.index.cache import cached_doc_embeddings
from ragsearch.index.embedder import DenseEmbedder
from ragsearch.index.inverted import InvertedIndex
from ragsearch.retrieve.bm25 import BM25Retriever
from ragsearch.retrieve.dense import DenseRetriever
from ragsearch.retrieve.fusion import HybridRetriever
from ragsearch.retrieve.rerank import CrossEncoderReranker, IdentityReranker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "optimize", "test"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--method", default="weighted", choices=["rrf", "weighted"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    bench = load_benchmark()
    doc_ids = list(bench.corpus)
    texts = [doc_text(bench.corpus[d]) for d in doc_ids]
    corpus_map = dict(zip(doc_ids, texts))

    bm25 = BM25Retriever(InvertedIndex.build(corpus_map))
    embedder = DenseEmbedder()
    vecs = cached_doc_embeddings(embedder, doc_ids, texts)
    dense = DenseRetriever.build(
        {d: "" for d in doc_ids}, embedder=embedder, precomputed=vecs
    )
    hybrid = HybridRetriever(
        {"bm25": bm25, "dense": dense},
        method=args.method,
        weights={"bm25": 1.0, "dense": 1.0},
        candidate_k=200,
        corpus=corpus_map,
        reranker=CrossEncoderReranker(top_n=100) if args.rerank else IdentityReranker(),
    )

    queries = bench.subset(args.split)
    report = diagnose_hybrid(hybrid, queries, bench.qrels, k=args.k)

    print(f"\nsplit={args.split}  method={args.method}  rerank={args.rerank}\n")
    print(report.summary())

    for mode in FAILURE_MODES:
        ex = report.examples(mode, args.examples)
        if not ex:
            continue
        print(f"\n--- {mode} (showing {len(ex)}) ---")
        for d in ex:
            q = queries[d.qid]
            q = q if len(q) <= 90 else q[:87] + "..."
            print(f"  [{d.qid}] pool={d.best_pool_rank} fused={d.best_fused_rank} "
                  f"final={d.best_final_rank} found_by={d.found_by or '-'}")
            print(f"       {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
