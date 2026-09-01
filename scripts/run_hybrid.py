"""Smoke-test hybrid fusion (and, optionally, the cross-encoder reranker).

Compares BM25 alone, dense alone, RRF fusion and weighted fusion on one
split, using the cached document embeddings. With ``--rerank`` it also runs
the cross-encoder over the RRF candidates (slow on CPU).
"""

from __future__ import annotations

import argparse
import time

from ragsearch.benchmark import doc_text, load_benchmark
from ragsearch.index.cache import cached_doc_embeddings
from ragsearch.index.embedder import DenseEmbedder
from ragsearch.index.inverted import InvertedIndex
from ragsearch.retrieve.bm25 import BM25Retriever
from ragsearch.retrieve.dense import DenseRetriever
from ragsearch.retrieve.fusion import HybridRetriever
from ragsearch.retrieve.rerank import CrossEncoderReranker


def recall_at_k(hits, relevant, k):
    got = {h.doc_id for h in hits[:k]}
    return len(got & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(hits, relevant, k):
    for rank, h in enumerate(hits[:k], start=1):
        if h.doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(name, search_fn, qids, bench, ks=(10, 100)):
    agg = {f"r@{k}": 0.0 for k in ks}
    agg["mrr@10"] = 0.0
    t0 = time.perf_counter()
    for qid in qids:
        relevant = {d for d, r in bench.qrels.get(qid, {}).items() if r > 0}
        hits = search_fn(bench.queries[qid])
        for k in ks:
            agg[f"r@{k}"] += recall_at_k(hits, relevant, k)
        agg["mrr@10"] += mrr_at_k(hits, relevant, 10)
    dt = time.perf_counter() - t0
    n = len(qids)
    row = "  ".join(f"{m}={v / n:.4f}" for m, v in agg.items())
    print(f"{name:<22} {row}   ({dt:.1f}s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="optimize", choices=["train", "optimize", "test"])
    ap.add_argument("--rerank", action="store_true", help="also run the cross-encoder")
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

    qids = [
        q for q in bench.splits[args.split]
        if any(r > 0 for r in bench.qrels.get(q, {}).values())
    ]
    print(f"split={args.split}  queries={len(qids)}\n")

    evaluate("bm25", lambda q: bm25.search(q, k=100), qids, bench)
    evaluate("dense", lambda q: dense.search(q, k=100), qids, bench)

    rrf = HybridRetriever(
        {"bm25": bm25, "dense": dense}, method="rrf", candidate_k=200
    )
    evaluate("hybrid-rrf", lambda q: rrf.search(q, k=100), qids, bench)

    weighted = HybridRetriever(
        {"bm25": bm25, "dense": dense}, method="weighted",
        weights={"bm25": 1.0, "dense": 1.0}, candidate_k=200,
    )
    evaluate("hybrid-weighted", lambda q: weighted.search(q, k=100), qids, bench)

    if args.rerank:
        reranked = HybridRetriever(
            {"bm25": bm25, "dense": dense}, method="rrf", candidate_k=200,
            corpus=corpus_map, reranker=CrossEncoderReranker(top_n=100),
        )
        evaluate("hybrid-rrf+ce", lambda q: reranked.search(q, k=100), qids, bench)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
