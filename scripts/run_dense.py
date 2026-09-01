"""Build the dense index over SciFact and smoke-test it on one split.

Document embeddings are cached under ``data/scifact/cache/`` so re-runs skip
the encode. As with the BM25 script, the metrics here are a sanity check, not
the real evaluation harness.
"""

from __future__ import annotations

import argparse
import time

from ragsearch.benchmark import doc_text, load_benchmark
from ragsearch.index.cache import cached_doc_embeddings
from ragsearch.index.embedder import DEFAULT_MODEL, DenseEmbedder
from ragsearch.retrieve.dense import DenseRetriever


def recall_at_k(hits, relevant, k):
    got = {h.doc_id for h in hits[:k]}
    return len(got & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(hits, relevant, k):
    for rank, h in enumerate(hits[:k], start=1):
        if h.doc_id in relevant:
            return 1.0 / rank
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="optimize", choices=["train", "optimize", "test"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--index-type", default="flat", choices=["flat", "hnsw"])
    args = ap.parse_args()

    bench = load_benchmark()
    embedder = DenseEmbedder(model_name=args.model)

    doc_ids = list(bench.corpus)
    texts = [doc_text(bench.corpus[d]) for d in doc_ids]
    vecs = cached_doc_embeddings(embedder, doc_ids, texts)

    retr = DenseRetriever.build(
        {d: "" for d in doc_ids}, embedder=embedder,
        index_type=args.index_type, precomputed=vecs,
    )
    print(f"split={args.split}  model={args.model}  dim={vecs.shape[1]}  "
          f"index={args.index_type}")

    queries = bench.subset(args.split)
    agg = {"r@10": 0.0, "r@100": 0.0, "mrr@10": 0.0}
    scored = 0
    t0 = time.perf_counter()
    q_ids = [q for q in queries if any(
        r > 0 for r in bench.qrels.get(q, {}).values())]
    results = retr.search_many([queries[q] for q in q_ids], k=100)
    for qid, hits in zip(q_ids, results):
        relevant = {d for d, r in bench.qrels.get(qid, {}).items() if r > 0}
        agg["r@10"] += recall_at_k(hits, relevant, 10)
        agg["r@100"] += recall_at_k(hits, relevant, 100)
        agg["mrr@10"] += mrr_at_k(hits, relevant, 10)
        scored += 1
    dt = time.perf_counter() - t0

    print(f"scored {scored} queries in {dt:.2f}s "
          f"({1000 * dt / max(scored, 1):.1f} ms/query incl. encode)")
    for name, total in agg.items():
        print(f"  {name:<7}= {total / max(scored, 1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
