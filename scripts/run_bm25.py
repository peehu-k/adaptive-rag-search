"""Build the BM25 index over SciFact and run a smoke check on one split.

The numbers printed here (recall@k, MRR@10) are a sanity check that the index
and scorer work -- the real evaluation harness with significance testing comes
later. Run against the ``optimize`` split by default.
"""

from __future__ import annotations

import argparse
import time

from ragsearch.benchmark import doc_text, load_benchmark
from ragsearch.index.inverted import InvertedIndex
from ragsearch.index.tokenizer import Analyzer
from ragsearch.retrieve.bm25 import BM25Retriever


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
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--stem", action="store_true", help="enable light stemming")
    ap.add_argument("--keep-stopwords", action="store_true")
    args = ap.parse_args()

    bench = load_benchmark()
    analyzer = Analyzer(stem=args.stem, remove_stopwords=not args.keep_stopwords)

    documents = {did: doc_text(doc) for did, doc in bench.corpus.items()}
    t0 = time.perf_counter()
    index = InvertedIndex.build(documents, analyzer=analyzer)
    build_s = time.perf_counter() - t0

    print(f"split={args.split}  analyzer={analyzer}")
    print(
        f"index: {index.num_docs} docs, {len(index.postings)} terms, "
        f"avgdl={index.avg_doc_len:.1f}, built in {build_s:.2f}s"
    )

    retriever = BM25Retriever(index, k1=args.k1, b=args.b)
    queries = bench.subset(args.split)

    agg = {"r@10": 0.0, "r@100": 0.0, "mrr@10": 0.0}
    scored = 0
    t0 = time.perf_counter()
    for qid, qtext in queries.items():
        relevant = {d for d, rel in bench.qrels.get(qid, {}).items() if rel > 0}
        if not relevant:
            continue
        hits = retriever.search(qtext, k=100)
        agg["r@10"] += recall_at_k(hits, relevant, 10)
        agg["r@100"] += recall_at_k(hits, relevant, 100)
        agg["mrr@10"] += mrr_at_k(hits, relevant, 10)
        scored += 1
    query_s = time.perf_counter() - t0

    print(f"scored {scored} queries in {query_s:.2f}s "
          f"({1000 * query_s / max(scored, 1):.1f} ms/query)")
    for name, total in agg.items():
        print(f"  {name:<7}= {total / max(scored, 1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
