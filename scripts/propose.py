"""Diagnose the default pipeline on the train split and print proposed mutations."""

from __future__ import annotations

import argparse

from ragsearch.benchmark import doc_text, load_benchmark
from ragsearch.diagnose.cluster import diagnose_hybrid
from ragsearch.index.cache import cached_doc_embeddings
from ragsearch.index.embedder import DenseEmbedder
from ragsearch.search.build import build_pipeline
from ragsearch.search.config import DEFAULT_CONFIG
from ragsearch.search.mutate import MutationProposer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "optimize", "test"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    bench = load_benchmark()
    doc_ids = list(bench.corpus)
    texts = [doc_text(bench.corpus[d]) for d in doc_ids]
    corpus_map = dict(zip(doc_ids, texts))

    vecs = cached_doc_embeddings(DenseEmbedder(), doc_ids, texts)
    pipeline = build_pipeline(DEFAULT_CONFIG, corpus_map, doc_embeddings=vecs)

    report = diagnose_hybrid(
        pipeline, bench.subset(args.split), bench.qrels, k=args.k
    )
    print(f"\nbase config {DEFAULT_CONFIG.fingerprint()}  split={args.split}\n")
    print(report.summary())

    proposer = MutationProposer()
    mutations = proposer.propose(DEFAULT_CONFIG, report)
    print(f"\n{len(mutations)} proposed mutations:\n")
    for m in mutations:
        print(f"* {m.name}  ->  {m.fingerprint}   [{m.targets}]")
        print(f"    {m.rationale}")
        print(f"    delta: {m.param_delta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
