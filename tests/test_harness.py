"""Evaluation harness: scoring a run, driving a retriever, comparing runs."""

import pytest

from ragsearch import benchmark
from ragsearch.eval.harness import compare, evaluate_retrieval, score_run
from ragsearch.retrieve.base import Hit


class FakeRetriever:
    def __init__(self, table):
        self.table = table

    def search(self, query, k=10):
        return [Hit(d, 1.0 / i) for i, d in enumerate(self.table[query][:k], start=1)]


QRELS = {
    "q1": {"a": 1, "b": 1},
    "q2": {"c": 1},
    "q3": {"d": 1},  # not asked -> ignored by scoring
}


def test_score_run_skips_unjudged_queries():
    rankings = {"q1": ["a", "x", "b"], "q2": ["y", "c"], "qX": ["z"]}
    res = score_run("r", rankings, QRELS, ks=(1, 3))
    assert set(res.per_query) == {"q1", "q2"}  # qX has no qrels
    assert res.per_query["q1"]["recall@3"] == pytest.approx(1.0)
    assert res.per_query["q2"]["recall@1"] == pytest.approx(0.0)


def test_aggregate_is_mean_over_queries():
    rankings = {"q1": ["a", "b"], "q2": ["c"]}
    res = score_run("r", rankings, QRELS, ks=(1,))
    # q1 recall@1 = 0.5, q2 recall@1 = 1.0 -> mean 0.75
    assert res.aggregate()["recall@1"] == pytest.approx(0.75)


def test_evaluate_retrieval_end_to_end():
    retr = FakeRetriever({"q1": ["z", "a", "b"], "q2": ["c", "z"]})
    res = evaluate_retrieval("fake", retr, {"q1": "q1", "q2": "q2"}, QRELS, ks=(1, 5))
    assert res.aggregate()["hit@5"] == pytest.approx(1.0)
    assert res.aggregate()["ndcg@1"] == pytest.approx(0.5)  # q1 miss@1, q2 hit@1


def test_compare_returns_signed_delta():
    weak = score_run("weak", {"q1": ["x", "a", "b"], "q2": ["y", "c"]}, QRELS, ks=(1, 3))
    strong = score_run("strong", {"q1": ["a", "b", "x"], "q2": ["c", "y"]}, QRELS, ks=(1, 3))
    sig = compare(weak, strong, "mrr@3", iterations=2000)
    assert sig.delta > 0
    assert 0.0 <= sig.p_value <= 1.0


@pytest.mark.skipif(
    not (benchmark.SCIFACT_DIR / "corpus.jsonl").exists(),
    reason="run scripts/get_data.py to fetch the SciFact benchmark",
)
def test_hybrid_beats_bm25_on_scifact_optimize():
    from ragsearch.index.cache import cached_doc_embeddings
    from ragsearch.index.embedder import DenseEmbedder
    from ragsearch.index.inverted import InvertedIndex
    from ragsearch.retrieve.bm25 import BM25Retriever
    from ragsearch.retrieve.dense import DenseRetriever
    from ragsearch.retrieve.fusion import HybridRetriever

    bench = benchmark.load_benchmark()
    doc_ids = list(bench.corpus)
    texts = [benchmark.doc_text(bench.corpus[d]) for d in doc_ids]
    embedder = DenseEmbedder()
    try:
        vecs = cached_doc_embeddings(embedder, doc_ids, texts, verbose=False)
    except Exception as exc:  # model/network unavailable
        pytest.skip(f"embeddings unavailable: {exc}")

    bm25 = BM25Retriever(InvertedIndex.build(dict(zip(doc_ids, texts))))
    dense = DenseRetriever.build(
        {d: "" for d in doc_ids}, embedder=embedder, precomputed=vecs
    )
    hybrid = HybridRetriever(
        {"bm25": bm25, "dense": dense}, method="weighted",
        weights={"bm25": 1.0, "dense": 1.0}, candidate_k=200,
    )
    queries = bench.subset("optimize")
    base = evaluate_retrieval("bm25", bm25, queries, bench.qrels, depth=100)
    hyb = evaluate_retrieval("hybrid", hybrid, queries, bench.qrels, depth=100)

    assert hyb.aggregate()["ndcg@10"] > base.aggregate()["ndcg@10"]
    assert hyb.aggregate()["recall@100"] > base.aggregate()["recall@100"]
    sig = compare(base, hyb, "ndcg@10", iterations=5000)
    assert sig.delta > 0 and sig.p_value < 0.05
