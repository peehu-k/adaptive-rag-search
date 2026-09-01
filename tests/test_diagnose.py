"""Failure-mode labelling and clustering."""

import pytest

from ragsearch import benchmark
from ragsearch.diagnose.cluster import FailureReport, diagnose_hybrid
from ragsearch.diagnose.failures import (
    FUSION_MISS,
    GENERATION_MISS,
    NO_GOLD,
    RECALL_MISS,
    RERANKER_MISS,
    SUCCESS,
    diagnose_query,
)


def _d(per_retriever, fused, final, gold=("g",), **kw):
    return diagnose_query(
        "q", list(gold), per_retriever=per_retriever, fused=fused, final=final, **kw
    )


def test_success_when_gold_in_final_topk():
    d = _d({"bm25": ["g", "x"]}, ["g", "x"], ["g", "x"], k=10)
    assert d.mode == SUCCESS
    assert d.best_final_rank == 1


def test_recall_miss_when_gold_in_no_pool():
    d = _d(
        {"bm25": ["a", "b", "c"], "dense": ["d", "e"]},
        ["a", "b", "c", "d", "e"],
        ["a", "b", "c", "d", "e"],
        k=3,
        pool_depth=200,
    )
    assert d.mode == RECALL_MISS
    assert d.found_by == []


def test_fusion_miss_when_gold_in_pool_but_not_topk():
    pool = ["x"] * 40 + ["g"] + ["y"] * 40
    d = _d(
        {"bm25": pool, "dense": ["z"]},
        pool,  # fused keeps it deep
        pool,
        k=10,
        pool_depth=200,
        has_reranker=False,
    )
    assert d.mode == FUSION_MISS
    assert d.found_by == ["bm25"]
    assert d.recoverable_by == ["bm25"]


def test_reranker_miss_when_reranker_drops_a_topk_gold():
    fused = ["g", "a", "b", "c"]          # gold was rank 1 pre-rerank
    final = ["a", "b", "c", "d", "g"]     # reranker pushed it to 5
    d = _d(
        {"bm25": ["g", "a", "b", "c"]},
        fused,
        final,
        k=3,
        has_reranker=True,
    )
    assert d.mode == RERANKER_MISS
    assert d.best_fused_rank == 1 and d.best_final_rank == 5


def test_no_reranker_means_deep_final_is_fusion_miss_not_reranker():
    fused = ["g", "a", "b", "c"]
    final = ["a", "b", "c", "d", "g"]
    d = _d({"bm25": ["g", "a"]}, fused, final, k=3, has_reranker=False)
    assert d.mode == FUSION_MISS


def test_generation_miss_when_context_good_but_answer_wrong():
    d = _d({"bm25": ["g"]}, ["g"], ["g", "x"], k=10, answer_correct=False)
    assert d.mode == GENERATION_MISS


def test_correct_answer_stays_success():
    d = _d({"bm25": ["g"]}, ["g"], ["g"], k=10, answer_correct=True)
    assert d.mode == SUCCESS


def test_no_gold_is_its_own_bucket():
    d = _d({"bm25": ["a"]}, ["a"], ["a"], gold=())
    assert d.mode == NO_GOLD
    assert not d.is_failure


def test_failure_report_counts_and_recoverable():
    diags = [
        _d({"bm25": ["g"]}, ["g"], ["g"], k=10),                       # success
        _d({"bm25": ["a"], "dense": ["b"]}, ["a", "b"], ["a", "b"], k=1),  # recall miss
        _d({"bm25": ["x"] * 30 + ["g"]}, ["x"] * 30 + ["g"],
           ["x"] * 30 + ["g"], k=10, pool_depth=200),                  # fusion miss (bm25)
        _d({"bm25": ["a"]}, ["a"], ["a"], gold=()),                    # no gold
    ]
    report = FailureReport(diagnoses=diags, k=10)
    counts = report.counts()
    assert counts[SUCCESS] == 1
    assert counts[RECALL_MISS] == 1
    assert counts[FUSION_MISS] == 1
    assert counts[NO_GOLD] == 1
    assert len(report.scored()) == 3
    assert report.recoverable_by_retriever() == {"bm25": 1}
    assert "fusion_miss" in report.summary()


@pytest.mark.skipif(
    not (benchmark.SCIFACT_DIR / "corpus.jsonl").exists(),
    reason="run scripts/get_data.py to fetch the SciFact benchmark",
)
def test_diagnose_real_hybrid_slice():
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
    except Exception as exc:
        pytest.skip(f"embeddings unavailable: {exc}")

    bm25 = BM25Retriever(InvertedIndex.build(dict(zip(doc_ids, texts))))
    dense = DenseRetriever.build(
        {d: "" for d in doc_ids}, embedder=embedder, precomputed=vecs
    )
    hybrid = HybridRetriever(
        {"bm25": bm25, "dense": dense}, method="weighted",
        weights={"bm25": 1.0, "dense": 1.0}, candidate_k=200,
    )
    ids = bench.splits["train"][:40]
    queries = {q: bench.queries[q] for q in ids}
    report = diagnose_hybrid(hybrid, queries, bench.qrels, k=10)

    # every scored query gets exactly one mode, and they partition the set
    assert sum(report.counts().values()) == len(queries)
    assert len(report.scored()) + report.counts()[NO_GOLD] == len(queries)
    # on 40 train queries we expect at least one genuine failure to inspect
    assert len(report.failures()) >= 1
