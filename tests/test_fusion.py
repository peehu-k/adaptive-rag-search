"""Fusion math, the hybrid retriever, and the reranker hook."""

import pytest

from ragsearch.retrieve.base import Hit
from ragsearch.retrieve.fusion import (
    HybridRetriever,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from ragsearch.retrieve.rerank import IdentityReranker


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, k=10):
        return list(self._hits[:k])


def test_rrf_scores_match_formula():
    a = [Hit("d1", 9), Hit("d2", 8), Hit("d3", 7)]
    b = [Hit("d3", 5), Hit("d1", 4)]
    fused = {h.doc_id: h.score for h in reciprocal_rank_fusion([a, b], rrf_k=60)}
    assert fused["d1"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["d3"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["d2"] == pytest.approx(1 / 62)
    # d1 (1st + 2nd) outranks d3 (3rd + 1st) outranks d2 (2nd only)
    assert fused["d1"] > fused["d3"] > fused["d2"]


def test_rrf_weights_shift_the_winner():
    a = [Hit("x", 1), Hit("y", 1)]
    b = [Hit("y", 1), Hit("x", 1)]
    even = reciprocal_rank_fusion([a, b])
    assert {h.doc_id for h in even} == {"x", "y"}
    biased = reciprocal_rank_fusion([a, b], weights=[3.0, 1.0])
    assert biased[0].doc_id == "x"


def test_rrf_is_a_union_of_inputs():
    a = [Hit("only_a", 1)]
    b = [Hit("only_b", 1)]
    fused = reciprocal_rank_fusion([a, b])
    assert {h.doc_id for h in fused} == {"only_a", "only_b"}


def test_weighted_fusion_minmax_and_zero_weight():
    strong = [Hit("d1", 100), Hit("d2", 50), Hit("d3", 0)]
    noise = [Hit("d3", 1.0), Hit("d2", 0.9), Hit("d1", 0.8)]
    # noise down-weighted to zero -> ranking is exactly `strong`
    fused = weighted_score_fusion([strong, noise], weights=[1.0, 0.0])
    assert [h.doc_id for h in fused] == ["d1", "d2", "d3"]


def test_weighted_fusion_length_mismatch_raises():
    with pytest.raises(ValueError):
        weighted_score_fusion([[Hit("a", 1)]], weights=[1.0, 2.0])


def test_hybrid_retriever_promotes_consensus_docs():
    # a and b are near the top of both lists; c/d/e/f appear in only one
    r1 = FakeRetriever([Hit("a", 4), Hit("b", 3), Hit("c", 2), Hit("d", 1)])
    r2 = FakeRetriever([Hit("b", 4), Hit("a", 3), Hit("e", 2), Hit("f", 1)])
    hybrid = HybridRetriever({"r1": r1, "r2": r2}, method="rrf", candidate_k=10)
    out = hybrid.search("q", k=4)
    assert {out[0].doc_id, out[1].doc_id} == {"a", "b"}


def test_hybrid_identity_reranker_is_noop():
    r1 = FakeRetriever([Hit("a", 3), Hit("b", 2)])
    hybrid = HybridRetriever(
        {"r1": r1}, method="rrf", candidate_k=10, reranker=IdentityReranker()
    )
    assert [h.doc_id for h in hybrid.search("q", k=2)] == ["a", "b"]


def test_hybrid_reranker_hook_changes_order():
    class ReverseReranker:
        def rerank(self, query, candidates, corpus):
            return list(reversed(candidates))

    r1 = FakeRetriever([Hit("a", 3), Hit("b", 2), Hit("c", 1)])
    hybrid = HybridRetriever(
        {"r1": r1}, method="rrf", candidate_k=10, reranker=ReverseReranker()
    )
    assert [h.doc_id for h in hybrid.search("q", k=3)] == ["c", "b", "a"]


def test_unknown_method_raises():
    hybrid = HybridRetriever({"r1": FakeRetriever([])}, method="bogus")
    with pytest.raises(ValueError):
        hybrid.search("q")


# --- real cross-encoder, skipped when the model is unavailable --------
def test_cross_encoder_reranker_runs_on_toy_corpus():
    st = pytest.importorskip("sentence_transformers")
    from ragsearch.retrieve.rerank import CrossEncoderReranker

    try:
        ce = CrossEncoderReranker(top_n=3)
        corpus = {
            "match": "The mitochondrion is the powerhouse of the cell.",
            "off1": "Quarterly revenue guidance was revised upward.",
            "off2": "The Treaty of Westphalia was signed in 1648.",
        }
        # feed it in a deliberately wrong order
        cand = [Hit("off1", 0.9), Hit("off2", 0.8), Hit("match", 0.1)]
        out = ce.rerank("what part of the cell produces energy", cand, corpus)
    except Exception as exc:  # offline / no cached model
        pytest.skip(f"cross-encoder unavailable: {exc}")
    assert out[0].doc_id == "match"
    assert {h.doc_id for h in out} == set(corpus)
