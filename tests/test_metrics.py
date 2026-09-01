"""Ranking-metric unit checks against hand-computed values."""

import math

import pytest

from ragsearch.eval.metrics import (
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
    score_ranking,
)

# doc2 and doc4 are relevant
QRELS = {"doc2": 1, "doc4": 1}
RANKING = ["doc1", "doc2", "doc3", "doc4", "doc5"]


def test_recall_and_precision():
    assert recall_at_k(RANKING, QRELS, 2) == pytest.approx(0.5)
    assert recall_at_k(RANKING, QRELS, 4) == pytest.approx(1.0)
    assert precision_at_k(RANKING, QRELS, 2) == pytest.approx(0.5)
    assert precision_at_k(RANKING, QRELS, 4) == pytest.approx(0.5)


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(RANKING, QRELS) == pytest.approx(0.5)  # first hit at rank 2
    assert reciprocal_rank(["a", "b"], QRELS) == 0.0


def test_average_precision_hand_value():
    # hits at ranks 2 and 4 -> (1/2 + 2/4) / 2 = 0.5
    assert average_precision(RANKING, QRELS) == pytest.approx(0.5)


def test_dcg_matches_definition():
    # gains [1, 0, 1] -> 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5
    assert dcg_at_k([1, 0, 1], 3) == pytest.approx(1.5)


def test_ndcg_perfect_ranking_is_one():
    perfect = ["doc2", "doc4", "doc1"]
    assert ndcg_at_k(perfect, QRELS, 10) == pytest.approx(1.0)


def test_ndcg_hand_value():
    # ranking hits at positions 2 and 4:
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKING, QRELS, 10) == pytest.approx(dcg / idcg)


def test_no_relevant_docs_scores_zero_not_error():
    assert recall_at_k(RANKING, {}, 10) == 0.0
    assert ndcg_at_k(RANKING, {}, 10) == 0.0
    assert average_precision(RANKING, {}, 10) == 0.0


def test_score_ranking_flat_keys():
    out = score_ranking(RANKING, QRELS, ks=(1, 10))
    assert set(out) == {
        f"{m}@{k}" for m in ("recall", "precision", "hit", "mrr", "map", "ndcg")
        for k in (1, 10)
    }
    assert out["hit@1"] == 0.0 and out["hit@10"] == 1.0
