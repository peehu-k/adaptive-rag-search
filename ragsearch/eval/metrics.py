"""Ranking metrics.

Everything here works on a ranked list of document ids plus a relevance map
``{doc_id: gain}``. SciFact judgments are binary, but the code accepts graded
gains so it carries over to other benchmarks.

nDCG follows the trec_eval / BEIR convention::

    DCG@k  = sum_{i=1..k} gain_i / log2(i + 1)
    IDCG@k = DCG@k of the ideal ordering
    nDCG@k = DCG@k / IDCG@k          (0 when no document is relevant)
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _relevant_set(qrels: Mapping[str, float]) -> set[str]:
    return {d for d, g in qrels.items() if g > 0}


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    rel = _relevant_set(qrels)
    if not rel:
        return 0.0
    return len(set(ranking[:k]) & rel) / len(rel)


def precision_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = _relevant_set(qrels)
    return len(set(ranking[:k]) & rel) / k


def hit_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    rel = _relevant_set(qrels)
    return 1.0 if set(ranking[:k]) & rel else 0.0


def reciprocal_rank(ranking: Sequence[str], qrels: Mapping[str, float], k: int | None = None) -> float:
    rel = _relevant_set(qrels)
    cutoff = len(ranking) if k is None else k
    for i, doc in enumerate(ranking[:cutoff], start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranking: Sequence[str], qrels: Mapping[str, float], k: int | None = None) -> float:
    rel = _relevant_set(qrels)
    if not rel:
        return 0.0
    cutoff = len(ranking) if k is None else k
    hits = 0
    total = 0.0
    for i, doc in enumerate(ranking[:cutoff], start=1):
        if doc in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def dcg_at_k(gains: Iterable[float], k: int) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(list(gains)[:k], start=1))


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    gains = [max(qrels.get(doc, 0.0), 0.0) for doc in ranking[:k]]
    ideal = sorted((max(g, 0.0) for g in qrels.values()), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


# names -> (fn, needs_k)
_METRICS = {
    "recall": (recall_at_k, True),
    "precision": (precision_at_k, True),
    "hit": (hit_at_k, True),
    "mrr": (reciprocal_rank, True),
    "map": (average_precision, True),
    "ndcg": (ndcg_at_k, True),
}


def score_ranking(
    ranking: Sequence[str],
    qrels: Mapping[str, float],
    ks: Sequence[int] = (1, 5, 10, 100),
) -> dict[str, float]:
    """All metrics at all cutoffs for a single query, flat-keyed like ``ndcg@10``."""
    out: dict[str, float] = {}
    for name, (fn, _needs_k) in _METRICS.items():
        for k in ks:
            out[f"{name}@{k}"] = fn(ranking, qrels, k)
    return out
