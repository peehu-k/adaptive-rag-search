"""Hybrid fusion of several ranked lists, plus the reranker hook.

Two fusion methods:

* ``rrf`` -- Reciprocal Rank Fusion. Each list contributes
  ``weight / (rrf_k + rank)`` to every doc it ranks. Rank-based, so it does
  not care that BM25 and cosine scores live on different scales.
* ``weighted`` -- min-max normalize each list's scores to [0, 1], then take a
  weighted sum. The weights are a natural target for the search loop to tune.

``HybridRetriever`` ties named base retrievers, a fusion method, and an
optional reranker into one object that still satisfies the ``Retriever``
protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ragsearch.retrieve.base import Hit, Retriever
from ragsearch.retrieve.rerank import IdentityReranker, Reranker


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    rrf_k: float = 60.0,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match number of rankings")
    acc: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, hit in enumerate(ranking, start=1):
            acc[hit.doc_id] = acc.get(hit.doc_id, 0.0) + w / (rrf_k + rank)
    return [Hit(d, s) for d, s in sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))]


def weighted_score_fusion(
    rankings: Sequence[Sequence[Hit]],
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match number of rankings")
    acc: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        ids = [h.doc_id for h in ranking]
        norm = _minmax([h.score for h in ranking])
        for doc_id, s in zip(ids, norm):
            acc[doc_id] = acc.get(doc_id, 0.0) + w * s
    return [Hit(d, s) for d, s in sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))]


@dataclass
class HybridRetriever:
    retrievers: dict[str, Retriever]
    method: str = "rrf"  # "rrf" | "weighted"
    weights: dict[str, float] = field(default_factory=dict)
    rrf_k: float = 60.0
    candidate_k: int = 200
    corpus: Mapping[str, str] = field(default_factory=dict)
    reranker: Reranker = field(default_factory=IdentityReranker)

    def _ordered_names(self) -> list[str]:
        return list(self.retrievers)

    def _weight_vector(self) -> list[float]:
        return [self.weights.get(name, 1.0) for name in self._ordered_names()]

    def fuse(self, rankings: Sequence[Sequence[Hit]]) -> list[Hit]:
        w = self._weight_vector()
        if self.method == "rrf":
            return reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k, weights=w)
        if self.method == "weighted":
            return weighted_score_fusion(rankings, weights=w)
        raise ValueError(f"unknown fusion method {self.method!r}")

    def search(self, query: str, k: int = 10) -> list[Hit]:
        rankings = [
            self.retrievers[name].search(query, k=self.candidate_k)
            for name in self._ordered_names()
        ]
        fused = self.fuse(rankings)
        reranked = self.reranker.rerank(query, fused, self.corpus)
        return reranked[:k]
