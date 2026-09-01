"""Reranker hook.

The hybrid retriever calls ``reranker.rerank(query, candidates, corpus)`` on
the fused candidate list and uses whatever order it returns. The default is a
no-op passthrough so the hook is always present; a cross-encoder
implementation is provided for when the search loop wants to switch it on.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Protocol, runtime_checkable

from ragsearch.retrieve.base import Hit


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[Hit], corpus: Mapping[str, str]
    ) -> list[Hit]:
        ...


@dataclass(frozen=True)
class IdentityReranker:
    """Keeps the fused order untouched."""

    def rerank(self, query, candidates, corpus):
        return list(candidates)


@lru_cache(maxsize=2)
def _load_cross_encoder(name: str, max_length: int):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, max_length=max_length)


# (model_name, query, doc_id) -> score. Process-wide so the search loop, which
# reranks overlapping fused heads many times, pays the model cost once per pair.
_SCORE_CACHE: dict[tuple[str, str, str], float] = {}


@dataclass(frozen=True)
class CrossEncoderReranker:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 100
    max_length: int = 512
    batch_size: int = 64
    use_cache: bool = True

    def rerank(self, query, candidates, corpus):
        head = candidates[: self.top_n]
        tail = candidates[self.top_n :]
        if not head:
            return list(candidates)

        scores: dict[str, float] = {}
        to_score = []
        for h in head:
            key = (self.model_name, query, h.doc_id)
            if self.use_cache and key in _SCORE_CACHE:
                scores[h.doc_id] = _SCORE_CACHE[key]
            else:
                to_score.append(h)

        if to_score:
            model = _load_cross_encoder(self.model_name, self.max_length)
            pairs = [[query, corpus.get(h.doc_id, "")] for h in to_score]
            fresh = model.predict(
                pairs, batch_size=self.batch_size, show_progress_bar=False
            )
            for h, s in zip(to_score, fresh):
                scores[h.doc_id] = float(s)
                if self.use_cache:
                    _SCORE_CACHE[(self.model_name, query, h.doc_id)] = float(s)

        rescored = [Hit(h.doc_id, scores[h.doc_id]) for h in head]
        rescored.sort(key=lambda x: x.score, reverse=True)
        # anything past top_n keeps its fused position, below the reranked head
        return rescored + list(tail)
