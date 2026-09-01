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


@dataclass(frozen=True)
class CrossEncoderReranker:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 100
    max_length: int = 512
    batch_size: int = 64

    def rerank(self, query, candidates, corpus):
        head = candidates[: self.top_n]
        tail = candidates[self.top_n :]
        if not head:
            return list(candidates)
        model = _load_cross_encoder(self.model_name, self.max_length)
        pairs = [[query, corpus.get(h.doc_id, "")] for h in head]
        scores = model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        rescored = [Hit(h.doc_id, float(s)) for h, s in zip(head, scores)]
        rescored.sort(key=lambda x: x.score, reverse=True)
        # anything past top_n keeps its fused position, below the reranked head
        return rescored + list(tail)
