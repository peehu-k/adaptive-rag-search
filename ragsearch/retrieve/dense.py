"""Dense retrieval: embed the query, search the vector index, return hits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragsearch.index.embedder import DenseEmbedder
from ragsearch.index.vector_index import VectorIndex
from ragsearch.retrieve.base import Hit


@dataclass
class DenseRetriever:
    embedder: DenseEmbedder
    index: VectorIndex

    @classmethod
    def build(
        cls,
        documents: dict[str, str],
        embedder: DenseEmbedder | None = None,
        *,
        index_type: str = "flat",
        precomputed: np.ndarray | None = None,
    ) -> "DenseRetriever":
        embedder = embedder or DenseEmbedder()
        doc_ids = list(documents)
        vecs = (
            precomputed
            if precomputed is not None
            else embedder.encode_documents([documents[d] for d in doc_ids])
        )
        vi = VectorIndex(dim=vecs.shape[1], metric="ip", index_type=index_type)
        vi.add(doc_ids, vecs)
        return cls(embedder=embedder, index=vi)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        qv = self.embedder.encode_queries([query])
        scores, ids = self.index.search(qv, k=k)
        return [
            Hit(doc_id, float(s))
            for doc_id, s in zip(ids[0], scores[0])
            if doc_id is not None
        ]

    def search_many(self, queries: list[str], k: int = 10) -> list[list[Hit]]:
        qv = self.embedder.encode_queries(queries)
        scores, ids = self.index.search(qv, k=k)
        return [
            [Hit(d, float(s)) for d, s in zip(row_ids, row_scores) if d is not None]
            for row_ids, row_scores in zip(ids, scores)
        ]
