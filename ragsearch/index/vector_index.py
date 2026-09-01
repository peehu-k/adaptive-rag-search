"""A small vector index over document embeddings.

Backed by FAISS. The default is an exact inner-product flat index, which on a
few-thousand-document benchmark is fast and removes approximate-search noise
from the retrieval numbers; an HNSW graph is available for larger runs. The
string document ids are stored beside the FAISS index and returned from
``search``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorIndex:
    def __init__(self, dim: int, metric: str = "ip", index_type: str = "flat",
                 hnsw_m: int = 32):
        import faiss

        self.dim = dim
        self.metric = metric
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.doc_ids: list[str] = []

        if index_type == "flat":
            self._index = (
                faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
            )
        elif index_type == "hnsw":
            space = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
            self._index = faiss.IndexHNSWFlat(dim, hnsw_m, space)
        else:
            raise ValueError(f"unknown index_type {index_type!r}")

    def __len__(self) -> int:
        return len(self.doc_ids)

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        if len(doc_ids) != vectors.shape[0]:
            raise ValueError("doc_ids and vectors length mismatch")
        self._index.add(vectors)
        self.doc_ids.extend(doc_ids)

    def search(self, queries: np.ndarray, k: int = 10):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        k = min(k, len(self.doc_ids)) or 1
        scores, idx = self._index.search(queries, k)
        ids = [[self.doc_ids[i] if i != -1 else None for i in row] for row in idx]
        return scores, ids

    # --- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "dim": self.dim,
                    "metric": self.metric,
                    "index_type": self.index_type,
                    "hnsw_m": self.hnsw_m,
                    "doc_ids": self.doc_ids,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        import faiss

        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj.dim = meta["dim"]
        obj.metric = meta["metric"]
        obj.index_type = meta["index_type"]
        obj.hnsw_m = meta.get("hnsw_m", 32)
        obj.doc_ids = list(meta["doc_ids"])
        obj._index = faiss.read_index(str(path.with_suffix(".faiss")))
        return obj
