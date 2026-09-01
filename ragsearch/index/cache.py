"""On-disk cache for document embeddings.

Encoding the whole corpus is by far the slowest thing in the pipeline, and it
only depends on the corpus and the embedder config -- not on any retrieval or
search-loop choice. Cache it under ``data/scifact/cache/`` keyed by both.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ragsearch.benchmark import SCIFACT_DIR
from ragsearch.index.embedder import DenseEmbedder

CACHE_DIR = SCIFACT_DIR / "cache"


def _key(embedder: DenseEmbedder, doc_ids: list[str]) -> str:
    parts = [
        embedder.model_name,
        f"norm={embedder.normalize}",
        f"msl={embedder.max_seq_length}",
    ]
    if embedder.doc_prefix:
        parts.append(f"dpref={embedder.doc_prefix}")
    parts += [
        f"n={len(doc_ids)}",
        doc_ids[0] if doc_ids else "",
        doc_ids[-1] if doc_ids else "",
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def cached_doc_embeddings(
    embedder: DenseEmbedder,
    doc_ids: list[str],
    texts: list[str],
    *,
    verbose: bool = True,
) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"emb_{_key(embedder, doc_ids)}.npy"
    if path.exists():
        if verbose:
            print(f"loaded cached embeddings {path.name}")
        return np.load(path)
    if verbose:
        print(f"encoding {len(texts)} documents (no cache hit) ...")
    vecs = embedder.encode_documents(texts)
    np.save(path, vecs)
    if verbose:
        print(f"  wrote {path.name}")
    return vecs
