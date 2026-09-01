"""Dense text embeddings from a real sentence-transformer model.

Wraps a :class:`sentence_transformers.SentenceTransformer` in a small, frozen
config so a pipeline can carry "which encoder, normalized or not" around the
same way it carries the lexical analyzer. Encoding runs in eval mode with no
gradient and a fixed model, so it is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _load_model(name: str, max_seq_length: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    if max_seq_length:
        model.max_seq_length = max_seq_length
    model.eval()
    return model


@dataclass(frozen=True)
class DenseEmbedder:
    model_name: str = DEFAULT_MODEL
    normalize: bool = True
    batch_size: int = 64
    max_seq_length: int = 256
    query_prefix: str = ""
    doc_prefix: str = ""

    @property
    def dim(self) -> int:
        return _load_model(self.model_name, self.max_seq_length).get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        model = _load_model(self.model_name, self.max_seq_length)
        if prefix:
            texts = [prefix + t for t in texts]
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.doc_prefix)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)
