"""Checks on the vector index and the dense retriever.

The vector-index tests only need FAISS. The retriever tests also need the
sentence-transformer model, so they are skipped where it (or the network to
fetch it) is unavailable.
"""

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from ragsearch.index.vector_index import VectorIndex  # noqa: E402


def _unit(rows):
    m = np.asarray(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_vector_index_returns_nearest_by_inner_product():
    vi = VectorIndex(dim=3, metric="ip", index_type="flat")
    vecs = _unit([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
    vi.add(["x", "y", "z"], vecs)
    scores, ids = vi.search(_unit([[1, 0, 0]]), k=3)
    assert ids[0][0] == "x"
    assert ids[0][1] == "z"  # closest to the x axis after x itself
    assert scores[0][0] >= scores[0][1] >= scores[0][2]


def test_vector_index_dim_mismatch_raises():
    vi = VectorIndex(dim=4)
    with pytest.raises(ValueError):
        vi.add(["a"], np.zeros((1, 3), dtype=np.float32))


def test_vector_index_k_larger_than_corpus():
    vi = VectorIndex(dim=2)
    vi.add(["a", "b"], _unit([[1, 0], [0, 1]]))
    scores, ids = vi.search(_unit([[1, 1]]), k=10)
    assert len(ids[0]) == 2


def test_vector_index_save_load_roundtrip(tmp_path):
    vi = VectorIndex(dim=3)
    vecs = _unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vi.add(["a", "b", "c"], vecs)
    vi.save(tmp_path / "vi")
    back = VectorIndex.load(tmp_path / "vi")
    assert back.doc_ids == vi.doc_ids
    assert back.dim == vi.dim
    s1, i1 = vi.search(vecs[:1], k=3)
    s2, i2 = back.search(vecs[:1], k=3)
    assert i1 == i2
    np.testing.assert_allclose(s1, s2, rtol=1e-5)


# --- retriever tests (need the model) ----------------------------------
st = pytest.importorskip("sentence_transformers")

from ragsearch.index.embedder import DenseEmbedder  # noqa: E402
from ragsearch.retrieve.dense import DenseRetriever  # noqa: E402


@pytest.fixture(scope="module")
def embedder():
    try:
        emb = DenseEmbedder()
        _ = emb.dim  # forces model load / download
        return emb
    except Exception as exc:  # offline, no cached model, etc.
        pytest.skip(f"embedding model unavailable: {exc}")


def test_dense_retriever_ranks_semantic_match_first(embedder):
    docs = {
        "canine": "The dog barked loudly at the mail carrier.",
        "market": "Equity markets fell sharply amid recession fears.",
        "cooking": "She simmered the tomato sauce for two hours.",
    }
    r = DenseRetriever.build(docs, embedder=embedder)
    hits = r.search("a puppy making noise outside", k=3)
    assert hits[0].doc_id == "canine"
    assert hits[0].score >= hits[-1].score


def test_dense_search_many_matches_single(embedder):
    docs = {"a": "photosynthesis in green plants", "b": "quarterly tax filing"}
    r = DenseRetriever.build(docs, embedder=embedder)
    one = r.search("how plants make energy from light", k=2)
    many = r.search_many(["how plants make energy from light"], k=2)[0]
    assert [h.doc_id for h in one] == [h.doc_id for h in many]
