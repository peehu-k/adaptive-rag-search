"""PipelineConfig, the config->retriever builder, and the mutation proposer."""

import numpy as np
import pytest

from ragsearch import benchmark
from ragsearch.diagnose.cluster import FailureReport
from ragsearch.diagnose.failures import FUSION_MISS, RECALL_MISS, diagnose_query
from ragsearch.search.build import build_pipeline
from ragsearch.search.config import DEFAULT_CONFIG, PipelineConfig
from ragsearch.search.mutate import MutationProposer

TOY = {
    "d1": "quantum error correction protects fragile qubits",
    "d2": "the cat sat quietly on the warm windowsill",
    "d3": "photosynthesis converts light energy into sugars",
    "d4": "superconducting qubits require millikelvin temperatures",
}


# --- PipelineConfig ---------------------------------------------------
def test_config_roundtrip_and_fingerprint_stable():
    cfg = DEFAULT_CONFIG
    again = PipelineConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert again.fingerprint() == cfg.fingerprint()


def test_with_section_is_a_targeted_copy():
    cfg = DEFAULT_CONFIG.with_section("fusion", weight_dense=2.0)
    assert cfg.fusion.weight_dense == 2.0
    assert cfg.fusion.weight_bm25 == DEFAULT_CONFIG.fusion.weight_bm25
    assert cfg.analyzer == DEFAULT_CONFIG.analyzer
    assert cfg.fingerprint() != DEFAULT_CONFIG.fingerprint()


def test_config_rejects_no_retriever():
    with pytest.raises(ValueError):
        PipelineConfig(use_bm25=False, use_dense=False)


# --- build_pipeline -------------------------------------------------
def test_build_bm25_only_pipeline_searches():
    cfg = PipelineConfig(use_bm25=True, use_dense=False)
    retr = build_pipeline(cfg, TOY)
    hits = retr.search("qubits", k=3)
    assert hits and hits[0].doc_id in {"d1", "d4"}


def test_build_hybrid_pipeline_with_injected_embeddings():
    # doc vectors must share the query encoder's dimension (MiniLM -> 384);
    # values are random, we only check the pipeline wires up and returns hits
    from ragsearch.index.embedder import DenseEmbedder

    dim = DenseEmbedder().dim
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(TOY), dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    cfg = PipelineConfig()  # hybrid
    retr = build_pipeline(cfg, TOY, doc_embeddings=emb)
    from ragsearch.retrieve.fusion import HybridRetriever

    assert isinstance(retr, HybridRetriever)
    assert {"bm25", "dense"} == set(retr.retrievers)
    assert len(retr.search("qubits", k=2)) == 2


# --- MutationProposer ----------------------------------------------
def _fusion_miss_report(recover="dense", n=30):
    diags = []
    for i in range(n):
        pool = ["x"] * 40 + ["g"] + ["y"] * 40
        per = {recover: pool} if recover else {"bm25": pool}
        diags.append(diagnose_query(
            f"q{i}", ["g"], per_retriever=per, fused=pool, final=pool,
            k=10, pool_depth=200,
        ))
    diags += [diagnose_query(f"s{i}", ["g"], per_retriever={"bm25": ["g"]},
                             fused=["g"], final=["g"], k=10) for i in range(20)]
    return FailureReport(diagnoses=diags, k=10)


def test_no_failures_means_no_mutations():
    ok = [diagnose_query(f"s{i}", ["g"], per_retriever={"bm25": ["g"]},
                         fused=["g"], final=["g"], k=10) for i in range(10)]
    muts = MutationProposer().propose(DEFAULT_CONFIG, FailureReport(ok, k=10))
    assert muts == []


def test_fusion_miss_proposes_dense_upweight():
    report = _fusion_miss_report(recover="dense")
    muts = MutationProposer().propose(DEFAULT_CONFIG, report)
    assert muts
    assert all(m.config.fingerprint() != DEFAULT_CONFIG.fingerprint() for m in muts)
    assert all(m.rationale and m.targets for m in muts)
    # at least one lifts the dense weight above bm25
    assert any(
        m.config.fusion.weight_dense > m.config.fusion.weight_bm25
        for m in muts
        if "upweight" in m.name
    )
    # all mutation configs are distinct
    assert len({m.fingerprint for m in muts}) == len(muts)


def test_fusion_miss_recoverable_by_bm25_upweights_bm25():
    report = _fusion_miss_report(recover="bm25")
    muts = MutationProposer().propose(DEFAULT_CONFIG, report)
    assert any(
        m.config.fusion.weight_bm25 > m.config.fusion.weight_dense
        for m in muts if "upweight" in m.name
    )


def test_fusion_miss_also_proposes_reranker_and_param_sweeps():
    report = _fusion_miss_report(recover="dense")
    names = {m.name for m in MutationProposer().propose(DEFAULT_CONFIG, report)}
    assert "rerank_enable" in names
    assert any(n.startswith("bm25_k1_") for n in names)
    assert any(n.startswith("bm25_b_") for n in names)
    # the rerank_enable mutation actually flips the config flag
    mut = next(
        m for m in MutationProposer().propose(DEFAULT_CONFIG, report)
        if m.name == "rerank_enable"
    )
    assert mut.config.rerank.enabled and mut.config.rerank.top_n == 50


def test_mutation_family_strips_numeric_suffix():
    report = _fusion_miss_report(recover="dense")
    fams = {m.family for m in MutationProposer().propose(DEFAULT_CONFIG, report)}
    assert "upweight_dense" in fams  # from upweight_dense_x1.50 / _x2.00
    assert "bm25_k1" in fams


def test_recall_miss_proposes_analyzer_change():
    diags = []
    for i in range(20):
        diags.append(diagnose_query(
            f"q{i}", ["g"], per_retriever={"bm25": ["a", "b"], "dense": ["c"]},
            fused=["a", "b", "c"], final=["a", "b", "c"], k=10, pool_depth=200,
        ))
    muts = MutationProposer().propose(DEFAULT_CONFIG, FailureReport(diags, k=10))
    assert any(m.name.startswith("analyzer_") for m in muts)
    assert any("analyzer" in k for m in muts for k in m.param_delta)


@pytest.mark.skipif(
    not (benchmark.SCIFACT_DIR / "corpus.jsonl").exists(),
    reason="run scripts/get_data.py to fetch the SciFact benchmark",
)
def test_proposed_mutations_build_on_real_corpus():
    from ragsearch.diagnose.cluster import diagnose_hybrid
    from ragsearch.index.cache import cached_doc_embeddings
    from ragsearch.index.embedder import DenseEmbedder

    bench = benchmark.load_benchmark()
    doc_ids = list(bench.corpus)
    texts = [benchmark.doc_text(bench.corpus[d]) for d in doc_ids]
    corpus_map = dict(zip(doc_ids, texts))
    try:
        vecs = cached_doc_embeddings(DenseEmbedder(), doc_ids, texts, verbose=False)
    except Exception as exc:
        pytest.skip(f"embeddings unavailable: {exc}")

    base = build_pipeline(DEFAULT_CONFIG, corpus_map, doc_embeddings=vecs)
    slice_q = {q: bench.queries[q] for q in bench.splits["train"][:60]}
    report = diagnose_hybrid(base, slice_q, bench.qrels, k=10)
    muts = MutationProposer().propose(DEFAULT_CONFIG, report)
    assert muts, "expected at least one mutation from a real diagnosis"

    # each proposed config actually instantiates and searches
    m = muts[0]
    retr = build_pipeline(m.config, corpus_map, doc_embeddings=vecs)
    hits = retr.search(next(iter(slice_q.values())), k=10)
    assert len(hits) == 10
