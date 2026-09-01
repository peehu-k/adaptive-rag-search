"""Instantiate a live retriever from a :class:`PipelineConfig`."""

from __future__ import annotations

from typing import Mapping

from ragsearch.index.inverted import InvertedIndex
from ragsearch.index.tokenizer import Analyzer
from ragsearch.retrieve.bm25 import BM25Retriever
from ragsearch.retrieve.fusion import HybridRetriever
from ragsearch.retrieve.rerank import CrossEncoderReranker, IdentityReranker
from ragsearch.search.config import PipelineConfig


def _build_bm25(config: PipelineConfig, corpus_map: Mapping[str, str]) -> BM25Retriever:
    analyzer = Analyzer(
        lowercase=config.analyzer.lowercase,
        remove_stopwords=config.analyzer.remove_stopwords,
        stem=config.analyzer.stem,
        min_token_len=config.analyzer.min_token_len,
    )
    index = InvertedIndex.build(dict(corpus_map), analyzer=analyzer)
    return BM25Retriever(index, k1=config.bm25.k1, b=config.bm25.b)


def _build_dense(config: PipelineConfig, corpus_map, doc_embeddings, query_cache):
    from ragsearch.index.embedder import DenseEmbedder
    from ragsearch.retrieve.dense import DenseRetriever

    embedder = DenseEmbedder(
        model_name=config.dense.model_name,
        normalize=config.dense.normalize,
        max_seq_length=config.dense.max_seq_length,
    )
    doc_ids = list(corpus_map)
    if doc_embeddings is None:
        from ragsearch.index.cache import cached_doc_embeddings

        doc_embeddings = cached_doc_embeddings(
            embedder, doc_ids, [corpus_map[d] for d in doc_ids], verbose=False
        )
    return DenseRetriever.build(
        {d: "" for d in doc_ids},
        embedder=embedder,
        precomputed=doc_embeddings,
        query_cache=query_cache,
    )


def build_pipeline(
    config: PipelineConfig,
    corpus_map: Mapping[str, str],
    *,
    doc_embeddings=None,
    query_cache=None,
):
    """Return an object with ``.search(query, k)`` matching ``config``.

    ``doc_embeddings`` (an ``(N, dim)`` array aligned with ``corpus_map``'s key
    order) is used when dense retrieval is enabled; if omitted it falls back to
    the on-disk embedding cache. ``query_cache`` is an optional shared
    ``{query: vector}`` dict so repeated rebuilds don't re-encode queries.
    """
    retrievers: dict = {}
    if config.use_bm25:
        retrievers["bm25"] = _build_bm25(config, corpus_map)
    if config.use_dense:
        retrievers["dense"] = _build_dense(
            config, corpus_map, doc_embeddings, query_cache
        )

    if len(retrievers) == 1:
        return next(iter(retrievers.values()))

    reranker = (
        CrossEncoderReranker(
            model_name=config.rerank.model_name, top_n=config.rerank.top_n
        )
        if config.rerank.enabled
        else IdentityReranker()
    )
    return HybridRetriever(
        retrievers=retrievers,
        method=config.fusion.method,
        weights={"bm25": config.fusion.weight_bm25, "dense": config.fusion.weight_dense},
        rrf_k=config.fusion.rrf_k,
        candidate_k=config.fusion.candidate_k,
        corpus=dict(corpus_map),
        reranker=reranker,
    )
