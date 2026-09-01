# Adaptive RAG Search

A retrieval pipeline is a search space. This project treats it as one: it
builds a lexical + dense + fusion + rerank pipeline for a fixed benchmark,
**diagnoses why individual queries fail**, proposes **targeted** config
mutations from those diagnoses, and runs a search loop that keeps only the
mutations that beat the incumbent on a held-out fitness split — with a full
experiment lineage. The final numbers are reported on a test split the
optimizer never touches.

Benchmark: **SciFact** (BEIR) — 5,183 scientific abstracts, 1,109 claim
queries, graded relevance judgments.

## Headline

Held-out test split (300 queries), nDCG@10 / recall@100, paired randomization test:

| pipeline | nDCG@10 | recall@100 |
|---|---|---|
| BM25 (from scratch) | 0.667 | 0.876 |
| dense (MiniLM) | 0.648 | 0.925 |
| **hybrid (weighted fusion)** | **0.709** | **0.955** |

Hybrid vs BM25-only: **ΔnDCG@10 +0.042 (p = 0.0003)**, Δrecall@100 +0.079
(p = 0.0001) — both significant, CIs exclude zero.

The diagnosis-driven search loop explored 12 targeted configs per run (on two
different fitness metrics) and **found no change that beats the equal-weight
hybrid at p < 0.05**. That is the honest result: the hybrid is already a
strong local optimum, and the loop's significance gate correctly refuses to
lock in noise. The real, significant quality comes from the pipeline
structure, quantified in the ablation above and in `results/RESULTS.md`.

## Layout

```
ragsearch/
  benchmark.py        SciFact loader + the train / optimize / test split
  index/
    tokenizer.py      configurable analyzer (lowercase, stopwords, stemming)
    inverted.py        inverted index built from scratch
    embedder.py        sentence-transformer embeddings
    vector_index.py    FAISS wrapper (exact flat / HNSW)
    cache.py           on-disk embedding cache
  retrieve/
    bm25.py            Okapi BM25 over the inverted index (from scratch)
    dense.py           dense retriever
    fusion.py          RRF + weighted fusion, HybridRetriever, search trace
    rerank.py          reranker hook + cross-encoder implementation
  eval/
    metrics.py         nDCG@k, recall@k, precision@k, MRR, MAP
    qa.py              SQuAD-style EM / F1, lexical answerer
    significance.py    paired randomization + bootstrap tests
    harness.py         run / score / compare
  diagnose/
    failures.py        label a miss: recall / fusion / reranker / generation
    cluster.py         aggregate diagnoses into a FailureReport
  search/
    config.py          PipelineConfig (the point in the search space)
    build.py           PipelineConfig -> live retriever
    mutate.py          diagnosis -> targeted mutations
    bandit.py          UCB1 over mutation families
    loop.py            the search loop + experiment lineage log
```

## Setup

```bash
python -m pip install -e ".[dense]"
python scripts/get_data.py        # downloads SciFact into data/ (git-ignored)
python scripts/build_splits.py    # materializes the split
```

`data/` and `runs/` are git-ignored; committed experiment output lives in
`results/`.

## Reproduce

```bash
python scripts/run_bm25.py --split optimize        # BM25 sanity numbers
python scripts/run_dense.py --split optimize        # dense sanity numbers
python scripts/run_hybrid.py --split optimize       # fusion comparison
python scripts/evaluate.py --split optimize         # metrics + significance
python scripts/diagnose.py --split train            # failure clusters
python scripts/propose.py --split train             # proposed mutations
python scripts/search.py --max-rounds 3             # the search loop
python scripts/final_run.py                         # search + held-out test + report
```

`scripts/final_run.py` writes `results/RESULTS.md`, `results/test_metrics.json`,
`results/best_config.json`, and `results/search_lineage.jsonl`.

## Method notes

- **BM25** is a real inverted index (term → `[(doc, tf)]`, doc lengths,
  collection stats) with Okapi scoring and the non-negative idf variant — not
  a library.
- **Splits.** `test` is BEIR's official test qrels, never subdivided and never
  read by diagnosis or the search loop (the loop raises if a test id appears
  in its query sets). `train`/`optimize` is a deterministic 70/30 cut of
  BEIR's train qrels.
- **Diagnosis** classifies every miss by the stage responsible: `recall_miss`
  (gold never entered any candidate pool), `fusion_miss` (pooled but ranked
  below k), `reranker_miss` (in the fused top-k, reranker dropped it),
  `generation_miss` (in context, answer still wrong).
- **Mutations are targeted.** A `fusion_miss` cluster with its golds mostly
  recoverable via the dense retriever produces "raise the dense fusion
  weight", etc. Each mutation records the rationale and the counts it responds
  to.
- **Acceptance.** A mutation is kept only if it beats the incumbent on the
  fitness metric *and* a paired randomization test gives p < 0.05.
- **Significance.** Config comparisons use a paired randomization test (10k
  iterations) and a bootstrap 95% CI.

## Tests

```bash
python -m pytest -q
```

Tests that need the benchmark or the embedding model skip cleanly when those
are absent.
