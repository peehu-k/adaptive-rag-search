# Results

SciFact (BEIR). Retrieval over a 5,183-abstract corpus. Splits: 566 train (diagnosis), 243 optimize (search fitness), 300 test (held out, evaluated once).

## Held-out test split

| config | ndcg@10 | recall@10 | recall@100 | mrr@10 | map@100 |
|---|---|---|---|---|---|
| bm25_only | 0.6669 | 0.7856 | 0.8759 | 0.6355 | 0.6282 |
| dense_only | 0.6484 | 0.7883 | 0.9250 | 0.6068 | 0.6055 |
| hybrid_rrf | 0.6927 | 0.8319 | 0.9583 | 0.6533 | 0.6490 |
| hybrid_weighted (baseline) | 0.7089 | 0.8393 | 0.9550 | 0.6717 | 0.6680 |
| hybrid_weighted + reranker@10 | 0.7056 | 0.8393 | 0.9550 | 0.6719 | 0.6624 |
| optimized (search) | 0.7089 | 0.8393 | 0.9550 | 0.6717 | 0.6680 |

## Search

- fitness metric: `recall@10` on the optimize split
- configs evaluated: 12
- optimize fitness: 0.8368 (baseline) -> 0.8368 (no significant improvement)
- accepted mutations: none

Bandit arms (mutation family, pulls, mean reward = fraction that improved fitness):

| family | pulls | mean reward |
|---|---|---|
| analyzer_enable_stem | 1 | 1.0 |
| analyzer_keep_stopwords | 1 | 0.0 |
| analyzer_min_token_len | 1 | 0.0 |
| bm25_b | 2 | 0.5 |
| bm25_k1 | 2 | 0.0 |
| candidate_k | 1 | 0.0 |
| fusion_method_rrf | 1 | 0.0 |
| upweight_dense | 2 | 0.0 |

## Significance (paired, held-out test)

```
optimized (search) vs hybrid_weighted (baseline)  ndcg@10: 0.7089 -> 0.7089  Δ+0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
optimized (search) vs hybrid_weighted (baseline)  recall@10: 0.8393 -> 0.8393  Δ+0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
optimized (search) vs hybrid_weighted (baseline)  recall@100: 0.9550 -> 0.9550  Δ+0.0000  p=1.0000  95% CI [+0.0000, +0.0000]
hybrid_weighted (baseline) vs bm25_only  ndcg@10: 0.6669 -> 0.7089  Δ+0.0420  p=0.0003  95% CI [+0.0198, +0.0651]
hybrid_weighted (baseline) vs bm25_only  recall@100: 0.8759 -> 0.9550  Δ+0.0791  p=0.0001  95% CI [+0.0516, +0.1100]
hybrid_weighted (baseline) vs dense_only  ndcg@10: 0.6484 -> 0.7089  Δ+0.0605  p=0.0001  95% CI [+0.0342, +0.0873]
hybrid_weighted + reranker@10 vs hybrid_weighted (baseline)  ndcg@10: 0.7089 -> 0.7056  Δ-0.0033  p=0.7561  95% CI [-0.0236, +0.0169]
```

## Reading

- The search loop evaluated every proposed mutation and **accepted none**: no
  diagnosed-and-targeted change beat the equal-weight hybrid on the fitness
  metric at p < 0.05. The `optimized (search)` row therefore equals the
  baseline. This is the loop's significance gate refusing to lock in noise,
  not a failure to run. The same holds when the loop is run with `ndcg@10` as
  the fitness metric instead of `recall@10`.
- The significant, reproducible gains are structural: the from-scratch hybrid
  over BM25-only (+0.042 nDCG@10, p = 0.0003; +0.079 recall@100, p = 0.0001)
  and over dense-only (+0.061 nDCG@10, p = 0.0001). All three 95% CIs exclude
  zero.
- The shallow cross-encoder rerank (top-10) does not move nDCG@10 (Δ-0.003,
  p = 0.76); it is capped at depth 10 because this run was CPU-only.

Full per-config lineage: `results/search_lineage.jsonl`. Optimized config: `results/best_config.json`.
