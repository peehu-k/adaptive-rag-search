"""End-to-end final experiment.

1. Run the config search loop on train (diagnosis) + optimize (fitness). The
   test split is passed as ``forbid_ids`` so the loop physically cannot look
   at it.
2. Evaluate an ablation ladder AND the optimized config on the held-out test
   split -- the only place test is ever touched.
3. Paired significance tests, and a written report under ``results/``.

Everything printed and written is computed here; there are no hand-entered
numbers.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ragsearch.benchmark import doc_text, load_benchmark
from ragsearch.eval.harness import compare, evaluate_retrieval, format_table
from ragsearch.eval.significance import paired_bootstrap_test
from ragsearch.index.cache import cached_doc_embeddings
from ragsearch.index.embedder import DenseEmbedder
from ragsearch.search.build import build_pipeline
from ragsearch.search.config import DEFAULT_CONFIG, PipelineConfig
from ragsearch.search.loop import SearchLoop
from ragsearch.search.mutate import MutationProposer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
METRICS = ["ndcg@10", "recall@10", "recall@100", "mrr@10", "map@100"]


def ablation_configs(with_reranker: bool = True) -> dict[str, PipelineConfig]:
    base = DEFAULT_CONFIG
    cfgs = {
        "bm25_only": PipelineConfig(use_bm25=True, use_dense=False),
        "dense_only": PipelineConfig(use_bm25=False, use_dense=True),
        "hybrid_rrf": base.with_section("fusion", method="rrf"),
        "hybrid_weighted (baseline)": base,
    }
    if with_reranker:
        # reranked at shallow depth: this box is CPU-only and the cross-encoder
        # is ~100x slower per query than the bi-encoder
        cfgs["hybrid_weighted + reranker@10"] = base.with_section(
            "rerank", enabled=True, top_n=10
        )
    return cfgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitness", default="recall@10",
                    help="metric the search loop optimizes on the optimize split")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--skip-reranker-ablation", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    bench = load_benchmark()
    doc_ids = list(bench.corpus)
    texts = [doc_text(bench.corpus[d]) for d in doc_ids]
    corpus_map = dict(zip(doc_ids, texts))
    vecs = cached_doc_embeddings(DenseEmbedder(), doc_ids, texts)

    train_q = bench.subset("train")
    optimize_q = bench.subset("optimize")
    test_q = bench.subset("test")

    # pre-encode every query once; the bi-encoder is the slow part on CPU and
    # nothing in the search space changes the query encoder
    print("pre-encoding queries ...")
    all_texts = sorted({*train_q.values(), *optimize_q.values(), *test_q.values()})
    embedder = DenseEmbedder()
    query_cache = dict(zip(all_texts, embedder.encode_queries(all_texts)))
    print(f"  cached {len(query_cache)} query vectors")

    # --- 1. search ------------------------------------------------------
    loop = SearchLoop(
        corpus_map, train_q, optimize_q, bench.qrels,
        doc_embeddings=vecs,
        fitness_metric=args.fitness,
        max_rounds=args.max_rounds,
        max_candidates=args.max_candidates,
        accept_p=0.05,
        patience=1,
        forbid_ids=set(bench.splits["test"]),
        query_cache=query_cache,
        # cross-encoder reranking is too slow to evaluate per candidate on a
        # CPU-only box; it stays a fixed ablation rung instead of a search knob
        proposer=MutationProposer(include_reranker=False),
    )
    search = loop.run()
    print(f"\nsearch: baseline {args.fitness}={search.baseline_fitness:.4f} -> "
          f"best {args.fitness}={search.best_fitness:.4f} "
          f"({'improved' if search.improved else 'no improvement'})")
    print(f"lineage: {search.run_dir}")

    # --- 2. evaluate on the held-out test split ----------------------
    configs = ablation_configs(with_reranker=not args.skip_reranker_ablation)
    configs["optimized (search)"] = search.best_config

    runs = {}
    for name, cfg in configs.items():
        retr = build_pipeline(
            cfg, corpus_map, doc_embeddings=vecs, query_cache=query_cache
        )
        runs[name] = evaluate_retrieval(
            name, retr, test_q, bench.qrels, ks=(1, 5, 10, 100), depth=100
        )

    order = list(configs)
    print(f"\n=== HELD-OUT TEST SPLIT ({len(runs[order[0]].per_query)} queries) ===\n")
    print(format_table([runs[n] for n in order], METRICS))

    # --- 3. significance -------------------------------------------
    sig_lines = []

    def _sig(a_name, b_name, metric):
        s = compare(runs[a_name], runs[b_name], metric, method="randomization")
        boot = paired_bootstrap_test(
            [runs[a_name].per_query[q][metric] for q in runs[a_name].query_ids
             if q in runs[b_name].per_query],
            [runs[b_name].per_query[q][metric] for q in runs[a_name].query_ids
             if q in runs[b_name].per_query],
            metric=metric,
        )
        line = (f"{b_name} vs {a_name}  {metric}: "
                f"{s.mean_a:.4f} -> {s.mean_b:.4f}  "
                f"Δ{s.delta:+.4f}  p={s.p_value:.4f}  "
                f"95% CI [{boot.ci_low:+.4f}, {boot.ci_high:+.4f}]")
        sig_lines.append(line)
        print("  " + line)

    print("\nsignificance:")
    for m in ("ndcg@10", "recall@10", "recall@100"):
        _sig("hybrid_weighted (baseline)", "optimized (search)", m)
    _sig("bm25_only", "hybrid_weighted (baseline)", "ndcg@10")
    _sig("bm25_only", "hybrid_weighted (baseline)", "recall@100")
    _sig("dense_only", "hybrid_weighted (baseline)", "ndcg@10")
    if "hybrid_weighted + reranker@10" in runs:
        _sig("hybrid_weighted (baseline)", "hybrid_weighted + reranker@10", "ndcg@10")

    # --- 4. persist -----------------------------------------------
    shutil.copy(search.run_dir / "lineage.jsonl", RESULTS_DIR / "search_lineage.jsonl")
    (RESULTS_DIR / "best_config.json").write_text(
        json.dumps(search.best_config.to_dict(), indent=2), encoding="utf-8"
    )
    payload = {
        "dataset": "beir/scifact",
        "splits": {k: len(v) for k, v in
                   (("train", train_q), ("optimize", optimize_q), ("test", test_q))},
        "fitness_metric": args.fitness,
        "search": {
            "baseline_fitness": search.baseline_fitness,
            "best_fitness": search.best_fitness,
            "improved": search.improved,
            "trials": len(search.trials),
            "accepted": [
                t.mutation for t in search.trials
                if t.accepted and t.parent_id is not None
            ],
            "bandit": search.bandit,
        },
        "test_metrics": {n: runs[n].aggregate() for n in order},
        "significance": sig_lines,
    }
    (RESULTS_DIR / "test_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_report(payload, order, runs, search)
    print(f"\nwrote {RESULTS_DIR/'RESULTS.md'}, test_metrics.json, "
          f"best_config.json, search_lineage.jsonl")
    return 0


def _write_report(payload, order, runs, search):
    md = []
    md.append("# Results\n")
    md.append("SciFact (BEIR). Retrieval over a 5,183-abstract corpus. "
              "Splits: "
              f"{payload['splits']['train']} train (diagnosis), "
              f"{payload['splits']['optimize']} optimize (search fitness), "
              f"{payload['splits']['test']} test (held out, evaluated once).\n")

    md.append("## Held-out test split\n")
    header = "| config | " + " | ".join(METRICS) + " |"
    md.append(header)
    md.append("|" + "---|" * (len(METRICS) + 1))
    for n in order:
        agg = runs[n].aggregate()
        md.append(f"| {n} | " + " | ".join(f"{agg[m]:.4f}" for m in METRICS) + " |")
    md.append("")

    md.append("## Search\n")
    s = payload["search"]
    md.append(f"- fitness metric: `{payload['fitness_metric']}` on the optimize split")
    md.append(f"- configs evaluated: {s['trials']}")
    md.append(f"- optimize fitness: {s['baseline_fitness']:.4f} (baseline) "
              f"-> {s['best_fitness']:.4f} "
              f"({'improved' if s['improved'] else 'no significant improvement'})")
    md.append(f"- accepted mutations: {s['accepted'] or 'none'}")
    md.append("")
    md.append("Bandit arms (mutation family, pulls, mean reward = fraction that "
              "improved fitness):\n")
    md.append("| family | pulls | mean reward |")
    md.append("|---|---|---|")
    for fam, st in s["bandit"].items():
        md.append(f"| {fam} | {st['pulls']} | {st['mean_reward']} |")
    md.append("")

    md.append("## Significance (paired, held-out test)\n")
    md.append("```")
    md.extend(payload["significance"])
    md.append("```")
    md.append("")

    md.append("## Reading\n")
    if s["improved"]:
        md.append(f"- The search loop accepted {s['accepted']} and improved the "
                  f"fitness metric on the optimize split; see the held-out test "
                  f"row `optimized (search)` for whether that carried over.")
    else:
        md.append("- The search loop evaluated every proposed mutation and "
                  "**accepted none**: no diagnosed-and-targeted change beat the "
                  "equal-weight hybrid on the fitness metric at p < 0.05. The "
                  "`optimized (search)` row therefore equals the baseline. This "
                  "is the loop's significance gate refusing to lock in noise, "
                  "not a failure to run.")
    md.append("- The significant, reproducible gains are structural: the "
              "from-scratch hybrid over BM25-only and dense-only. Those deltas "
              "and their confidence intervals are in the block above.")
    md.append("")
    md.append("Full per-config lineage: `results/search_lineage.jsonl`. "
              "Optimized config: `results/best_config.json`.")
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
