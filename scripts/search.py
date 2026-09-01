"""Run the config search loop and print the experiment lineage."""

from __future__ import annotations

import argparse

from ragsearch.benchmark import doc_text, load_benchmark
from ragsearch.index.cache import cached_doc_embeddings
from ragsearch.index.embedder import DenseEmbedder
from ragsearch.search.loop import SearchLoop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--accept-p", type=float, default=0.05)
    ap.add_argument("--train-slice", type=int, default=0, help="0 = full train split")
    args = ap.parse_args()

    bench = load_benchmark()
    doc_ids = list(bench.corpus)
    texts = [doc_text(bench.corpus[d]) for d in doc_ids]
    corpus_map = dict(zip(doc_ids, texts))
    vecs = cached_doc_embeddings(DenseEmbedder(), doc_ids, texts)

    train_ids = bench.splits["train"]
    if args.train_slice:
        train_ids = train_ids[: args.train_slice]
    train_q = {q: bench.queries[q] for q in train_ids}
    optimize_q = bench.subset("optimize")

    loop = SearchLoop(
        corpus_map, train_q, optimize_q, bench.qrels,
        doc_embeddings=vecs,
        fitness_metric=args.metric,
        max_rounds=args.max_rounds,
        max_candidates=args.max_candidates,
        accept_p=args.accept_p,
        forbid_ids=set(bench.splits["test"]),
    )
    result = loop.run()

    print(f"\nfitness = {args.metric} on the optimize split ({len(optimize_q)} queries)")
    print(f"run dir : {result.run_dir}\n")
    hdr = f"{'id':>3} {'rnd':>3} {'par':>3}  {'mutation':<26} {'targets':<14} " \
          f"{'fitness':>9} {'Δparent':>9} {'p':>7}  acc"
    print(hdr)
    print("-" * len(hdr))
    for t in result.trials:
        print(f"{t.trial_id:>3} {t.round:>3} "
              f"{'' if t.parent_id is None else t.parent_id:>3}  "
              f"{t.mutation:<26} {t.targets:<14} "
              f"{t.fitness:>9.4f} {t.delta_vs_parent:>+9.4f} {t.p_value:>7.4f}  "
              f"{'Y' if t.accepted else '.'}")

    print(f"\nbaseline {args.metric} = {result.baseline_fitness:.4f}")
    print(f"best     {args.metric} = {result.best_fitness:.4f}  "
          f"({'improved' if result.improved else 'no improvement'})")
    print("\nbandit arms:")
    for name, s in result.bandit.items():
        print(f"  {name:<26} pulls={s['pulls']} mean_reward={s['mean_reward']}")
    print(f"\nbest config saved to {result.run_dir / 'best.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
