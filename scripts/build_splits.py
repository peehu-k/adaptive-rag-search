"""Materialize the train / optimize / held-out-test split and print its stats."""

from __future__ import annotations

from ragsearch.benchmark import SCIFACT_DIR, load_queries, materialize_splits


def main() -> int:
    manifest = materialize_splits()
    queries = load_queries()

    print(f"dataset:          {manifest['dataset']}")
    print(f"seed:             {manifest['seed']}")
    print(f"optimize_fraction:{manifest['optimize_fraction']:>6}")
    print(f"total queries:    {len(queries)}")
    print()
    print(f"{'split':<10}{'queries':>10}{'relevant docs':>16}")
    for name in ("train", "optimize", "test"):
        print(
            f"{name:<10}{manifest['counts'][name]:>10}"
            f"{manifest['relevant_docs'][name]:>16}"
        )
    print()

    # disjointness / coverage sanity checks
    splits = manifest["splits"]
    all_ids = [qid for ids in splits.values() for qid in ids]
    assert len(all_ids) == len(set(all_ids)), "splits overlap"
    assert set(all_ids).issubset(queries), "split references unknown query id"
    print(f"wrote {SCIFACT_DIR / 'splits.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
