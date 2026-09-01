"""Checks on the benchmark loader and the split policy.

Skipped entirely when the SciFact data has not been downloaded, so the suite
still runs on a clean checkout.
"""

import pytest

from ragsearch import benchmark

pytestmark = pytest.mark.skipif(
    not (benchmark.SCIFACT_DIR / "corpus.jsonl").exists(),
    reason="run scripts/get_data.py to fetch the SciFact benchmark",
)


@pytest.fixture(scope="module")
def raw_qrels():
    return benchmark.load_qrels()


def test_corpus_and_queries_load():
    corpus = benchmark.load_corpus()
    queries = benchmark.load_queries()
    assert len(corpus) > 4000
    assert len(queries) > 1000
    doc = next(iter(corpus.values()))
    assert set(doc) == {"title", "text"}


def test_split_is_deterministic(raw_qrels):
    a = benchmark.build_splits(raw_qrels)
    b = benchmark.build_splits(raw_qrels)
    assert a == b


def test_splits_are_disjoint_and_grounded(raw_qrels):
    splits = benchmark.build_splits(raw_qrels)
    queries = benchmark.load_queries()

    ids = [qid for group in splits.values() for qid in group]
    assert len(ids) == len(set(ids)), "splits overlap"
    assert set(ids).issubset(queries), "split references an unknown query id"

    # the held-out test split must be exactly BEIR's test qrels, untouched
    assert set(splits["test"]) == set(raw_qrels["test"])
    # optimize must not leak into train
    assert not (set(splits["train"]) & set(splits["optimize"]))


def test_optimize_fraction_is_respected(raw_qrels):
    splits = benchmark.build_splits(raw_qrels)
    trainval = len(splits["train"]) + len(splits["optimize"])
    frac = len(splits["optimize"]) / trainval
    assert abs(frac - benchmark.OPTIMIZE_FRACTION) < 0.02


def test_materialized_manifest_matches_loader():
    manifest_splits = benchmark.load_splits()
    fresh = benchmark.build_splits(benchmark.load_qrels())
    assert manifest_splits == fresh
