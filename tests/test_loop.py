"""Bandit ordering and the search loop's accept / lineage / held-out guards."""

import json

import pytest

from ragsearch import benchmark
from ragsearch.search.bandit import UCB1
from ragsearch.search.config import DEFAULT_CONFIG
from ragsearch.search.loop import SearchLoop


# --- UCB1 --------------------------------------------------------------
def test_ucb_tries_unpulled_arms_first():
    b = UCB1()
    b.update("a", 1.0)
    ordered = b.order(["a", "b", "c"])
    assert set(ordered[:2]) == {"b", "c"}  # unpulled arms come first
    assert ordered[-1] == "a"
    assert b.score("z") == float("inf")


def test_ucb_prefers_higher_reward_once_all_pulled():
    b = UCB1(c=0.1)  # small exploration term -> exploitation dominates
    for _ in range(5):
        b.update("good", 1.0)
        b.update("bad", 0.0)
    assert b.order(["good", "bad"]) == ["good", "bad"]


def test_ucb_snapshot_shape():
    b = UCB1()
    b.update("m", 1.0)
    b.update("m", 0.0)
    snap = b.snapshot()
    assert snap["m"]["pulls"] == 2
    assert snap["m"]["mean_reward"] == 0.5


# --- SearchLoop guards ----------------------------------------------
def test_loop_rejects_held_out_ids():
    with pytest.raises(ValueError):
        SearchLoop(
            {"d": "text"},
            train_queries={"q1": "a"},
            optimize_queries={"q2": "b"},
            qrels={},
            forbid_ids={"q1"},
        )


def test_loop_rejects_train_optimize_overlap():
    with pytest.raises(ValueError):
        SearchLoop(
            {"d": "text"},
            train_queries={"q1": "a"},
            optimize_queries={"q1": "a"},
            qrels={},
        )


# --- end to end on a small real slice ------------------------------
@pytest.mark.skipif(
    not (benchmark.SCIFACT_DIR / "corpus.jsonl").exists(),
    reason="run scripts/get_data.py to fetch the SciFact benchmark",
)
def test_search_loop_runs_and_logs_lineage(tmp_path, monkeypatch):
    from ragsearch.index.cache import cached_doc_embeddings
    from ragsearch.index.embedder import DenseEmbedder
    from ragsearch.search import loop as loop_mod

    bench = benchmark.load_benchmark()
    doc_ids = list(bench.corpus)
    texts = [benchmark.doc_text(bench.corpus[d]) for d in doc_ids]
    corpus_map = dict(zip(doc_ids, texts))
    try:
        vecs = cached_doc_embeddings(DenseEmbedder(), doc_ids, texts, verbose=False)
    except Exception as exc:
        pytest.skip(f"embeddings unavailable: {exc}")

    monkeypatch.setattr(loop_mod, "RUNS_DIR", tmp_path / "runs")

    train_q = {q: bench.queries[q] for q in bench.splits["train"][:40]}
    optimize_q = {q: bench.queries[q] for q in bench.splits["optimize"][:40]}

    sl = SearchLoop(
        corpus_map, train_q, optimize_q, bench.qrels,
        doc_embeddings=vecs,
        fitness_metric="ndcg@10",
        max_rounds=2,
        max_candidates=3,
        forbid_ids=set(bench.splits["test"]),
    )
    result = sl.run()

    # baseline trial always present and accepted
    assert result.trials[0].mutation == "baseline"
    assert result.trials[0].accepted

    # incumbent is monotone: best fitness never below baseline
    assert result.best_fitness >= result.baseline_fitness - 1e-9

    # every accepted non-baseline trial genuinely improved on its parent
    for t in result.trials:
        if t.accepted and t.parent_id is not None:
            assert t.delta_vs_parent > 0 and t.p_value < 0.05

    # lineage.jsonl on disk has one row per trial
    rows = [
        json.loads(line)
        for line in (result.run_dir / "lineage.jsonl").read_text().splitlines()
    ]
    assert len(rows) == len(result.trials)
    assert (result.run_dir / "best.json").exists()

    # the held-out test split was never evaluated
    scored_ids = set(optimize_q)
    assert not scored_ids & set(bench.splits["test"])
