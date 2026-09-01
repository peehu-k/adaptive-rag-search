"""Benchmark loading and the train / optimize / held-out-test split.

The benchmark is SciFact from the BEIR collection: a scientific-claim
verification set with a ~5K abstract corpus and short claim queries. It is
small enough to index and search many times during optimization while still
being a real retrieval task with graded relevance judgments.

Split policy
------------
- ``test``      -- BEIR's official SciFact test qrels. This is the held-out
                   split. Nothing in diagnosis or the search loop is allowed
                   to read it; it is only touched by the final report.
- ``train``     -- 70% of BEIR's train qrels. Used to diagnose why queries
                   fail and to fit any learned pipeline component.
- ``optimize``  -- the remaining 30% of BEIR's train qrels. Used as the
                   fitness signal that decides whether a mutation is kept.

The train/optimize cut is a deterministic function of the query ids (sorted,
then shuffled with a fixed seed), so the split is reproducible from the raw
data alone and is re-materialized to ``data/scifact/splits.json``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SCIFACT_DIR = DATA_ROOT / "scifact"

SPLIT_SEED = 20240501
OPTIMIZE_FRACTION = 0.30


@dataclass(frozen=True)
class Benchmark:
    """A loaded benchmark: corpus, queries, judgments, and the id splits."""

    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    splits: dict[str, list[str]]

    def subset(self, split: str) -> dict[str, str]:
        """Return ``{query_id: query_text}`` for one split."""
        if split not in self.splits:
            raise KeyError(f"unknown split {split!r}; have {sorted(self.splits)}")
        return {qid: self.queries[qid] for qid in self.splits[split]}


def doc_text(doc: dict[str, str]) -> str:
    """Flatten a corpus entry to the single string retrievers index over."""
    title = (doc.get("title") or "").strip()
    body = (doc.get("text") or "").strip()
    return f"{title}. {body}".strip() if title else body


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_corpus(scifact_dir: Path = SCIFACT_DIR) -> dict[str, dict[str, str]]:
    corpus: dict[str, dict[str, str]] = {}
    for row in _read_jsonl(scifact_dir / "corpus.jsonl"):
        corpus[str(row["_id"])] = {
            "title": row.get("title", "") or "",
            "text": row.get("text", "") or "",
        }
    return corpus


def load_queries(scifact_dir: Path = SCIFACT_DIR) -> dict[str, str]:
    queries: dict[str, str] = {}
    for row in _read_jsonl(scifact_dir / "queries.jsonl"):
        queries[str(row["_id"])] = row.get("text", "") or ""
    return queries


def load_qrels(scifact_dir: Path = SCIFACT_DIR) -> dict[str, dict[str, dict[str, int]]]:
    """Load the raw BEIR qrels, keyed by split name (``train`` / ``test``)."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for name in ("train", "test"):
        path = scifact_dir / "qrels" / f"{name}.tsv"
        judged: dict[str, dict[str, int]] = {}
        with path.open(encoding="utf-8") as fh:
            header = next(fh)  # query-id\tcorpus-id\tscore
            assert "query-id" in header, f"unexpected qrels header: {header!r}"
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                qid, did, score = line.split("\t")
                judged.setdefault(qid, {})[did] = int(score)
        out[name] = judged
    return out


def build_splits(
    raw_qrels: dict[str, dict[str, dict[str, int]]],
    *,
    seed: int = SPLIT_SEED,
    optimize_fraction: float = OPTIMIZE_FRACTION,
) -> dict[str, list[str]]:
    """Turn raw BEIR qrels into the train / optimize / test id lists."""
    test_ids = sorted(raw_qrels["test"])
    trainval_ids = sorted(raw_qrels["train"])

    shuffled = list(trainval_ids)
    random.Random(seed).shuffle(shuffled)
    n_optimize = round(len(shuffled) * optimize_fraction)
    optimize_ids = sorted(shuffled[:n_optimize])
    train_ids = sorted(shuffled[n_optimize:])

    return {"train": train_ids, "optimize": optimize_ids, "test": test_ids}


def materialize_splits(scifact_dir: Path = SCIFACT_DIR) -> dict:
    """Build splits from raw data and write ``splits.json``. Returns the manifest."""
    raw_qrels = load_qrels(scifact_dir)
    splits = build_splits(raw_qrels)
    merged_qrels: dict[str, dict[str, int]] = {}
    merged_qrels.update(raw_qrels["train"])
    merged_qrels.update(raw_qrels["test"])

    manifest = {
        "dataset": "beir/scifact",
        "seed": SPLIT_SEED,
        "optimize_fraction": OPTIMIZE_FRACTION,
        "counts": {name: len(ids) for name, ids in splits.items()},
        "relevant_docs": {
            name: sum(len(merged_qrels.get(qid, {})) for qid in ids)
            for name, ids in splits.items()
        },
        "splits": splits,
    }
    out_path = scifact_dir / "splits.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_splits(scifact_dir: Path = SCIFACT_DIR) -> dict[str, list[str]]:
    manifest = json.loads((scifact_dir / "splits.json").read_text(encoding="utf-8"))
    return manifest["splits"]


def load_benchmark(scifact_dir: Path = SCIFACT_DIR) -> Benchmark:
    raw_qrels = load_qrels(scifact_dir)
    qrels: dict[str, dict[str, int]] = {}
    qrels.update(raw_qrels["train"])
    qrels.update(raw_qrels["test"])
    return Benchmark(
        corpus=load_corpus(scifact_dir),
        queries=load_queries(scifact_dir),
        qrels=qrels,
        splits=load_splits(scifact_dir),
    )
