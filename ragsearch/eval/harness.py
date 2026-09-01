"""Run a retriever over a query set, score it, and compare two runs.

``evaluate_retrieval`` drives a retriever; ``score_run`` scores an
already-materialized ``{qid: [doc_id, ...]}`` mapping (useful for cached runs
and for the search loop, which reuses rankings). ``compare`` runs a paired
significance test between two :class:`RunResult` objects on one metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ragsearch.eval.metrics import score_ranking
from ragsearch.eval.significance import (
    SigResult,
    paired_bootstrap_test,
    paired_randomization_test,
)
from ragsearch.retrieve.base import Hit

DEFAULT_KS = (1, 5, 10, 100)


@dataclass
class RunResult:
    name: str
    ks: Sequence[int]
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    rankings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def query_ids(self) -> list[str]:
        return list(self.per_query)

    def aggregate(self) -> dict[str, float]:
        if not self.per_query:
            return {}
        keys = next(iter(self.per_query.values())).keys()
        n = len(self.per_query)
        return {
            key: sum(q[key] for q in self.per_query.values()) / n for key in keys
        }

    def column(self, metric: str) -> list[float]:
        return [self.per_query[q][metric] for q in self.query_ids]


def score_run(
    name: str,
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    ks: Sequence[int] = DEFAULT_KS,
) -> RunResult:
    res = RunResult(name=name, ks=tuple(ks))
    for qid, ranking in rankings.items():
        judged = qrels.get(qid, {})
        if not any(g > 0 for g in judged.values()):
            continue
        res.rankings[qid] = list(ranking)
        res.per_query[qid] = score_ranking(ranking, judged, ks)
    return res


def evaluate_retrieval(
    name: str,
    retriever,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
    ks: Sequence[int] = DEFAULT_KS,
    depth: int = 100,
) -> RunResult:
    depth = max(depth, max(ks))
    rankings: dict[str, list[str]] = {}
    for qid, text in queries.items():
        hits = retriever.search(text, k=depth)
        rankings[qid] = [h.doc_id if isinstance(h, Hit) else h[0] for h in hits]
    return score_run(name, rankings, qrels, ks)


def compare(
    baseline: RunResult,
    candidate: RunResult,
    metric: str,
    *,
    method: str = "randomization",
    iterations: int = 10000,
    seed: int = 12345,
) -> SigResult:
    shared = [q for q in baseline.query_ids if q in candidate.per_query]
    if not shared:
        raise ValueError("runs share no scored queries")
    a = [baseline.per_query[q][metric] for q in shared]
    b = [candidate.per_query[q][metric] for q in shared]
    test = paired_randomization_test if method == "randomization" else paired_bootstrap_test
    return test(a, b, metric=metric, iterations=iterations, seed=seed)


def format_table(results: Sequence[RunResult], metrics: Sequence[str]) -> str:
    header = f"{'config':<22}" + "".join(f"{m:>12}" for m in metrics)
    lines = [header, "-" * len(header)]
    for r in results:
        agg = r.aggregate()
        lines.append(
            f"{r.name:<22}" + "".join(f"{agg.get(m, 0.0):>12.4f}" for m in metrics)
        )
    return "\n".join(lines)
