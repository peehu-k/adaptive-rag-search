"""Run failure diagnosis across a whole query set and cluster the results."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ragsearch.diagnose.failures import (
    ALL_MODES,
    FAILURE_MODES,
    QueryDiagnosis,
    diagnose_query,
)
from ragsearch.retrieve.base import Hit


def _ids(hits: Sequence) -> list[str]:
    return [h.doc_id if isinstance(h, Hit) else h[0] for h in hits]


@dataclass
class FailureReport:
    diagnoses: list[QueryDiagnosis]
    k: int

    def by_mode(self) -> dict[str, list[QueryDiagnosis]]:
        out: dict[str, list[QueryDiagnosis]] = {m: [] for m in ALL_MODES}
        for d in self.diagnoses:
            out[d.mode].append(d)
        return out

    def counts(self) -> dict[str, int]:
        c = Counter(d.mode for d in self.diagnoses)
        return {m: c.get(m, 0) for m in ALL_MODES}

    def scored(self) -> list[QueryDiagnosis]:
        return [d for d in self.diagnoses if d.mode != "no_gold"]

    def failures(self) -> list[QueryDiagnosis]:
        return [d for d in self.diagnoses if d.is_failure]

    def recoverable_by_retriever(self) -> dict[str, int]:
        tally: dict[str, int] = defaultdict(int)
        for d in self.failures():
            for name in d.recoverable_by:
                tally[name] += 1
        return dict(tally)

    def summary(self) -> str:
        scored = self.scored()
        n = len(scored) or 1
        lines = [f"diagnosed {len(scored)} scored queries (k={self.k})", ""]
        lines.append(f"{'mode':<16}{'count':>8}{'share':>9}")
        lines.append("-" * 33)
        counts = self.counts()
        for mode in ALL_MODES:
            if mode == "no_gold" and counts[mode] == 0:
                continue
            denom = n if mode != "no_gold" else len(self.diagnoses)
            lines.append(f"{mode:<16}{counts[mode]:>8}{counts[mode] / denom:>8.1%}")
        rec = self.recoverable_by_retriever()
        if rec:
            lines.append("")
            lines.append("failures with a gold already in one retriever's pool:")
            for name, cnt in sorted(rec.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {name:<12}{cnt:>6}")
        return "\n".join(lines)

    def examples(self, mode: str, n: int = 5) -> list[QueryDiagnosis]:
        return [d for d in self.diagnoses if d.mode == mode][:n]


def diagnose_hybrid(
    hybrid,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    k: int = 10,
    answer_correct: Mapping[str, bool] | None = None,
) -> FailureReport:
    pool_depth = hybrid.candidate_k
    diagnoses: list[QueryDiagnosis] = []
    for qid, text in queries.items():
        gold = [d for d, g in qrels.get(qid, {}).items() if g > 0]
        trace = hybrid.search_trace(text)
        diagnoses.append(
            diagnose_query(
                qid,
                gold,
                per_retriever={n: _ids(h) for n, h in trace.per_retriever.items()},
                fused=_ids(trace.fused),
                final=_ids(trace.final),
                k=k,
                pool_depth=pool_depth,
                has_reranker=trace.has_reranker,
                answer_correct=None if answer_correct is None else answer_correct.get(qid),
            )
        )
    return FailureReport(diagnoses=diagnoses, k=k)
