"""Label *why* a query failed, not just that it did.

Given, for one query, the per-retriever candidate lists, the fused order, the
final (post-rerank) order, and the gold document ids, decide which stage is
responsible:

``recall_miss``
    No gold document appears in any base retriever's candidate pool. Fusion
    and reranking never had a chance; the fix is upstream (analyzer,
    chunking, a different / additional retriever).
``fusion_miss``
    A gold document is in the pool but fusion ranked it below ``k``. The
    signal exists; weights / method / candidate depth are the lever.
``reranker_miss``
    A gold document was inside the fused top-``k`` but the reranker pushed it
    out. Only possible when a reranker is active.
``generation_miss``
    A gold document is in the final top-``k`` context, yet the produced
    answer is wrong -- the reader ignored the evidence.
``success``
    A gold document is in the final top-``k`` (and, if an answer was scored,
    it is correct).
``no_gold``
    The query has no judgments; nothing to diagnose.

Each diagnosis also records ``found_by`` -- which base retrievers surfaced a
gold anywhere in their pool -- so the mutation proposer knows whether the
missing signal is lexical or semantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

RECALL_MISS = "recall_miss"
FUSION_MISS = "fusion_miss"
RERANKER_MISS = "reranker_miss"
GENERATION_MISS = "generation_miss"
SUCCESS = "success"
NO_GOLD = "no_gold"

FAILURE_MODES = (RECALL_MISS, FUSION_MISS, RERANKER_MISS, GENERATION_MISS)
ALL_MODES = FAILURE_MODES + (SUCCESS, NO_GOLD)

_INF = 10**9


def _best_rank(ranking: Sequence[str], gold: set[str], limit: int | None = None) -> int:
    stop = len(ranking) if limit is None else min(limit, len(ranking))
    for i in range(stop):
        if ranking[i] in gold:
            return i + 1
    return _INF


@dataclass
class QueryDiagnosis:
    qid: str
    mode: str
    gold: list[str]
    k: int
    pool_depth: int
    best_pool_rank: int
    best_fused_rank: int
    best_final_rank: int
    found_by: list[str] = field(default_factory=list)
    answer_correct: bool | None = None

    @property
    def is_failure(self) -> bool:
        return self.mode in FAILURE_MODES

    @property
    def recoverable_by(self) -> list[str]:
        """Base retrievers that had a gold in-pool while the pipeline failed."""
        return self.found_by if self.is_failure else []


def diagnose_query(
    qid: str,
    gold_ids: Sequence[str],
    *,
    per_retriever: Mapping[str, Sequence[str]],
    fused: Sequence[str],
    final: Sequence[str],
    k: int = 10,
    pool_depth: int = 200,
    has_reranker: bool = False,
    answer_correct: bool | None = None,
) -> QueryDiagnosis:
    gold = {g for g in gold_ids}
    if not gold:
        return QueryDiagnosis(qid, NO_GOLD, [], k, pool_depth, _INF, _INF, _INF)

    found_by = [
        name
        for name, ranking in per_retriever.items()
        if _best_rank(ranking, gold, pool_depth) < _INF
    ]
    best_pool_rank = min(
        (_best_rank(r, gold, pool_depth) for r in per_retriever.values()),
        default=_INF,
    )
    best_fused_rank = _best_rank(fused, gold)
    best_final_rank = _best_rank(final, gold)

    if best_final_rank <= k:
        mode = SUCCESS if answer_correct in (None, True) else GENERATION_MISS
    elif best_pool_rank == _INF:
        mode = RECALL_MISS
    elif has_reranker and best_fused_rank <= k:
        mode = RERANKER_MISS
    else:
        mode = FUSION_MISS

    return QueryDiagnosis(
        qid=qid,
        mode=mode,
        gold=sorted(gold),
        k=k,
        pool_depth=pool_depth,
        best_pool_rank=best_pool_rank,
        best_fused_rank=best_fused_rank,
        best_final_rank=best_final_rank,
        found_by=found_by,
        answer_correct=answer_correct,
    )
