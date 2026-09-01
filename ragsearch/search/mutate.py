"""Turn a diagnosed failure cluster into targeted pipeline-config changes.

The proposer reads a :class:`FailureReport` and emits :class:`Mutation`
objects. Each mutation is one concrete edit to the current
:class:`PipelineConfig`, tagged with the failure mode it is meant to fix and
a rationale that cites the counts it is responding to. The search loop then
evaluates the mutations and keeps only the ones that actually help.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragsearch.diagnose.cluster import FailureReport
from ragsearch.diagnose.failures import (
    FUSION_MISS,
    GENERATION_MISS,
    RECALL_MISS,
    RERANKER_MISS,
)
from ragsearch.search.config import PipelineConfig


@dataclass
class Mutation:
    name: str
    targets: str  # the failure mode this addresses
    rationale: str
    config: PipelineConfig
    param_delta: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return self.config.fingerprint()

    @property
    def family(self) -> str:
        """Name with any trailing numeric / ``xN.NN`` suffix stripped."""
        return re.sub(r"_(x?\d+(\.\d+)?)$", "", self.name)


def _dominant_failures(report: FailureReport) -> list[tuple[str, int]]:
    counts = report.counts()
    modes = [(m, counts[m]) for m in (RECALL_MISS, FUSION_MISS, RERANKER_MISS, GENERATION_MISS)]
    return sorted((mc for mc in modes if mc[1] > 0), key=lambda mc: -mc[1])


class MutationProposer:
    def __init__(self, *, weight_step: float = 0.5, max_per_mode: int = 4):
        self.weight_step = weight_step
        self.max_per_mode = max_per_mode

    def propose(self, base: PipelineConfig, report: FailureReport) -> list[Mutation]:
        out: list[Mutation] = []
        scored = max(len(report.scored()), 1)
        for mode, count in _dominant_failures(report):
            share = count / scored
            if mode == FUSION_MISS:
                out += self._fusion_miss(base, report, count, share)
            elif mode == RECALL_MISS:
                out += self._recall_miss(base, report, count, share)
            elif mode == RERANKER_MISS:
                out += self._reranker_miss(base, report, count, share)
            elif mode == GENERATION_MISS:
                out += self._generation_miss(base, report, count, share)

        # drop no-ops and exact duplicates of the base or of each other
        seen = {base.fingerprint()}
        unique: list[Mutation] = []
        for m in out:
            fp = m.fingerprint
            if fp in seen:
                continue
            seen.add(fp)
            unique.append(m)
        return unique

    # --- per-mode strategies -----------------------------------------
    def _fusion_miss(self, base, report, count, share):
        rec = report.recoverable_by_retriever()
        muts: list[Mutation] = []
        cite = f"{count} fusion misses ({share:.0%} of scored); recoverable_by={rec or '-'}"

        lean_dense = rec.get("dense", 0) >= rec.get("bm25", 0)
        strong, weak = ("dense", "bm25") if lean_dense else ("bm25", "dense")
        for factor in (1 + self.weight_step, 1 + 2 * self.weight_step):
            w = {"bm25": base.fusion.weight_bm25, "dense": base.fusion.weight_dense}
            w[strong] = round(w[strong] * factor, 3)
            cfg = base.with_section(
                "fusion", weight_bm25=w["bm25"], weight_dense=w["dense"]
            )
            muts.append(Mutation(
                name=f"upweight_{strong}_x{factor:.2f}",
                targets=FUSION_MISS,
                rationale=f"{cite}; lift {strong} weight so pooled hits reach top-k",
                config=cfg,
                param_delta={"fusion.weight_bm25": w["bm25"],
                             "fusion.weight_dense": w["dense"]},
            ))

        # try the other fusion method
        other = "rrf" if base.fusion.method == "weighted" else "weighted"
        muts.append(Mutation(
            name=f"fusion_method_{other}",
            targets=FUSION_MISS,
            rationale=f"{cite}; swap fusion method {base.fusion.method}->{other}",
            config=base.with_section("fusion", method=other),
            param_delta={"fusion.method": other},
        ))

        # deeper candidate pool in case fusion is clipping the tail
        if base.fusion.candidate_k < 400:
            deeper = min(base.fusion.candidate_k * 2, 500)
            muts.append(Mutation(
                name=f"candidate_k_{deeper}",
                targets=FUSION_MISS,
                rationale=f"{cite}; widen candidate pool {base.fusion.candidate_k}->{deeper}",
                config=base.with_section("fusion", candidate_k=deeper),
                param_delta={"fusion.candidate_k": deeper},
            ))
        return muts[: self.max_per_mode]

    def _recall_miss(self, base, report, count, share):
        cite = f"{count} recall misses ({share:.0%}); gold absent from every candidate pool"
        muts: list[Mutation] = []
        if not base.analyzer.stem:
            muts.append(Mutation(
                name="analyzer_enable_stem",
                targets=RECALL_MISS,
                rationale=f"{cite}; enable stemming to merge inflected term variants",
                config=base.with_section("analyzer", stem=True),
                param_delta={"analyzer.stem": True},
            ))
        if base.analyzer.remove_stopwords:
            muts.append(Mutation(
                name="analyzer_keep_stopwords",
                targets=RECALL_MISS,
                rationale=f"{cite}; keep stopwords (claims are short, terms are scarce)",
                config=base.with_section("analyzer", remove_stopwords=False),
                param_delta={"analyzer.remove_stopwords": False},
            ))
        if base.analyzer.min_token_len > 1:
            muts.append(Mutation(
                name="analyzer_min_token_len_1",
                targets=RECALL_MISS,
                rationale=f"{cite}; index 1-char tokens (gene / symbol names)",
                config=base.with_section("analyzer", min_token_len=1),
                param_delta={"analyzer.min_token_len": 1},
            ))
        deeper = min(max(base.fusion.candidate_k * 2, 300), 1000)
        if deeper != base.fusion.candidate_k:
            muts.append(Mutation(
                name=f"candidate_k_{deeper}",
                targets=RECALL_MISS,
                rationale=f"{cite}; retrieve deeper before fusion",
                config=base.with_section("fusion", candidate_k=deeper),
                param_delta={"fusion.candidate_k": deeper},
            ))
        return muts[: self.max_per_mode]

    def _reranker_miss(self, base, report, count, share):
        cite = f"{count} reranker misses ({share:.0%}); rerank dropped a top-k gold"
        muts: list[Mutation] = []
        if base.rerank.enabled:
            tighter = max(base.rerank.top_n // 2, 20)
            muts.append(Mutation(
                name=f"rerank_top_n_{tighter}",
                targets=RERANKER_MISS,
                rationale=f"{cite}; rerank a shorter head so good fused hits survive",
                config=base.with_section("rerank", top_n=tighter),
                param_delta={"rerank.top_n": tighter},
            ))
            muts.append(Mutation(
                name="rerank_disabled",
                targets=RERANKER_MISS,
                rationale=f"{cite}; disable reranker and keep fused order",
                config=base.with_section("rerank", enabled=False),
                param_delta={"rerank.enabled": False},
            ))
        return muts[: self.max_per_mode]

    def _generation_miss(self, base, report, count, share):
        # A generation miss means retrieval already put the gold in context, so
        # there is no retrieval-config edit that fixes it. It is surfaced by the
        # diagnosis report and left for the QA stage; the proposer emits nothing.
        return []
