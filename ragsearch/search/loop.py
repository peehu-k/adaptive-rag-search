"""The config search loop.

Greedy evolutionary hill-climb with a UCB1 bandit deciding which mutation
families to evaluate first:

1. Evaluate the incumbent config on the ``optimize`` split -> baseline fitness.
2. Each round: diagnose the incumbent on ``train``, ask the proposer for
   targeted mutations, order them by the bandit, and evaluate each on
   ``optimize``.
3. A mutation is *accepted* only if it beats the incumbent on the fitness
   metric with a paired randomization p-value below ``accept_p``. The best
   accepted mutation becomes the new incumbent.
4. Stop after ``max_rounds`` or ``patience`` improvement-free rounds.

Every config that is evaluated -- accepted or not -- is written to
``lineage.jsonl`` with its parent, the mutation that produced it, its scores,
and the significance test against its parent. The ``test`` split is never
referenced; the loop raises if a forbidden id shows up in its query sets.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from ragsearch.diagnose.cluster import diagnose_hybrid
from ragsearch.eval.harness import RunResult, evaluate_retrieval
from ragsearch.eval.significance import paired_randomization_test
from ragsearch.search.bandit import UCB1
from ragsearch.search.build import build_pipeline
from ragsearch.search.config import DEFAULT_CONFIG, PipelineConfig
from ragsearch.search.mutate import MutationProposer

RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"


@dataclass
class Trial:
    trial_id: int
    round: int
    parent_id: int | None
    mutation: str
    family: str
    targets: str
    fingerprint: str
    fitness: float
    delta_vs_parent: float
    p_value: float
    accepted: bool
    seconds: float
    aggregate: dict = field(default_factory=dict)
    param_delta: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    baseline_fitness: float
    best_fitness: float
    best_config: PipelineConfig
    trials: list[Trial]
    run_dir: Path
    bandit: dict

    @property
    def improved(self) -> bool:
        return self.best_fitness > self.baseline_fitness


class SearchLoop:
    def __init__(
        self,
        corpus_map: Mapping[str, str],
        train_queries: Mapping[str, str],
        optimize_queries: Mapping[str, str],
        qrels: Mapping[str, Mapping[str, float]],
        *,
        doc_embeddings=None,
        start_config: PipelineConfig = DEFAULT_CONFIG,
        fitness_metric: str = "ndcg@10",
        k: int = 10,
        accept_p: float = 0.05,
        max_rounds: int = 4,
        patience: int = 1,
        max_candidates: int = 6,
        forbid_ids: set[str] | None = None,
        seed: int = 12345,
    ):
        forbid = forbid_ids or set()
        leaked = (set(train_queries) | set(optimize_queries)) & forbid
        if leaked:
            raise ValueError(
                f"search query sets contain {len(leaked)} forbidden (held-out) ids"
            )
        if set(train_queries) & set(optimize_queries):
            raise ValueError("train and optimize query sets overlap")

        self.corpus_map = dict(corpus_map)
        self.train_queries = dict(train_queries)
        self.optimize_queries = dict(optimize_queries)
        self.qrels = qrels
        self.doc_embeddings = doc_embeddings
        self.start_config = start_config
        self.fitness_metric = fitness_metric
        self.k = k
        self.accept_p = accept_p
        self.max_rounds = max_rounds
        self.patience = patience
        self.max_candidates = max_candidates
        self.seed = seed

        self.query_cache: dict = {}
        self.bandit = UCB1()
        self.proposer = MutationProposer()
        self._trials: list[Trial] = []
        self._next_id = 0

    # --- helpers ------------------------------------------------------
    def _evaluate(self, config: PipelineConfig) -> RunResult:
        pipeline = build_pipeline(
            config,
            self.corpus_map,
            doc_embeddings=self.doc_embeddings,
            query_cache=self.query_cache,
        )
        return evaluate_retrieval(
            config.fingerprint(),
            pipeline,
            self.optimize_queries,
            self.qrels,
            ks=(1, 5, self.k, 100),
            depth=100,
        )

    def _fit(self, result: RunResult) -> float:
        return result.aggregate().get(self.fitness_metric, 0.0)

    # --- main loop --------------------------------------------------
    def run(self) -> SearchResult:
        RUNS_DIR.mkdir(exist_ok=True)
        run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        lineage = (run_dir / "lineage.jsonl").open("w", encoding="utf-8")

        t0 = time.perf_counter()
        incumbent_cfg = self.start_config
        incumbent_res = self._evaluate(incumbent_cfg)
        baseline_fit = self._fit(incumbent_res)
        seen: set[str] = {incumbent_cfg.fingerprint()}
        incumbent_id = self._log(
            lineage, round_=0, parent_id=None, mutation="baseline", family="baseline",
            targets="-", config=incumbent_cfg, result=incumbent_res,
            delta=0.0, p=1.0, accepted=True, seconds=time.perf_counter() - t0,
            param_delta={},
        )

        no_improve = 0
        for rnd in range(1, self.max_rounds + 1):
            report = diagnose_hybrid(
                build_pipeline(
                    incumbent_cfg, self.corpus_map,
                    doc_embeddings=self.doc_embeddings, query_cache=self.query_cache,
                ),
                self.train_queries, self.qrels, k=self.k,
            )
            candidates = [
                m for m in self.proposer.propose(incumbent_cfg, report)
                if m.fingerprint not in seen
            ]
            if not candidates:
                no_improve += 1
                if no_improve > self.patience:
                    break
                continue
            fam_first = self.bandit.order({m.family for m in candidates})
            candidates.sort(key=lambda m: fam_first.index(m.family))
            candidates = candidates[: self.max_candidates]

            round_best = None
            for mut in candidates:
                seen.add(mut.fingerprint)
                ct0 = time.perf_counter()
                res = self._evaluate(mut.config)
                a = [incumbent_res.per_query[q][self.fitness_metric]
                     for q in incumbent_res.query_ids if q in res.per_query]
                b = [res.per_query[q][self.fitness_metric]
                     for q in incumbent_res.query_ids if q in res.per_query]
                sig = paired_randomization_test(
                    a, b, metric=self.fitness_metric, iterations=5000, seed=self.seed
                )
                accepted = sig.delta > 0 and sig.p_value < self.accept_p
                self.bandit.update(mut.family, 1.0 if sig.delta > 0 else 0.0)
                tid = self._log(
                    lineage, round_=rnd, parent_id=incumbent_id, mutation=mut.name,
                    family=mut.family, targets=mut.targets, config=mut.config,
                    result=res, delta=sig.delta, p=sig.p_value, accepted=accepted,
                    seconds=time.perf_counter() - ct0, param_delta=mut.param_delta,
                )
                if accepted and (round_best is None or sig.delta > round_best[2]):
                    round_best = (tid, mut, sig.delta, res)

            if round_best is None:
                no_improve += 1
                if no_improve > self.patience:
                    break
                continue

            incumbent_id, mut, _, incumbent_res = round_best
            incumbent_cfg = mut.config
            no_improve = 0

        lineage.close()
        best_fit = self._fit(incumbent_res)
        result = SearchResult(
            baseline_fitness=baseline_fit,
            best_fitness=best_fit,
            best_config=incumbent_cfg,
            trials=self._trials,
            run_dir=run_dir,
            bandit=self.bandit.snapshot(),
        )
        (run_dir / "best.json").write_text(
            json.dumps(
                {
                    "fitness_metric": self.fitness_metric,
                    "baseline_fitness": baseline_fit,
                    "best_fitness": best_fit,
                    "improved": result.improved,
                    "best_config": incumbent_cfg.to_dict(),
                    "bandit": self.bandit.snapshot(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    # --- logging ---------------------------------------------------
    def _log(self, lineage, *, round_, parent_id, mutation, family, targets,
             config, result, delta, p, accepted, seconds, param_delta) -> int:
        tid = self._next_id
        self._next_id += 1
        trial = Trial(
            trial_id=tid,
            round=round_,
            parent_id=parent_id,
            mutation=mutation,
            family=family,
            targets=targets,
            fingerprint=config.fingerprint(),
            fitness=round(self._fit(result), 6),
            delta_vs_parent=round(delta, 6),
            p_value=round(p, 6),
            accepted=accepted,
            seconds=round(seconds, 2),
            aggregate={k: round(v, 6) for k, v in result.aggregate().items()},
            param_delta=param_delta,
            config=config.to_dict(),
        )
        self._trials.append(trial)
        lineage.write(json.dumps(asdict(trial)) + "\n")
        lineage.flush()
        return tid
