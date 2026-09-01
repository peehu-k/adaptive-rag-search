"""One serializable description of a whole retrieval pipeline.

The search loop treats a :class:`PipelineConfig` as a point in the search
space: mutate a field, rebuild, re-evaluate. Every sub-stage's knobs live in
a small frozen dataclass so a mutation can target exactly one of them, and
``fingerprint()`` gives a stable id for the lineage log and for de-duping
configs that have already been tried.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class AnalyzerCfg:
    lowercase: bool = True
    remove_stopwords: bool = True
    stem: bool = False
    min_token_len: int = 2


@dataclass(frozen=True)
class BM25Cfg:
    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class DenseCfg:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    max_seq_length: int = 256


@dataclass(frozen=True)
class FusionCfg:
    method: str = "weighted"  # "weighted" | "rrf"
    weight_bm25: float = 1.0
    weight_dense: float = 1.0
    rrf_k: float = 60.0
    candidate_k: int = 200


@dataclass(frozen=True)
class RerankCfg:
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 100


@dataclass(frozen=True)
class PipelineConfig:
    use_bm25: bool = True
    use_dense: bool = True
    analyzer: AnalyzerCfg = AnalyzerCfg()
    bm25: BM25Cfg = BM25Cfg()
    dense: DenseCfg = DenseCfg()
    fusion: FusionCfg = FusionCfg()
    rerank: RerankCfg = RerankCfg()

    # --- validation -----------------------------------------------------
    def __post_init__(self) -> None:
        if not (self.use_bm25 or self.use_dense):
            raise ValueError("at least one of bm25 / dense must be enabled")
        if self.fusion.method not in ("weighted", "rrf"):
            raise ValueError(f"bad fusion method {self.fusion.method!r}")

    @property
    def is_hybrid(self) -> bool:
        return self.use_bm25 and self.use_dense

    # --- editing ------------------------------------------------------
    def with_section(self, name: str, **changes) -> "PipelineConfig":
        """Return a copy with one sub-config's fields overridden."""
        current = getattr(self, name)
        return replace(self, **{name: replace(current, **changes)})

    def with_fields(self, **changes) -> "PipelineConfig":
        return replace(self, **changes)

    # --- serialization ------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        return cls(
            use_bm25=data.get("use_bm25", True),
            use_dense=data.get("use_dense", True),
            analyzer=AnalyzerCfg(**data.get("analyzer", {})),
            bm25=BM25Cfg(**data.get("bm25", {})),
            dense=DenseCfg(**data.get("dense", {})),
            fusion=FusionCfg(**data.get("fusion", {})),
            rerank=RerankCfg(**data.get("rerank", {})),
        )

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]


DEFAULT_CONFIG = PipelineConfig()
