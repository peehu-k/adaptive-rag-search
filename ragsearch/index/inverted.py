"""A plain inverted index built from scratch.

Stores, for every analyzed term, the postings list ``[(doc_ordinal, tf), ...]``
plus the per-document length and the collection statistics BM25 needs
(``N``, ``avgdl``, document frequencies). Documents are referred to inside the
index by a contiguous integer ordinal; the string ids are kept in ``doc_ids``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ragsearch.index.tokenizer import Analyzer, DEFAULT_ANALYZER


@dataclass
class InvertedIndex:
    analyzer: Analyzer = DEFAULT_ANALYZER
    doc_ids: list[str] = field(default_factory=list)
    doc_len: list[int] = field(default_factory=list)
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    # --- derived collection statistics -------------------------------------
    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def total_terms(self) -> int:
        return sum(self.doc_len)

    @property
    def avg_doc_len(self) -> float:
        return (self.total_terms / self.num_docs) if self.num_docs else 0.0

    def doc_freq(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    # --- construction -----------------------------------------------------
    @classmethod
    def build(
        cls,
        documents: dict[str, str],
        analyzer: Analyzer = DEFAULT_ANALYZER,
    ) -> "InvertedIndex":
        idx = cls(analyzer=analyzer)
        accum: dict[str, list[tuple[int, int]]] = {}
        for ordinal, (doc_id, text) in enumerate(documents.items()):
            terms = analyzer(text)
            idx.doc_ids.append(doc_id)
            idx.doc_len.append(len(terms))
            for term, tf in Counter(terms).items():
                accum.setdefault(term, []).append((ordinal, tf))
        # freeze postings in ordinal order (they already are, by construction)
        idx.postings = accum
        return idx

    # --- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "analyzer": {
                "lowercase": self.analyzer.lowercase,
                "remove_stopwords": self.analyzer.remove_stopwords,
                "stem": self.analyzer.stem,
                "min_token_len": self.analyzer.min_token_len,
            },
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "postings": {t: p for t, p in self.postings.items()},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        analyzer = Analyzer(**payload["analyzer"])
        postings = {
            term: [(int(o), int(tf)) for o, tf in plist]
            for term, plist in payload["postings"].items()
        }
        return cls(
            analyzer=analyzer,
            doc_ids=list(payload["doc_ids"]),
            doc_len=list(payload["doc_len"]),
            postings=postings,
        )
