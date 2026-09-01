"""BM25 ranking over an :class:`InvertedIndex`, implemented from scratch.

Scoring is the Robertson/Sparck-Jones Okapi BM25 with the non-negative idf
variant::

    idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(q, d) = sum over t in q of
                  idf(t) * ( f(t,d) * (k1 + 1) )
                          / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

Repeated query terms contribute proportionally to their query-side count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ragsearch.index.inverted import InvertedIndex
from ragsearch.retrieve.base import Hit


@dataclass
class BM25Retriever:
    index: InvertedIndex
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self._avgdl = self.index.avg_doc_len or 1.0
        n = self.index.num_docs
        # precompute idf per term; clamp at 0 so a term in almost every doc
        # can never subtract from a document's score
        self._idf: dict[str, float] = {}
        for term, plist in self.index.postings.items():
            df = len(plist)
            self._idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def idf(self, term: str) -> float:
        return self._idf.get(term, 0.0)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q_terms = self.index.analyzer(query)
        if not q_terms:
            return []

        k1, b, avgdl = self.k1, self.b, self._avgdl
        doc_len = self.index.doc_len
        scores: dict[int, float] = {}

        q_counts: dict[str, int] = {}
        for t in q_terms:
            q_counts[t] = q_counts.get(t, 0) + 1

        for term, qtf in q_counts.items():
            plist = self.index.postings.get(term)
            if not plist:
                continue
            idf = self._idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            weight = idf * qtf
            for ordinal, tf in plist:
                denom = tf + k1 * (1.0 - b + b * doc_len[ordinal] / avgdl)
                scores[ordinal] = scores.get(ordinal, 0.0) + weight * (
                    tf * (k1 + 1.0)
                ) / denom

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        doc_ids = self.index.doc_ids
        return [Hit(doc_ids[o], s) for o, s in ranked]
