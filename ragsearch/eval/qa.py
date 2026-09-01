"""QA answer-string metrics: SQuAD-style exact match and token F1.

Kept independent of any particular reader. Feed it predicted answer strings
and one-or-more gold strings per question; a simple lexical answerer that
pulls the best sentence out of retrieved context is provided so the QA path
is runnable end to end on datasets that have short answers.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Mapping, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    npred = normalize_answer(prediction)
    return 1.0 if any(npred == normalize_answer(g) for g in golds) else 0.0


def f1(prediction: str, golds: Sequence[str]) -> float:
    pred_toks = _tokens(prediction)
    best = 0.0
    for gold in golds:
        gold_toks = _tokens(gold)
        if not pred_toks and not gold_toks:
            best = max(best, 1.0)
            continue
        if not pred_toks or not gold_toks:
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_toks)
        recall = overlap / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def evaluate_qa(
    predictions: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    ids = list(references)
    if not ids:
        return {"em": 0.0, "f1": 0.0, "n": 0}
    em = sum(exact_match(predictions.get(qid, ""), references[qid]) for qid in ids)
    f = sum(f1(predictions.get(qid, ""), references[qid]) for qid in ids)
    return {"em": em / len(ids), "f1": f / len(ids), "n": len(ids)}


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class LexicalAnswerer:
    """Pick the retrieved-context sentence with the most question-term overlap."""

    def __init__(self, context_docs: int = 3):
        self.context_docs = context_docs

    def answer(self, question: str, ranked_texts: Sequence[str]) -> str:
        q_terms = Counter(_tokens(question))
        best_sent, best_score = "", -1.0
        for text in ranked_texts[: self.context_docs]:
            for sent in _SENT_SPLIT.split(text):
                s_terms = Counter(_tokens(sent))
                score = sum((q_terms & s_terms).values())
                if score > best_score:
                    best_sent, best_score = sent.strip(), score
        return best_sent
