"""QA metric checks: SQuAD-style normalization, token F1, lexical answerer."""

import pytest

from ragsearch.eval.qa import (
    LexicalAnswerer,
    evaluate_qa,
    exact_match,
    f1,
    normalize_answer,
)


def test_normalize_strips_articles_punctuation_case():
    assert normalize_answer("  The Quick, Brown FOX! ") == "quick brown fox"
    assert normalize_answer("an apple") == "apple"


def test_exact_match_is_normalization_insensitive():
    assert exact_match("The answer.", ["answer"]) == 1.0
    assert exact_match("answer", ["something else", "the ANSWER"]) == 1.0
    assert exact_match("nope", ["answer"]) == 0.0


def test_f1_partial_overlap():
    # pred "happy brown dog" vs gold "brown dog": p=2/3, r=2/2 -> F1 = 0.8
    assert f1("happy brown dog", ["brown dog"]) == pytest.approx(0.8)
    assert f1("totally wrong", ["brown dog"]) == 0.0
    assert f1("brown dog", ["brown dog"]) == pytest.approx(1.0)
    # articles are stripped before scoring, like SQuAD
    assert f1("the brown dog", ["brown dog"]) == pytest.approx(1.0)


def test_f1_takes_best_gold():
    assert f1("brown dog", ["cat", "brown dog"]) == pytest.approx(1.0)


def test_evaluate_qa_aggregates():
    preds = {"q1": "Paris", "q2": "a blue whale here", "q3": "wrong"}
    refs = {"q1": ["paris"], "q2": ["blue whale"], "q3": ["right"]}
    out = evaluate_qa(preds, refs)
    assert out["n"] == 3
    # q1 exact, q2 not exact (extra token), q3 miss
    assert out["em"] == pytest.approx(1 / 3)
    # q2 f1: p=2/3, r=1 -> 0.8
    assert out["f1"] == pytest.approx((1.0 + 0.8 + 0.0) / 3)


def test_lexical_answerer_picks_overlapping_sentence():
    ans = LexicalAnswerer(context_docs=2)
    texts = [
        "Mitochondria are organelles. They produce ATP through respiration.",
        "The stock market fell today amid inflation worries.",
    ]
    out = ans.answer("what do mitochondria produce through respiration", texts)
    assert "ATP" in out
