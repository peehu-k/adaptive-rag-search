"""Unit checks on the analyzer, the inverted index, and BM25 scoring."""

import math

import pytest

from ragsearch import benchmark
from ragsearch.index.inverted import InvertedIndex
from ragsearch.index.tokenizer import Analyzer, light_stem
from ragsearch.retrieve.bm25 import BM25Retriever

TOY = {
    "d1": "the quick brown fox jumps over the lazy dog",
    "d2": "a quick brown dog outpaces a quick fox",
    "d3": "lazy cats sleep all day in the warm sun",
    "d4": "quantum chromodynamics describes the strong interaction",
}


def test_analyzer_lowercases_and_drops_stopwords():
    toks = Analyzer()("The Quick Brown Fox and THE dog")
    assert "the" not in toks and "and" not in toks
    assert toks == ["quick", "brown", "fox", "dog"]


def test_analyzer_stopwords_can_be_kept():
    toks = Analyzer(remove_stopwords=False)("the quick fox")
    assert toks == ["the", "quick", "fox"]


def test_light_stem_examples():
    assert light_stem("studies") == "study"
    assert light_stem("running") == "run"
    assert light_stem("classes") == "classe" or light_stem("classes") == "class"
    assert light_stem("dog") == "dog"  # too short to touch


def test_index_collection_stats():
    idx = InvertedIndex.build(TOY)
    assert idx.num_docs == 4
    assert idx.doc_freq("quick") == 2
    assert idx.doc_freq("lazy") == 2
    assert idx.doc_freq("nonexistent") == 0
    assert idx.avg_doc_len == pytest.approx(idx.total_terms / 4)


def test_idf_decreases_with_document_frequency():
    r = BM25Retriever(InvertedIndex.build(TOY))
    # "quantum" appears in 1 doc, "quick" in 2 -> quantum must be rarer/higher
    assert r.idf("quantum") > r.idf("quick") > 0


def test_relevant_doc_outranks_irrelevant():
    r = BM25Retriever(InvertedIndex.build(TOY))
    hits = r.search("quick fox", k=4)
    assert hits[0].doc_id in {"d1", "d2"}
    top = {h.doc_id for h in hits[:2]}
    assert top == {"d1", "d2"}
    assert hits[0].score >= hits[-1].score


def test_length_normalization_prefers_shorter_doc():
    # same query term count, shorter document should score higher when b>0
    docs = {
        "short": "signal signal",
        "long": "signal signal " + " ".join(f"filler{i}" for i in range(40)),
    }
    r = BM25Retriever(InvertedIndex.build(docs), b=0.75)
    hits = {h.doc_id: h.score for h in r.search("signal", k=2)}
    assert hits["short"] > hits["long"]


def test_empty_query_returns_nothing():
    r = BM25Retriever(InvertedIndex.build(TOY))
    assert r.search("the a of") == []


def test_save_load_roundtrip(tmp_path):
    idx = InvertedIndex.build(TOY, analyzer=Analyzer(stem=True))
    path = tmp_path / "idx.json"
    idx.save(path)
    back = InvertedIndex.load(path)
    assert back.doc_ids == idx.doc_ids
    assert back.doc_len == idx.doc_len
    assert back.postings == idx.postings
    assert back.analyzer == idx.analyzer

    a = BM25Retriever(idx).search("quick brown", k=4)
    b = BM25Retriever(back).search("quick brown", k=4)
    assert a == b


@pytest.mark.skipif(
    not (benchmark.SCIFACT_DIR / "corpus.jsonl").exists(),
    reason="run scripts/get_data.py to fetch the SciFact benchmark",
)
def test_bm25_reaches_expected_recall_on_scifact():
    bench = benchmark.load_benchmark()
    documents = {d: benchmark.doc_text(doc) for d, doc in bench.corpus.items()}
    r = BM25Retriever(InvertedIndex.build(documents))

    hit, total = 0, 0
    for qid in bench.splits["optimize"]:
        relevant = {d for d, rel in bench.qrels.get(qid, {}).items() if rel > 0}
        if not relevant:
            continue
        got = {h.doc_id for h in r.search(bench.queries[qid], k=100)}
        hit += len(got & relevant)
        total += len(relevant)
    recall_at_100 = hit / total
    assert recall_at_100 > 0.85, recall_at_100
