"""Run-2 guarantees: excluded third-party spans stay out, quotes get trimmed."""
from __future__ import annotations

import json

import pytest
import yaml

from src.gates import (MAX_QUOTE_WORDS, TRIM_MARKER, check_quotes,
                       trim_to_limit)
from src.process import CHUNKS_JSONL, SOURCES, load_chunks
from src.textutil import word_count

pytestmark = pytest.mark.skipif(not CHUNKS_JSONL.exists(),
                                reason="corpus not built; run `make process`")


@pytest.fixture(scope="module")
def corpus_text():
    return " ".join(c["text"] for c in load_chunks()).casefold()


@pytest.fixture(scope="module")
def sources():
    return yaml.safe_load(SOURCES.read_text(encoding="utf-8"))["documents"]


def test_every_excluded_span_marker_is_absent(corpus_text, sources):
    """The begin marker of each excluded span must not survive into the index."""
    checked = 0
    for entry in sources:
        for span in entry.get("exclude_spans") or []:
            checked += 1
            assert span["begin"].casefold() not in corpus_text, (
                f"{entry['document_id']}: excluded span survived: {span['begin']!r}")
    assert checked >= 4, f"expected at least 4 excluded spans, found {checked}"


def test_teh_cheang_wan_letter_is_absent(corpus_text):
    """A letter sent TO Lee is not Lee speaking."""
    for phrase in ("i have been feeling very sad and depressed",
                   "as an honourable oriental gentleman",
                   "yours faithfully"):
        assert phrase not in corpus_text


def test_drake_readers_digest_extracts_are_absent(corpus_text):
    for phrase in ("vietnam is the key to the safety",
                   "even more critical, however",
                   "60 miles wide at this narrowest point",
                   "ships drop anchor in singapore"):
        assert phrase not in corpus_text


def test_parliament_document_is_not_indexed_for_answering():
    """answer_use: false keeps it in the manifest but out of the index."""
    ids = {c["document_id"] for c in load_chunks()}
    assert "lky_1965-12-14_parliament-motion-of-thanks" not in ids
    manifest = json.loads((CHUNKS_JSONL.parent.parent / "manifest.json").read_text())
    row = next(d for d in manifest["documents"]
               if d["document_id"] == "lky_1965-12-14_parliament-motion-of-thanks")
    assert row["answer_use"] is False
    assert row["n_chunks"] == 0
    assert row["n_chunks_built"] > 0, "kept in the manifest for transparency"


def test_fitzgerald_passages_are_no_longer_answerable(corpus_text):
    """The two passages that leaked into run 1's answer (a)."""
    assert "aristocratic society dominated by the malay feudal families" not in corpus_text
    assert "clear majority of the population" not in corpus_text


# --- trimming --------------------------------------------------------------

SOURCE = ("We have to make a living. Nobody owes us a living, and nobody will "
          "give us a living. We must earn it ourselves, by being better "
          "organised and more disciplined than our neighbours.")


def test_long_verbatim_quote_is_trimmed_and_stays_a_substring():
    long_quote = " ".join(SOURCE.split()[:31])
    assert word_count(long_quote) == 31
    trimmed = trim_to_limit(long_quote, SOURCE)
    assert trimmed is not None
    assert trimmed.endswith(TRIM_MARKER)
    body = trimmed[:-len(TRIM_MARKER)]
    assert word_count(body) <= MAX_QUOTE_WORDS
    assert body in SOURCE, "the trimmed quote must still be verbatim"


def test_gate_auto_trims_and_counts_it():
    chunk = {"chunk_id": "d#000", "text": SOURCE, "document_id": "d",
             "quotable": True, "verbatim": True}
    answer = {"answer": "x", "mode": "documented", "citations": ["d#000"],
              "quotes": [{"text": " ".join(SOURCE.split()[:31]), "chunk_id": "d#000"}]}
    failures = check_quotes(answer, [chunk])
    assert failures == [], failures
    assert answer["_auto_trimmed"] == 1
    assert answer["_quote_limit_broken"] is True
    assert answer["quotes"][0]["text"].endswith(TRIM_MARKER)
    assert word_count(answer["quotes"][0]["text"]) <= MAX_QUOTE_WORDS + 1


def test_long_non_verbatim_quote_still_fails():
    chunk = {"chunk_id": "d#000", "text": SOURCE, "document_id": "d",
             "quotable": True, "verbatim": True}
    paraphrase = ("We must earn our own living because nobody anywhere owes this "
                  "country anything at all, and we must be better organised than "
                  "every single one of our neighbours")
    assert word_count(paraphrase) > MAX_QUOTE_WORDS
    answer = {"answer": "x", "mode": "documented", "citations": ["d#000"],
              "quotes": [{"text": paraphrase, "chunk_id": "d#000"}]}
    failures = check_quotes(answer, [chunk])
    assert failures, "a long paraphrase must not be rescued by trimming"
    assert answer.get("_auto_trimmed") is None


def test_long_quote_from_non_quotable_document_still_fails():
    chunk = {"chunk_id": "s#000", "text": SOURCE, "document_id": "s",
             "quotable": False, "verbatim": False}
    answer = {"answer": "x", "mode": "documented", "citations": ["s#000"],
              "quotes": [{"text": " ".join(SOURCE.split()[:31]), "chunk_id": "s#000"}]}
    failures = check_quotes(answer, [chunk])
    assert failures
    assert answer.get("_auto_trimmed") is None
