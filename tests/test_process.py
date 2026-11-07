"""Corpus integrity tests. No API key, no network.

These exist to catch one failure: text in the index that Lee Kuan Yew did not
say. Each test names the document it guards and what would be wrong if it
failed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.process import (
    CHUNKS_JSONL,
    MANIFEST,
    REPLACEMENT_CHAR,
    chunk_text,
    drop_furniture,
    keep_only_speaker,
    load_chunks,
    normalise_quotes,
    quotable_chunks,
    strip_quoted_blocks,
)
from src.textutil import word_count

NYT = "lky_2010-09-01_interview-nyt-seth-mydans"
PARLIAMENT = "lky_1965-12-14_parliament-motion-of-thanks"
SUMMARY_DOC = "lky_1970-08-16_river-valley-national-day-SUMMARY"

pytestmark = pytest.mark.skipif(
    not CHUNKS_JSONL.exists(),
    reason="corpus not built; run `python -m src.process`",
)


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus_text(chunks):
    return "\n".join(c["text"] for c in chunks)


def _doc(chunks, document_id):
    return [c for c in chunks if c["document_id"] == document_id]


# --- the interviewer must be gone --------------------------------------------

def test_no_interviewer_turns_survive_in_the_nyt_interview(chunks):
    """Seth Mydans' questions are not Lee's words. The raw file has 73 "Q:"
    turns against 75 "Mr Lee:" turns, so an unfiltered index would be about a
    quarter other people."""
    offenders = [c["chunk_id"] for c in _doc(chunks, NYT)
                 if re.search(r"\bQ\s*:", c["text"])]
    assert offenders == [], f"interviewer turns survived in {offenders}"


def test_no_third_participant_survives_in_the_nyt_interview(chunks):
    offenders = [c["chunk_id"] for c in _doc(chunks, NYT)
                 if re.search(r"Mdm\s+Yeong", c["text"], re.I)]
    assert offenders == [], f"Mdm Yeong's turns survived in {offenders}"


def test_speaker_filter_keeps_the_answers(chunks):
    """Guard against the filter being so strict it empties the document."""
    doc = _doc(chunks, NYT)
    assert len(doc) > 10
    assert sum(c["word_count"] for c in doc) > 3000


def test_keep_only_speaker_drops_other_turns():
    text = ("Mr Lee: I said this.\nand it continued.\n"
            "Q: Did you?\nMdm Yeong: One more.\n"
            "Mr Lee: And I said that.")
    out, removed = keep_only_speaker(text, "Mr Lee")
    assert "Did you?" not in out
    assert "One more." not in out
    assert "I said this." in out and "and it continued." in out
    assert "And I said that." in out
    assert removed > 0


# --- parliament: other speakers and quoted extracts ---------------------------

def test_no_speaker_turn_labels_survive_in_parliament(chunks):
    """A line beginning "Mr. Speaker:" is the Speaker talking, not Lee.

    Lee's own vocative "Mr. Speaker, Sir, ..." is his speech and is kept on
    purpose, so this looks for the turn label, not for the name.
    """
    offenders = [c["chunk_id"] for c in _doc(chunks, PARLIAMENT)
                 if re.search(r"(?:^|\n)\s*Mr\.?\s+Speaker\s*:", c["text"])]
    assert offenders == [], f"Speaker turns survived in {offenders}"


def test_lees_vocative_to_the_speaker_is_kept():
    """The opposite failure: stripping every "Mr. Speaker" would delete Lee's
    own words.

    Run 2 took the parliament record out of the answering index (answer_use:
    false), so this is asserted against the document as processed rather than
    against the chunks, which no longer contain it.
    """
    from src.process import PROCESSED
    text = (PROCESSED / f"{PARLIAMENT}.txt").read_text(encoding="utf-8")
    assert "Mr. Speaker, Sir" in text


def test_no_long_quoted_span_survives_in_parliament(chunks):
    """Lee reads Barisan Sosialis, PLEBIAN EXPRESS and Malayan Monitor into
    the record at length. Those words are not his."""
    offenders = []
    for c in _doc(chunks, PARLIAMENT):
        for m in re.finditer(r'"([^"]+)"', c["text"]):
            if word_count(m.group(1)) > 40:
                offenders.append((c["chunk_id"], word_count(m.group(1))))
    assert offenders == [], f"long quoted spans survived: {offenders}"


def test_strip_quoted_blocks_removes_a_multi_line_extract():
    text = ('He said this himself.\n'
            '"This is the start of a long extract\n'
            'that runs across several lines\n'
            'and finally closes here."\n'
            'And Lee resumes.')
    out, blocks, removed = strip_quoted_blocks(text)
    assert blocks == 1
    assert removed > 0
    assert "long extract" not in out
    assert "He said this himself." in out
    assert "And Lee resumes." in out


def test_strip_quoted_blocks_introduces_no_glued_words():
    """The rule this replaced deleted text between two short quotations and
    produced "phoneyphoney"."""
    text = 'He used "phoney" then later used "phoney" again in the debate.'
    out, blocks, removed = strip_quoted_blocks(text)
    assert blocks == 0 and removed == 0
    assert out == text


def test_short_inline_quotes_are_kept():
    text = 'They called it a "phoney merger" and I disagreed.'
    out, blocks, _ = strip_quoted_blocks(text)
    assert blocks == 0
    assert "phoney merger" in out


# --- archive furniture --------------------------------------------------------

PAGE_MARKER_PATTERNS = [
    (r"(?m)^\s*\d+\s*$", "bare page number"),
    (r"(?i)lky[\\/]\d{4}[\\/]lky\w*\.doc", "archive path marker"),
    (r"(?im)^\s*lky\d{6,8}[a-z]?\s*$", "lky document id"),
    (r"(?im)^\s*page\s+\d+(\s+of\s+\d+)?\s*$", "Page N"),
    (r"(?im)^\s*p\s+a\s+g\s+e\s*\|?\s*\d+\s*$", "P a g e | N"),
    (r"(?m)^\s*[-=_*]{3,}\s*$", "rule of dashes"),
]


@pytest.mark.parametrize("pattern,label", PAGE_MARKER_PATTERNS)
def test_no_page_markers_anywhere(corpus_text, pattern, label):
    found = re.findall(pattern, corpus_text)
    assert found == [], f"{label} survived: {found[:5]}"


def test_no_replacement_characters(corpus_text):
    """U+FFFD means the source encoding was lost and the text is unreliable."""
    assert REPLACEMENT_CHAR not in corpus_text


@pytest.mark.parametrize("needle", ["@", "Tel:", "DID:"])
def test_no_press_release_contact_details(corpus_text, needle):
    """The 2005 KL address arrived with a media-contact header."""
    lowered = corpus_text.casefold()
    assert needle.casefold() not in lowered, (
        f"{needle!r} survived, so a press-release block is in the corpus")


def test_drop_furniture_removes_markers_but_keeps_speech():
    text = ("1\n"
            "lky\\1965\\lky1017b.doc\n"
            "P a g e | 4\n"
            "-----\n"
            "This is the speech.\n"
            "Tel: 6837-9666\n"
            "It continues here.")
    out, dropped = drop_furniture(text)
    assert dropped == 5
    assert "This is the speech." in out
    assert "It continues here." in out
    assert "lky1017b" not in out and "6837" not in out


def test_quote_normalisation_handles_every_variant():
    raw = "‘a’ ‚b‛ “c” „d‟"
    out = normalise_quotes(raw)
    assert out == "'a' 'b' \"c\" \"d\""
    for ch in "‘’‚‛“”„‟":
        assert ch not in out


# --- manifest and chunk store agree ------------------------------------------

def test_manifest_chunk_total_matches_the_jsonl(manifest, chunks):
    assert manifest["n_chunks"] == len(chunks)


def test_manifest_per_document_counts_match(manifest, chunks):
    from collections import Counter
    actual = Counter(c["document_id"] for c in chunks)
    for doc in manifest["documents"]:
        assert doc["n_chunks"] == actual[doc["document_id"]], doc["document_id"]


def test_manifest_covers_every_document(manifest):
    assert manifest["n_documents"] == 17
    assert len(manifest["documents"]) == 17


def test_manifest_contains_no_corpus_text(manifest):
    """The manifest is committed; the corpus is not."""
    blob = json.dumps(manifest)
    assert "text" not in {k for d in manifest["documents"] for k in d}
    for chunk_field in ("text", "chunks"):
        assert f'"{chunk_field}":' not in blob


def test_every_manifest_url_is_on_nas_or_null(manifest):
    """URLs are filled in by hand. Nothing here invents one.

    Run 2 carries the URLs the author entered in config/sources.yaml. The test
    no longer requires null, but still refuses a URL off nas.gov.sg, which is
    the only place this corpus came from.
    """
    for d in manifest["documents"]:
        url = d["url"]
        assert url is None or url.startswith("https://www.nas.gov.sg/"), d["document_id"]


def test_every_document_has_a_raw_sha256(manifest):
    for d in manifest["documents"]:
        assert re.fullmatch(r"[0-9a-f]{64}", d["sha256_raw"]), d["document_id"]


def test_chunk_ids_are_unique_and_stable(chunks):
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c["chunk_id"] == f"{c['document_id']}#{c['chunk_index']:04d}"


def test_no_chunk_spans_two_documents(chunks):
    for c in chunks:
        assert c["chunk_id"].startswith(c["document_id"])


# --- the quote gate -----------------------------------------------------------

def test_summary_document_is_not_offered_to_the_quote_gate():
    """The 1970 River Valley record is headed "SUMMARY OF SPEECH", so the
    words are the summariser's, not Lee's."""
    ids = {c["document_id"] for c in quotable_chunks()}
    assert SUMMARY_DOC not in ids


def test_parliament_document_is_not_offered_to_the_quote_gate():
    """Quote stripping removes 70% of it, more than the 48% measured inside
    long quoted spans, so what survives cannot be certified as Lee's."""
    ids = {c["document_id"] for c in quotable_chunks()}
    assert PARLIAMENT not in ids


def test_quotable_chunks_are_a_strict_subset(chunks):
    quotable = quotable_chunks()
    assert 0 < len(quotable) < len(chunks)


def test_every_quotable_chunk_is_verbatim_and_first_party():
    for c in quotable_chunks():
        assert c["verbatim"] is True
        assert c["third_party_quotes"] is False


def test_non_quotable_summary_is_still_retrievable(chunks):
    """Cite-only, not excluded from the index.

    The 1970 summary stays retrievable and unquotable. The parliament record is
    a separate case in run 2: it is out of the index altogether, which
    tests/test_run2.py asserts.
    """
    ids = {c["document_id"] for c in chunks}
    assert SUMMARY_DOC in ids
    assert PARLIAMENT not in ids


# --- chunking -----------------------------------------------------------------

def test_chunks_are_near_the_target_size(chunks):
    sizes = [c["word_count"] for c in chunks]
    assert max(sizes) <= 320, f"largest chunk {max(sizes)} words"
    median = sorted(sizes)[len(sizes) // 2]
    assert 150 <= median <= 300, f"median {median}"


def test_chunk_text_overlaps_consecutive_chunks():
    paragraphs = [f"Sentence number {i} about housing policy in Singapore."
                  for i in range(120)]
    pieces = chunk_text("\n\n".join(paragraphs))
    assert len(pieces) > 1
    tail = pieces[0].split("\n\n")[-1]
    assert tail in pieces[1], "consecutive chunks share no text"


def test_chunk_text_never_returns_empty_pieces():
    assert chunk_text("") == [] or all(p.strip() for p in chunk_text(""))


def test_translated_documents_are_flagged(chunks):
    translated = {c["document_id"] for c in chunks if c["translated"]}
    assert translated == {
        "lky_1965-10-17_leong-khay-huay-kuan-TRANSLATION",
        "lky_1965-11-14_kong-chow-huay-kuan-TRANSLATION",
    }


# --- document headings are not speech ----------------------------------------
# The archive prints "SPEECH BY MINISTER MENTOR LEE KUAN YEW AT ..." above each
# transcript. Indexed as body text it was retrievable: before this was fixed it
# came back as the 5th hit for a question about English as a working language,
# and a substring-only quote gate would have allowed it to be quoted.

HEADING_LEADS = ["TRANSCRIPT OF", "SUMMARY OF", "SPEECH BY", "SPEECH MADE BY",
                 "ADDRESS BY", "REMARKS BY", "TRANSLATION OF"]


@pytest.mark.parametrize("lead", HEADING_LEADS)
def test_no_heading_lead_phrase_survives_anywhere(corpus_text, lead):
    assert lead not in corpus_text, f"heading phrase {lead!r} is in the corpus"


def test_no_chunk_starts_with_a_heading(chunks):
    from src.process import looks_like_heading
    offenders = [c["chunk_id"] for c in chunks if looks_like_heading(c["text"])]
    assert offenders == [], f"chunks starting with a heading: {offenders}"


def test_no_chunk_contains_a_manifest_title(chunks, manifest):
    """A title in the body text could be quoted back as if spoken."""
    titles = [d["title"].casefold() for d in manifest["documents"]]
    offenders = [(c["chunk_id"], t) for c in chunks for t in titles
                 if t in c["text"].casefold()]
    assert offenders == [], f"manifest titles found in chunk text: {offenders[:3]}"


def test_strip_heading_block_removes_a_multi_line_header():
    from src.process import strip_heading_block
    text = ("SPEECH BY THE PRIME MINISTER, MR. LEE KUAN YEW,\n"
            "AT THE OPENING CEREMONY OF EVERTON PARK HOUSING\n"
            "ESTATE ON 8TH NOVEMBER, 1965.\n"
            "Mr. Chairman and Friends,\n"
            "It is always a problem.")
    out, removed = strip_heading_block(text)
    assert removed == 3
    assert "EVERTON PARK HOUSING" not in out
    assert "Mr. Chairman and Friends," in out
    assert "It is always a problem." in out


def test_strip_heading_block_swallows_a_trailing_date_line():
    from src.process import strip_heading_block
    text = ("TRANSLATION OF PRIME MINISTER'S SPEECH IN HOKKIEN\n"
            "GIVEN AT THE LEONG KHAY HUAY KUAN ON 17TH OCTOBER,\n"
            "1965.\n"
            "Mr. Chairman, Brothers & Friends,")
    out, removed = strip_heading_block(text)
    assert removed == 3
    assert out.strip().startswith("Mr. Chairman")


def test_strip_heading_block_leaves_capitalised_body_text_alone():
    """Only the top of the document is scanned, and only a lead phrase starts
    a heading, so shouted words inside the speech survive."""
    from src.process import strip_heading_block
    text = ("I want to talk about housing.\n"
            "The PLEBIAN EXPRESS said otherwise.\n"
            "THAT IS THEIR VIEW, not mine.")
    out, removed = strip_heading_block(text)
    assert removed == 0
    assert out == text


def test_looks_like_heading_distinguishes_headings_from_speech():
    from src.process import looks_like_heading
    assert looks_like_heading(
        "SPEECH BY SENIOR MINISTER LEE KUAN YEW AT THE DINNER")
    assert looks_like_heading(
        "TRANSCRIPT OF SPEECH BY THE PRIME MINISTER, MR. LEE KUAN YEW")
    assert not looks_like_heading("I am happy to launch the new Authority.")
    assert not looks_like_heading("THAT IS THEIR VIEW, not mine.")
    assert not looks_like_heading("")


# --- abbreviations are not sentence ends -------------------------------------

def test_abbreviations_do_not_split_a_sentence():
    """Splitting on every full stop left chunks beginning "Mr." alone."""
    from src.process import _sentences
    out = _sentences("Mr. Chairman and Friends, I am glad. Dr. Lee agreed.")
    assert out[0].startswith("Mr. Chairman")
    assert not any(s.strip() in {"Mr.", "Dr."} for s in out)


def test_no_chunk_is_an_orphaned_abbreviation(chunks):
    offenders = [c["chunk_id"] for c in chunks
                 if re.search(r"(^|\n\n)(Mr|Mrs|Dr|Ms|Prof)\.\s*(\n|$)",
                              c["text"])]
    assert offenders == [], f"orphaned abbreviation fragments: {offenders}"
