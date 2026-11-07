"""Step 2: clean, speaker-filter and chunk the raw transcripts. No LLM.

Everything here is deterministic. The point of this module is one specific
failure: a quote that passes a substring check against the corpus but was
never said by Lee Kuan Yew. That happens when the corpus contains words that
are not his, so the words that are not his are removed here, before anything
is indexed.

Three kinds of foreign text are removed:

  1. Archive furniture. Page numbers, "lky\\1965\\lky1017b.doc" path markers,
     "P a g e | 4", rules of dashes. These are not speech at all.
  2. Other speakers. The NYT interview is a dialogue, so only "Mr Lee:" turns
     survive. The 1965 parliament record carries "Mr. Speaker:" interjections.
  3. Quoted material. In the parliament speech Lee reads out Barisan Sosialis,
     PLEBIAN EXPRESS, Malayan Monitor and C.P. Fitzgerald at length in order to
     rebut it. Measured on the raw file, 18 double-quoted spans over 40 words
     hold 4,607 of its 9,595 words. Indexed unfiltered, roughly half that
     document would be attributable to Lee and wrong.

Press-release furniture (media contact blocks) is stripped, and the build then
fails if any "@", "Tel:" or "DID:" survives anywhere in the corpus.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .textutil import sha256_of, word_count

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "manifest.json"
SOURCES = ROOT / "config" / "sources.yaml"
CHUNKS_JSONL = PROCESSED / "chunks.jsonl"

TARGET_WORDS = 250
OVERLAP_WORDS = 40
THIRD_PARTY_QUOTE_WORDS = 40

REPLACEMENT_CHAR = "�"


class ProcessError(RuntimeError):
    """Raised when the corpus cannot be cleaned safely."""


# --- line-level furniture -----------------------------------------------------

_DROP_LINE = [
    re.compile(r"^\s*\d+\s*$"),                                  # page number
    re.compile(r"^\s*lky[\\/]\d{4}[\\/]lky\w*\.doc\s*$", re.I),  # archive path
    re.compile(r"^\s*lky\d{6,8}[a-z]?\s*$", re.I),               # lky19991228c
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.I),       # Page 3 of 9
    re.compile(r"^\s*p\s*a\s*g\s*e\s*\|?\s*\d+\s*$", re.I),      # P a g e | 4
    re.compile(r"^\s*[-=_*]{3,}\s*$"),                           # ---- or ====
]

# Press-release headers and footers. These are the publisher's words, not the
# speaker's, and they carry contact details that must never enter the corpus.
_DROP_PRESS = [
    re.compile(r"singapore government press release", re.I),
    re.compile(r"media relations division", re.I),
    re.compile(r"\bMITA Building\b", re.I),
    re.compile(r"ministry of information, communications and the arts", re.I),
    re.compile(r"^\s*tel\s*:", re.I),
    re.compile(r"^\s*DID\s*:", re.I),
    re.compile(r"@"),
    re.compile(r"^\s*embar[og]{2,4}ed\b", re.I),                 # EMBAROGED (sic)
    re.compile(r"^\s*please check against delivery\s*$", re.I),
]

# The build fails if any of these survive cleaning.
_FORBIDDEN_IN_CORPUS = [("@", "@"), ("Tel:", "tel:"), ("DID:", "did:")]


def drop_furniture(text: str) -> tuple[str, int]:
    """Remove archive and press-release lines. Returns (text, lines_dropped)."""
    kept, dropped = [], 0
    for line in text.splitlines():
        if any(p.search(line) for p in _DROP_LINE) or \
           any(p.search(line) for p in _DROP_PRESS):
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept), dropped


def normalise_quotes(text: str) -> str:
    """Curly and odd quotes to plain ' and ".

    The archive transcripts mix at least six quote characters. A substring
    check against a chunk fails purely on the character if this is not done.
    """
    for bad in "‘’‚‛′":
        text = text.replace(bad, "'")
    for bad in "“”„‟″":
        text = text.replace(bad, '"')
    return text


def assert_no_replacement_chars(document_id: str, text: str) -> None:
    n = text.count(REPLACEMENT_CHAR)
    if n:
        contexts = []
        for m in re.finditer(re.escape(REPLACEMENT_CHAR), text):
            lo, hi = max(0, m.start() - 30), m.end() + 30
            contexts.append(repr(text[lo:hi]))
            if len(contexts) >= 3:
                break
        raise ProcessError(
            f"{document_id}: {n} U+FFFD replacement character(s) in the text, "
            f"which means the source encoding was lost. Examples: "
            f"{'; '.join(contexts)}"
        )


# --- document headings --------------------------------------------------------

# The archive prints a heading block before the speech: who spoke, to whom,
# where and when, in capitals, often across several lines.
_HEADING_LEAD = re.compile(
    r"^\s*(TRANSCRIPT OF|SUMMARY OF|SPEECH (BY|MADE BY)|ADDRESS BY|REMARKS BY"
    r"|TRANSLATION OF|PRIME MINISTER'?S SPEECH|INTERVIEW WITH"
    r"|EXCERPTS? (OF|FROM))\b"
)


def _is_shouted(line: str, threshold: float = 0.7) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= threshold


def strip_heading_block(text: str, scan_lines: int = 12) -> tuple[str, int]:
    """Remove the capitalised heading block at the top of a transcript.

    The heading is the archive describing the record, not Lee speaking. Left
    in, it is indexed as if it were speech: before this, a query about English
    as a working language returned the line "SPEECH BY MINISTER MENTOR LEE
    KUAN YEW AT THE LAUNCHING OF THE BOOK..." as its fifth hit, and a quote
    gate that only checks substrings would have allowed it to be quoted.

    Only the top of the document is scanned, and only consecutive shouted
    lines are taken, so capitalised words inside the body are untouched. The
    title is kept in data/manifest.json as metadata.
    """
    lines = text.splitlines()
    removed = 0
    i = 0
    # skip leading blanks
    while i < len(lines) and not lines[i].strip():
        i += 1
    start = i
    in_heading = False
    while i < len(lines) and i < start + scan_lines:
        stripped = lines[i].strip()
        if not stripped:
            if in_heading:
                i += 1
                removed += 1
                continue
            break
        if _HEADING_LEAD.match(stripped) and _is_shouted(stripped):
            in_heading = True
        elif in_heading and _is_shouted(stripped):
            pass                      # continuation line of the same heading
        elif in_heading and re.fullmatch(r"[\d\s.,;:'\"()/-]+", stripped):
            pass                      # trailing date fragment, e.g. "1965."
        else:
            break
        i += 1
        removed += 1
    if not in_heading:
        return text, 0
    return "\n".join(lines[:start] + lines[i:]), removed


def looks_like_heading(text: str) -> bool:
    """Whether a span of text is an archive heading rather than speech.

    Used by the quote gate: a quote matching a heading is not something the
    speaker said.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    return bool(_HEADING_LEAD.match(stripped)) and _is_shouted(stripped, 0.6)


# --- paragraphs ---------------------------------------------------------------

def join_wrapped_lines(text: str) -> str:
    """Join hard-wrapped lines into paragraphs, keeping real paragraph breaks.

    A blank line is a paragraph break. Within a block the archives wrap at a
    fixed column, mid-sentence, so the lines are joined with a space.
    """
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
        # a word split across the wrap, e.g. "English-\nspeaking"
        joined = re.sub(r"(\w)- (\w)", r"\1-\2", joined)
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


# --- speaker filtering --------------------------------------------------------

_TURN_RE = re.compile(r"^\s*([A-Z][A-Za-z.\s]{0,24}?)\s*:\s*(.*)$")


def keep_only_speaker(text: str, speaker: str) -> tuple[str, int]:
    """Keep only the turns of one speaker in a dialogue.

    A turn runs until the next recognised "Name:" label, so a multi-paragraph
    answer stays whole. Returns (text, words_removed).
    """
    before = word_count(text)
    kept: list[str] = []
    keeping = False
    for line in text.splitlines():
        m = _TURN_RE.match(line)
        if m:
            label = m.group(1).strip().rstrip(".")
            keeping = label.casefold() == speaker.casefold()
            if keeping and m.group(2).strip():
                kept.append(m.group(2).strip())
            continue
        if keeping:
            kept.append(line)
    out = "\n".join(kept)
    return out, before - word_count(out)


def drop_speaker_turns(text: str, speakers: list[str]) -> tuple[str, int]:
    """Drop the turns of the named speakers, keeping everything else."""
    before = word_count(text)
    targets = {s.casefold().rstrip(".") for s in speakers}
    kept: list[str] = []
    dropping = False
    for line in text.splitlines():
        m = _TURN_RE.match(line)
        if m:
            label = m.group(1).strip().rstrip(".").casefold()
            if label in targets:
                dropping = True
                continue
            dropping = False
            rest = m.group(2).strip()
            kept.append(rest)
            continue
        if not dropping:
            kept.append(line)
    out = "\n".join(kept)
    return out, before - word_count(out)


def strip_quoted_blocks(text: str) -> tuple[str, int, int]:
    """Remove multi-line quoted extracts. Runs on LINES, before they are joined.

    Returns (text, blocks_removed, words_removed).

    Two earlier attempts failed on the real file and are worth recording:

    - Pairing quote characters left to right across the whole text. Quoting in
      the archive transcript is unbalanced, so pairs drift: the rule deleted
      the text BETWEEN two separate short quotations, which is Lee speaking,
      and glued the remainder together ("phoneyphoney", "Third ChinaSingapore").
    - Dropping whole paragraphs that are mostly quoted. The 1965 parliament
      file has 786 lines and only 8 blank ones, so joining produces 9
      paragraphs, one of 2,576 words. At that granularity the rule deleted an
      828-word paragraph of Lee's own words.

    Working line by line keeps the structure the archive actually has: a
    quoted extract starts a line with a quote character and closes on a later
    line. This removes no text from the middle of a line of Lee's speech and
    introduces no glued words.
    """
    before = word_count(text)
    kept: list[str] = []
    blocks = 0
    in_quote = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_quote:
            if stripped.count('"') % 2 == 1:
                in_quote = False
            continue
        if stripped.count('"') % 2 == 1:
            # a quotation opens here and does not close on this line
            idx = stripped.rfind('"')
            prefix = stripped[:idx].strip()
            if prefix:
                kept.append(prefix)
            in_quote = True
            blocks += 1
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    return cleaned, blocks, before - word_count(cleaned)


def drop_stub_paragraphs(text: str, min_words: int = 4) -> str:
    """Drop fragments left behind by quotation removal.

    A paragraph of three words or fewer after stripping is almost always the
    dangling lead-in to a quotation that is no longer there ("He said:").
    """
    kept = [p for p in text.split("\n\n")
            if word_count(p) >= min_words and re.search(r"[A-Za-z]", p)]
    return "\n\n".join(kept)


# --- chunking -----------------------------------------------------------------

# Splitting on every full stop cuts "Mr. Chairman" in half, which produced
# chunks beginning "Mr." on its own once the heading block was removed.
_ABBREVIATIONS = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Rt", "Hon",
    "No", "Nos", "vs", "etc", "Co", "Ltd", "Inc", "Capt", "Gen", "Col",
    "Sgt", "Lt", "Maj", "Govt", "Dept", "Univ", "Tan", "Dato", "Encik",
)
_ABBREV_RE = "|".join(_ABBREVIATIONS)


def _sentences(paragraph: str) -> list[str]:
    text = paragraph.strip()
    # protect "Mr." style abbreviations and single initials from the split
    protected = re.sub(rf"\b({_ABBREV_RE})\.",
                       lambda m: m.group(1) + "\x00", text)
    protected = re.sub(r"\b([A-Z])\.",
                       lambda m: m.group(1) + "\x01", protected)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    out = []
    for part in parts:
        restored = part.replace("\x00", ".").replace("\x01", ".").strip()
        if restored:
            out.append(restored)
    return out


def chunk_text(text: str, target: int = TARGET_WORDS,
               overlap: int = OVERLAP_WORDS) -> list[str]:
    """Chunk to roughly `target` words with roughly `overlap` words carried
    over, splitting only on paragraph and sentence boundaries."""
    units: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if word_count(paragraph) <= target:
            units.append(paragraph)
        else:
            units.extend(_sentences(paragraph))

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        n = word_count(unit)
        if current and count + n > target:
            chunks.append("\n\n".join(current))
            carried: list[str] = []
            carried_words = 0
            for prev in reversed(current):
                if carried_words >= overlap:
                    break
                carried.insert(0, prev)
                carried_words += word_count(prev)
            current = carried
            count = carried_words
        current.append(unit)
        count += n
    if current:
        tail = "\n\n".join(current)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks


# --- per-document pipeline ----------------------------------------------------

@dataclass
class DocResult:
    document_id: str
    title: str
    date: str
    type: str
    translated: bool
    verbatim: bool
    third_party_quotes: bool
    url: None
    word_count_raw: int
    word_count_processed: int
    n_chunks: int
    sha256_raw: str
    answer_use: bool = True
    n_chunks_built: int = 0
    lines_dropped: int = 0
    excluded_spans_removed: int = 0
    words_removed_excluded: int = 0
    words_removed_speaker: int = 0
    words_removed_quotations: int = 0
    quotation_spans_removed: int = 0
    chunks: list[dict] = field(default_factory=list)

    def manifest_entry(self) -> dict:
        """Manifest row. Carries no corpus text by design."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "date": self.date,
            "type": self.type,
            "translated": self.translated,
            "verbatim": self.verbatim,
            "third_party_quotes": self.third_party_quotes,
            "word_count_raw": self.word_count_raw,
            "word_count_processed": self.word_count_processed,
            "n_chunks": self.n_chunks,
            "n_chunks_built": self.n_chunks_built,
            "answer_use": self.answer_use,
            "sha256_raw": self.sha256_raw,
            "url": self.url,
            "lines_dropped": self.lines_dropped,
            "words_removed_speaker": self.words_removed_speaker,
            "words_removed_quotations": self.words_removed_quotations,
            "quotation_spans_removed": self.quotation_spans_removed,
            "excluded_spans_removed": self.excluded_spans_removed,
            "words_removed_excluded": self.words_removed_excluded,
        }


def _flexible(marker: str) -> re.Pattern:
    """Marker matcher tolerant of the line wrapping in the raw transcripts.

    The raw files are hard-wrapped, so a marker phrase routinely straddles a
    newline. Matching on literal text would miss exactly the spans that matter.
    """
    return re.compile(r"\s+".join(re.escape(w) for w in marker.split()), re.I)


def apply_exclude_spans(text: str, spans: list[dict]) -> tuple[str, int, int]:
    """Cut hand-identified third-party blocks out, begin marker to end marker.

    These are passages that are verbatim in the archive and verbatim in the
    chunk, and still not Lee's words: a letter somebody sent him, a magazine
    article he read aloud. The quote gate cannot tell them apart from his own
    speech, because by every mechanical test they *are* part of the record. So
    they are removed by hand-written markers in config/sources.yaml, and the
    tests assert they stay out.

    Markers are matched case-insensitively with flexible whitespace. A span
    whose begin or end marker is not found raises, rather than silently leaving
    third-party text in the corpus.
    """
    removed_spans = removed_words = 0
    for span in spans or []:
        begin, end = span["begin"], span["end"]
        mb = _flexible(begin).search(text)
        if mb is None:
            raise ProcessError(f"exclude_spans: begin marker not found: {begin!r}")
        me = _flexible(end).search(text, mb.end())
        if me is None:
            raise ProcessError(f"exclude_spans: end marker not found after begin: {end!r}")
        cut = text[mb.start():me.end()]
        removed_words += word_count(cut)
        removed_spans += 1
        text = text[:mb.start()] + " " + text[me.end():]
    return text, removed_spans, removed_words


def process_document(entry: dict) -> DocResult:
    document_id = entry["document_id"]
    path = RAW / f"{document_id}.txt"
    if not path.exists():
        raise ProcessError(f"{path} is missing")

    raw = path.read_text(encoding="utf-8", errors="strict")
    sha = sha256_of(raw)
    words_raw = word_count(raw)

    text = normalise_quotes(raw)
    assert_no_replacement_chars(document_id, text)
    text, lines_dropped = drop_furniture(text)
    text, heading_lines = strip_heading_block(text)
    lines_dropped += heading_lines

    answer_use = bool(entry.get("answer_use", True))
    excluded_spans, words_excluded = 0, 0
    if entry.get("exclude_spans"):
        text, excluded_spans, words_excluded = apply_exclude_spans(
            text, entry["exclude_spans"])

    words_speaker = 0
    if entry.get("speaker_filter"):
        text, words_speaker = keep_only_speaker(text, entry["speaker_filter"])
    elif entry["type"] == "parliament":
        text, words_speaker = drop_speaker_turns(
            text, ["Mr. Speaker", "Mr Speaker"])

    # Quoted extracts are removed while the line structure still exists.
    spans = words_quotes = 0
    if entry.get("third_party_quotes"):
        text, spans, words_quotes = strip_quoted_blocks(text)

    text = join_wrapped_lines(text)
    if entry.get("third_party_quotes"):
        text = drop_stub_paragraphs(text)

    for label, needle in _FORBIDDEN_IN_CORPUS:
        if needle in text.casefold():
            line = next((ln for ln in text.splitlines()
                         if needle in ln.casefold()), "")
            raise ProcessError(
                f"{document_id}: {label!r} survived cleaning, which means a "
                f"press-release contact block is still in the corpus. "
                f"Offending line: {line[:120]!r}"
            )

    pieces = chunk_text(text)
    chunks = [
        {
            "chunk_id": f"{document_id}#{i:04d}",
            "document_id": document_id,
            "date": str(entry["date"]),
            "type": entry["type"],
            "translated": bool(entry["translated"]),
            "verbatim": bool(entry["verbatim"]),
            "third_party_quotes": bool(entry["third_party_quotes"]),
            # Derived, and the only flag the quote gate reads. See
            # quotable_chunks() for why third_party_quotes also disqualifies.
            "answer_use": answer_use,
            "quotable": bool(entry["verbatim"])
            and not bool(entry["third_party_quotes"]),
            "chunk_index": i,
            "word_count": word_count(piece),
            "text": piece,
        }
        for i, piece in enumerate(pieces)
    ]

    return DocResult(
        document_id=document_id,
        title=entry["title"],
        date=str(entry["date"]),
        type=entry["type"],
        translated=bool(entry["translated"]),
        verbatim=bool(entry["verbatim"]),
        third_party_quotes=bool(entry["third_party_quotes"]),
        url=entry["url"],
        word_count_raw=words_raw,
        word_count_processed=word_count(text),
        n_chunks=len(chunks) if answer_use else 0,
        n_chunks_built=len(chunks),
        answer_use=answer_use,
        sha256_raw=sha,
        lines_dropped=lines_dropped,
        excluded_spans_removed=excluded_spans,
        words_removed_excluded=words_excluded,
        words_removed_speaker=words_speaker,
        words_removed_quotations=words_quotes,
        quotation_spans_removed=spans,
        chunks=chunks,
    )


def load_sources() -> list[dict]:
    data = yaml.safe_load(SOURCES.read_text(encoding="utf-8"))
    return data["documents"]


def build() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    results = [process_document(e) for e in load_sources()]

    with CHUNKS_JSONL.open("w", encoding="utf-8") as fh:
        for r in results:
            if not r.answer_use:
                continue          # kept in the manifest, kept out of the index
            for chunk in r.chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    manifest = {
        "corpus": "Lee Kuan Yew speeches, interviews and parliament records",
        "note": "Transcript text is never stored in this file. "
                "data/raw/ and data/processed/ are gitignored.",
        "n_documents": len(results),
        "n_chunks": sum(r.n_chunks for r in results),
        "n_chunks_built": sum(r.n_chunks_built for r in results),
        "n_documents_not_answerable": sum(1 for r in results if not r.answer_use),
        "word_count_raw": sum(r.word_count_raw for r in results),
        "word_count_processed": sum(r.word_count_processed for r in results),
        "chunking": {"target_words": TARGET_WORDS,
                     "overlap_words": OVERLAP_WORDS},
        "documents": [r.manifest_entry() for r in results],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_chunks() -> list[dict]:
    if not CHUNKS_JSONL.exists():
        raise FileNotFoundError(
            "data/processed/chunks.jsonl missing — run `python -m src.process`")
    with CHUNKS_JSONL.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def quotable_chunks() -> list[dict]:
    """Chunks a direct quote may be taken from.

    Two things disqualify a document:

    verbatim: false
        A summary written by someone else. Retrievable and usable to answer,
        but quoting it would attribute another person's paraphrase to Lee.
        This is the 1970 River Valley record, headed "SUMMARY OF SPEECH".

    third_party_quotes: true
        The 1965 parliament record. The quoted extracts are stripped, but the
        stripping removes 70% of the document, more than the 48% measured as
        sitting inside long quoted spans, because the transcript's quoting is
        unbalanced and the line-level rule errs towards removing too much.
        What survives is very probably Lee, but "very probably" is not a basis
        for putting quotation marks around something, so the document is
        retrievable and not quotable.
    """
    return [c for c in load_chunks() if c["quotable"]]


def main() -> int:
    manifest = build()
    print(f"{manifest['n_documents']} documents -> {manifest['n_chunks']} chunks")
    print(f"words: {manifest['word_count_raw']} raw -> "
          f"{manifest['word_count_processed']} processed")
    print()
    print(f"  {'document_id':50} {'chunks':>6} {'raw':>7} {'proc':>7}  flags")
    for d in manifest["documents"]:
        flags = []
        if d["translated"]:
            flags.append("translated")
        if not d["verbatim"]:
            flags.append("NOT-verbatim")
        if d["third_party_quotes"]:
            flags.append("3rd-party")
        print(f"  {d['document_id'][:50]:50} {d['n_chunks']:>6} "
              f"{d['word_count_raw']:>7} {d['word_count_processed']:>7}  "
              f"{', '.join(flags)}")
    print()
    for d in manifest["documents"]:
        if d["words_removed_speaker"] or d["words_removed_quotations"]:
            print(f"  {d['document_id']}: "
                  f"{d['words_removed_speaker']} words removed as other "
                  f"speakers, {d['words_removed_quotations']} words in "
                  f"{d['quotation_spans_removed']} quoted spans")
    print()
    print("chunks -> data/processed/chunks.jsonl (gitignored)")
    print("manifest -> data/manifest.json (committed, no corpus text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
