"""Step 2: chunk the cleaned speeches by paragraph.

Target 300-400 words per chunk with one paragraph of overlap, so a point made
across a paragraph boundary is retrievable from either side. Paragraphs are the
unit because a speech argues in paragraphs; splitting on a fixed token count
cuts sentences in half and makes verbatim quoting unreliable.

chunk_id is "<document_id>#<nnn>" and is stable for a given cleaned text, so a
citation in an eval result still means the same thing after a rebuild. It stops
being stable if the source text changes, which is what the manifest sha256 is
there to reveal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .textutil import word_count

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "manifest.json"
CHUNKS = PROCESSED / "chunks.json"

TARGET_MIN = 300
TARGET_MAX = 400


def _split_long_paragraph(paragraph: str, limit: int = TARGET_MAX) -> list[str]:
    """A paragraph over the limit is split on sentence ends, never mid-sentence."""
    if word_count(paragraph) <= limit:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, current, count = [], [], 0
    for sentence in sentences:
        n = word_count(sentence)
        if current and count + n > limit:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(text: str) -> list[str]:
    paragraphs: list[str] = []
    for raw in text.split("\n\n"):
        raw = raw.strip()
        if raw:
            paragraphs.extend(_split_long_paragraph(raw))

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        n = word_count(paragraph)
        # Close the chunk before a paragraph that would push it past the max,
        # but only once it is already big enough to stand on its own.
        if current and count >= TARGET_MIN and count + n > TARGET_MAX:
            chunks.append("\n\n".join(current))
            current = [current[-1]]            # one paragraph of overlap
            count = word_count(current[0])
            continue
        current.append(paragraph)
        count += n
        index += 1
        if count >= TARGET_MAX:
            chunks.append("\n\n".join(current))
            current = [current[-1]]
            count = word_count(current[0])
    # The tail is kept only if it carries something the previous chunk did not.
    if current and (not chunks or len(current) > 1):
        chunks.append("\n\n".join(current))
    return chunks


def build() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = []
    for doc in manifest["documents"]:
        path = PROCESSED / f"{doc['document_id']}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `python -m src.collect` first")
        pieces = chunk_text(path.read_text(encoding="utf-8"))
        doc["chunk_count"] = len(pieces)
        for seq, piece in enumerate(pieces, start=1):
            records.append({
                "chunk_id": f"{doc['document_id']}#{seq:03d}",
                "document_id": doc["document_id"],
                "title": doc["title"],
                "date": doc["date"],
                "source_url": doc["source_url"],
                "topic": doc["topic"],
                "sequence": seq,
                "word_count": word_count(piece),
                "text": piece,
            })
    CHUNKS.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"documents": len(manifest["documents"]), "chunks": len(records)}


def load_chunks() -> list[dict]:
    if not CHUNKS.exists():
        raise FileNotFoundError("data/processed/chunks.json missing — run `python -m src.process`")
    return json.loads(CHUNKS.read_text(encoding="utf-8"))


def main() -> int:
    stats = build()
    records = load_chunks()
    sizes = sorted(r["word_count"] for r in records)
    print(f"{stats['documents']} documents -> {stats['chunks']} chunks")
    if sizes:
        median = sizes[len(sizes) // 2]
        print(f"words per chunk: min {sizes[0]}, median {median}, max {sizes[-1]}")
    print(f"chunks -> data/processed/chunks.json (gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
