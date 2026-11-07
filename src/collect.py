"""Step 1: ingest and verify the corpus. This script does NOT crawl.

nas.gov.sg is served behind AWS WAF Bot Control. Every scripted request, the
request for robots.txt included, comes back as an empty HTTP 202 carrying
`x-amzn-waf-action: challenge` (evidence in docs/collection-blocked.md).
Defeating that challenge, or reading the site while its robots policy is
unknowable, is out of bounds for this project. So a human opens each record in
a normal browser and saves the transcript to data/raw/<document_id>.txt, and
this script verifies what they saved.

What it verifies, and refuses to proceed without:
  - every sources.yaml field is filled, with no placeholder left behind
  - document_id is unique and matches its file on disk
  - source_url is on nas.gov.sg
  - the speaker is Lee Kuan Yew, asserted in the metadata AND corroborated by
    the text itself
  - the text is long enough to be a speech rather than a stub

Outputs data/processed/<document_id>.txt (gitignored) and data/manifest.json
(committed: metadata and hashes only, never the text).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import yaml

from .textutil import normalize_ws, sha256_of, word_count

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "config" / "sources.yaml"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "manifest.json"

REQUIRED = ("document_id", "title", "date", "source_url", "speaker", "topic")
TOPICS = {
    "economy", "education", "leadership", "foreign_policy", "language_policy",
    "national_service", "housing", "governance", "society",
}
MIN_WORDS = 300
SPEAKER = "lee kuan yew"

# Boilerplate the archives' own transcripts carry on every page.
_NOISE = [
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.I),
    re.compile(r"^\s*-+\s*\d+\s*-+\s*$"),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*lky/\d{4}/\S*\s*$", re.I),
    re.compile(r"^\s*(national archives of singapore|archives online)\s*$", re.I),
    re.compile(r"^\s*(all rights reserved|copyright\b.*)$", re.I),
]


class Problem(Exception):
    pass


def clean(text: str) -> str:
    """Drop page furniture, rejoin hyphenated line breaks, keep paragraphs.

    Paragraph structure is load-bearing: process.py chunks on it. So blank
    lines survive and single newlines inside a paragraph are folded away.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    kept = [ln for ln in text.split("\n") if not any(p.match(ln) for p in _NOISE)]
    text = "\n".join(kept)
    # A word split across a line break by the scanner: "govern-\nment".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    paragraphs = [normalize_ws(p) for p in re.split(r"\n\s*\n", text)]
    return "\n\n".join(p for p in paragraphs if p)


def _check_entry(entry: dict, seen: set[str]) -> None:
    for field in REQUIRED:
        value = str(entry.get(field, "") or "").strip()
        if not value:
            raise Problem(f"field '{field}' is empty — fill it in config/sources.yaml")
    doc_id = entry["document_id"].strip()
    if not re.fullmatch(r"\d{8}-[a-z0-9-]+", doc_id):
        raise Problem(f"document_id '{doc_id}' must look like 19650809-independence-press-conference")
    if doc_id in seen:
        raise Problem(f"duplicate document_id '{doc_id}'")
    seen.add(doc_id)
    if "nas.gov.sg" not in entry["source_url"]:
        raise Problem(f"source_url must be on nas.gov.sg, got '{entry['source_url']}'")
    if SPEAKER not in entry["speaker"].casefold():
        raise Problem(f"speaker must be Lee Kuan Yew, got '{entry['speaker']}'")
    if entry["topic"] not in TOPICS:
        raise Problem(f"topic '{entry['topic']}' not in {sorted(TOPICS)}")
    if not re.fullmatch(r"\d{4}(-\d{2}-\d{2})?", str(entry["date"]).strip()):
        raise Problem(f"date '{entry['date']}' must be YYYY-MM-DD or YYYY")


def ingest(strict: bool = True) -> dict:
    if not SOURCES.exists():
        raise Problem("config/sources.yaml is missing")
    config = yaml.safe_load(SOURCES.read_text(encoding="utf-8")) or {}
    speeches = config.get("speeches") or []
    speeches = [s for s in speeches if any(str(v or "").strip() for v in s.values())]
    if not speeches:
        raise Problem("config/sources.yaml has no filled-in speeches yet")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    documents, failures, seen = [], [], set()

    for entry in speeches:
        try:
            _check_entry(entry, seen)
            doc_id = entry["document_id"].strip()
            raw_path = RAW / f"{doc_id}.txt"
            if not raw_path.exists():
                raise Problem(f"missing file {raw_path.relative_to(ROOT)} — save the transcript there")
            raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
            text = clean(raw_text)
            words = word_count(text)
            if words < MIN_WORDS:
                raise Problem(f"only {words} words after cleaning, expected >= {MIN_WORDS}")
            # Corroborate the speaker against the document, not just the YAML.
            head = text[:3000].casefold()
            if SPEAKER not in head:
                raise Problem(
                    "the transcript's first 3000 characters do not mention Lee Kuan Yew. "
                    "Confirm the speaker, and keep the archive's own header line "
                    "(e.g. 'SPEECH BY MR LEE KUAN YEW...') at the top of the file."
                )
            (PROCESSED / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            documents.append({
                "document_id": doc_id,
                "title": entry["title"].strip(),
                "date": str(entry["date"]).strip(),
                "source_url": entry["source_url"].strip(),
                "speaker": entry["speaker"].strip(),
                "topic": entry["topic"].strip(),
                "note": str(entry.get("note", "") or "").strip(),
                "retrieved_at": str(entry.get("retrieved_at") or dt.date.today().isoformat()),
                "word_count": words,
                "sha256": sha256_of(text),
                "chunk_count": None,  # filled by src/process.py
            })
        except Problem as exc:
            failures.append({"document_id": entry.get("document_id", "<unnamed>"), "error": str(exc)})

    for failure in failures:
        print(f"  FAIL {failure['document_id']}: {failure['error']}", file=sys.stderr)
    if failures and strict:
        raise Problem(f"{len(failures)} of {len(speeches)} entries failed — nothing written to the manifest")

    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "collection_method": (
            "Manual retrieval in a browser. nas.gov.sg returns an AWS WAF bot "
            "challenge to scripted clients, so this corpus was not crawled. "
            "See docs/collection-blocked.md."
        ),
        "copyright_note": (
            "Full texts are NOT redistributed. data/raw/ and data/processed/ are "
            "gitignored. This manifest carries metadata and hashes only."
        ),
        "document_count": len(documents),
        "failed_count": len(failures),
        "failures": failures,
        "documents": sorted(documents, key=lambda d: d["document_id"]),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lenient", action="store_true",
                        help="write the manifest even if some entries fail (failures are recorded in it)")
    args = parser.parse_args()
    try:
        manifest = ingest(strict=not args.lenient)
    except Problem as exc:
        print(f"collect.py stopped: {exc}", file=sys.stderr)
        return 1
    print(f"ingested {manifest['document_count']} documents, {manifest['failed_count']} failed")
    for doc in manifest["documents"]:
        print(f"  {doc['document_id']:<40} {doc['word_count']:>6} words  {doc['topic']}")
    print(f"manifest -> {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
