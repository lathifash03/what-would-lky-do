"""CLI chat. Prints the answer, the mode, and the sources behind it.

Every turn is logged to logs/, with the chat model and the persona prompt hash,
whether the gates passed or not. A turn that failed its gates is shown as a
failure, never dressed up as an answer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import chat_model, prompt_hash
from src.embed import Index
from src.rag import (DISCLOSURE, answer_question, load_threshold, log_turn,
                     manifest_titles)

MODE_NOTE = {
    "documented": "grounded in the retrieved speeches",
    "extrapolation": "reasoned beyond the record and labelled as such",
    "abstain": "the archive does not cover this",
}


def render(result) -> str:
    lines = []
    if not result.passed:
        lines.append("GATE FAILURE — this answer did not pass verification and is not shown.")
        last = result.attempts[-1] if result.attempts else None
        if last and last.error:
            lines.append(f"  error: {last.error}")
        for failure in (last.failures if last else []):
            lines.append(f"  {failure}")
        lines.append(f"  attempts: {len(result.attempts)}")
        return "\n".join(lines)

    answer = result.answer
    lines.append(answer["answer"].strip())
    lines.append("")
    lines.append(f"mode: {answer['mode']} — {MODE_NOTE.get(answer['mode'], '')}")
    lines.append(f"best similarity: {result.best_similarity:.3f} (threshold {result.threshold:.3f})")
    if answer.get("quotes"):
        lines.append("quotes (verified verbatim):")
        for quote in answer["quotes"]:
            lines.append(f"  \"{quote['text']}\"  — {quote['chunk_id']}")
    by_id = {c["chunk_id"]: c for c in result.retrieved}
    # Titles live in the manifest, not on the chunk: an archive heading left
    # in the indexed text would be retrievable and quotable as if it were
    # speech, so it is stripped at processing time.
    titles = manifest_titles()
    if answer.get("citations"):
        lines.append("sources:")
        for cid in answer["citations"]:
            chunk = by_id.get(cid, {})
            doc = chunk.get("document_id", "?")
            notes = []
            if chunk.get("translated"):
                notes.append("official translation")
            if not chunk.get("quotable", True):
                notes.append("cite only, not quotable")
            note = f"  [{'; '.join(notes)}]" if notes else ""
            lines.append(f"  {doc}  {chunk.get('date', '?')}  {cid}{note}")
            lines.append(f"      \"{titles.get(doc, '?')}\"")
    lines.append(f"attempts: {len(result.attempts)}"
                 f"{' (passed first try)' if result.first_attempt_passed else ' (passed after retry)'}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="ask one question and exit")
    args = parser.parse_args()

    threshold, _ = load_threshold()
    index = Index()

    print("=" * 72)
    print(DISCLOSURE)
    print("=" * 72)
    print(f"model {chat_model()} | persona prompt {prompt_hash('persona.md')} | "
          f"{len(index.chunks)} chunks | abstention threshold {threshold:.3f}")
    print()

    if args.question:
        question = " ".join(args.question)
        result = answer_question(index, question)
        print(render(result))
        print(f"\nlogged to {log_turn(result).name}")
        return 0 if result.passed else 1

    print("Ask a question, or Ctrl-D to leave.\n")
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        result = answer_question(index, question)
        print()
        print(render(result))
        log_turn(result)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
