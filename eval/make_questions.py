"""Build eval/questions.json: 12 synthetic + 6 hand-written + 6 adversarial.

The synthetic 12 are written BY the model FROM a known chunk, and that chunk is
the gold label. They are flagged `synthetic` because they are easier than real
questions by construction: the question is derived from the very passage that
should be retrieved, so its vocabulary already overlaps the target. Treat the
hit@5 number on this subset as an upper bound, not an estimate.

The hand-written 6 come from eval/handwritten.json, which a human fills in and
confirms. The adversarial 6 are fixed in eval/adversarial.json.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, chat_model
from src.process import load_chunks

OUT = ROOT / "eval" / "questions.json"
HANDWRITTEN = ROOT / "eval" / "handwritten.json"
ADVERSARIAL = ROOT / "eval" / "adversarial.json"
N_SYNTHETIC = 12
SEED = 20261107

INSTRUCTION = (
    "You are building a retrieval test set. Below is one passage from a speech by "
    "Lee Kuan Yew.\n\n"
    "Write ONE question that this passage answers. Constraints:\n"
    "- It must be answerable from this passage alone.\n"
    "- Do not copy distinctive phrases from the passage; a test whose question "
    "  repeats the answer's wording measures string overlap, not retrieval.\n"
    "- Ask about the substance, not about the document.\n"
    "- One sentence. No preamble.\n\n"
    "Return JSON: {\"question\": \"...\"}"
)


def pick_chunks(chunks: list[dict], n: int) -> list[dict]:
    """Spread the sample across documents before taking more than one from any."""
    rng = random.Random(SEED)
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["document_id"], []).append(chunk)
    for pool in by_doc.values():
        rng.shuffle(pool)
    picked, round_index = [], 0
    while len(picked) < n:
        added = False
        for doc_id in sorted(by_doc):
            if round_index < len(by_doc[doc_id]):
                picked.append(by_doc[doc_id][round_index])
                added = True
                if len(picked) == n:
                    break
        if not added:
            break
        round_index += 1
    return picked


def main() -> int:
    from openai import OpenAI

    chunks = load_chunks()
    if len(chunks) < N_SYNTHETIC:
        print(f"only {len(chunks)} chunks; need at least {N_SYNTHETIC}", file=sys.stderr)
        return 1

    client = OpenAI()
    model = chat_model()
    questions = []

    for i, chunk in enumerate(pick_chunks(chunks, N_SYNTHETIC), start=1):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": chunk["text"]},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        text = json.loads(response.choices[0].message.content)["question"].strip()
        questions.append({
            "id": f"syn-{i:02d}",
            "question": text,
            "answerable": True,
            "kind": "synthetic",
            "synthetic": True,
            "acceptable_modes": ["documented"],
            "gold_chunk_ids": [chunk["chunk_id"]],
            "gold_document_ids": [chunk["document_id"]],
            "generated_by": model,
            "note": "Written by the model from the gold chunk. Easier than a real question."
        })
        print(f"  syn-{i:02d}  [{chunk['chunk_id']}]  {text}")

    handwritten = json.loads(HANDWRITTEN.read_text(encoding="utf-8")) if HANDWRITTEN.exists() else []
    unconfirmed = [q for q in handwritten if not q.get("confirmed")]
    if unconfirmed:
        print(f"\nWARNING: {len(unconfirmed)} hand-written questions are not confirmed "
              f"(\"confirmed\": true) and are excluded.", file=sys.stderr)
    handwritten = [q for q in handwritten if q.get("confirmed")]

    adversarial = json.loads(ADVERSARIAL.read_text(encoding="utf-8"))
    everything = questions + handwritten + adversarial
    OUT.write_text(json.dumps(everything, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(questions)} synthetic + {len(handwritten)} hand-written + "
          f"{len(adversarial)} adversarial = {len(everything)} -> eval/questions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
