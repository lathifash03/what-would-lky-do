"""Calibrate the abstention threshold on the eval set.

The threshold is not a guess and not a round number someone liked. It is the
cosine score that best separates two populations of *retrieval* scores:

  positives — questions the corpus can answer (the synthetic and hand-written
              sets in eval/questions.json)
  negatives — questions it cannot (the unanswerable / adversarial set)

The chosen value maximises Youden's J (sensitivity + specificity - 1) over
every midpoint between observed scores. Ties go to the higher threshold, which
abstains more readily: a wrong abstention costs the user an answer, a wrong
answer costs them the truth.

This needs embeddings only, no generation, which is why it can run before
eval/run_eval.py and break the circular dependency between the two.

BIAS, stated plainly and repeated in eval/results.md: the threshold is tuned on
the same 24 questions it is then evaluated on. The abstention numbers in
results.md are therefore optimistic. With n=24 there was no honest way to hold
out a calibration split and still have a meaningful test set.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT
from .embed import Index

QUESTIONS = ROOT / "eval" / "questions.json"
THRESHOLD_FILE = ROOT / "config" / "threshold.json"


def _scores(index: Index, questions: list[dict]) -> list[dict]:
    out = []
    for question in questions:
        hits = index.search(question["question"], top_k=5)
        out.append({
            "id": question["id"],
            "answerable": bool(question["answerable"]),
            "best": hits[0]["score"] if hits else 0.0,
            "top_ids": [h["chunk_id"] for h in hits],
        })
    return out


def choose(scored: list[dict]) -> dict:
    positives = sorted(s["best"] for s in scored if s["answerable"])
    negatives = sorted(s["best"] for s in scored if not s["answerable"])
    if not positives or not negatives:
        raise RuntimeError("calibration needs both answerable and unanswerable questions")

    observed = sorted({s["best"] for s in scored})
    candidates = [observed[0] - 0.01]
    candidates += [(a + b) / 2 for a, b in zip(observed, observed[1:])]
    candidates.append(observed[-1] + 0.01)

    best = None
    for threshold in candidates:
        # answerable questions should score at or above the threshold
        sensitivity = sum(p >= threshold for p in positives) / len(positives)
        # unanswerable questions should fall below it
        specificity = sum(n < threshold for n in negatives) / len(negatives)
        j = sensitivity + specificity - 1
        if best is None or (j, threshold) > (best["youden_j"], best["threshold"]):
            best = {
                "threshold": round(threshold, 4),
                "youden_j": round(j, 4),
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
            }
    return best


def main() -> int:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    scored = _scores(Index(), questions)
    chosen = choose(scored)
    positives = [s["best"] for s in scored if s["answerable"]]
    negatives = [s["best"] for s in scored if not s["answerable"]]
    payload = {
        **chosen,
        "method": "Youden's J maximised over midpoints between observed best-chunk cosine scores",
        "calibrated_on": "eval/questions.json",
        "n_answerable": len(positives),
        "n_unanswerable": len(negatives),
        "answerable_best_score_range": [round(min(positives), 4), round(max(positives), 4)],
        "unanswerable_best_score_range": [round(min(negatives), 4), round(max(negatives), 4)],
        "bias_warning": (
            "Tuned on the same 24 questions used to report abstention accuracy. "
            "The reported abstention rate is optimistic."
        ),
        "per_question": scored,
    }
    THRESHOLD_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"threshold = {chosen['threshold']}  (Youden's J {chosen['youden_j']}, "
          f"sensitivity {chosen['sensitivity']}, specificity {chosen['specificity']})")
    print(f"answerable best scores   {min(positives):.3f} - {max(positives):.3f}  (n={len(positives)})")
    print(f"unanswerable best scores {min(negatives):.3f} - {max(negatives):.3f}  (n={len(negatives)})")
    print(f"-> config/threshold.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
