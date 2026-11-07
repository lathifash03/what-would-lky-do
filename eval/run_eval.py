"""Run the eval. Deterministic metrics only — no RAGAS, no LLM judge.

Every metric here is computable by string and set operations against the
retrieved chunks. Nothing is scored by asking a model whether an answer was
good, because that would make the evaluation as fallible as the thing being
evaluated, and would not be reproducible.

Writes eval/results.json (raw, every question) and eval/results.md (the table
and the biases). Both come from one real run; neither is written if the run
does not complete.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, chat_model, prompt_hash
from src.embed import Index
from src.gates import EXTRAPOLATION_LABEL, check_citations, check_quotes
from src.rag import answer_question, load_threshold, log_turn
from src.textutil import fold

QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS_JSON = ROOT / "eval" / "results.json"
RESULTS_MD = ROOT / "eval" / "results.md"


def hit_at_5(question: dict, retrieved: list[dict]) -> bool | None:
    """Did the gold chunk, or any chunk of the gold document, make the top 5?

    None when the question has no gold label (the unanswerable set), so it is
    excluded from the rate rather than counted as a miss.
    """
    gold_chunks = set(question.get("gold_chunk_ids") or [])
    gold_docs = set(question.get("gold_document_ids") or [])
    if not gold_chunks and not gold_docs:
        return None
    got_chunks = {c["chunk_id"] for c in retrieved}
    got_docs = {c["document_id"] for c in retrieved}
    if gold_chunks:
        return bool(gold_chunks & got_chunks)
    return bool(gold_docs & got_docs)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def evaluate() -> dict:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    threshold, threshold_meta = load_threshold()
    index = Index()
    model = chat_model()

    rows = []
    for question in questions:
        result = answer_question(index, question["question"])
        log_turn(result)
        answer = result.answer or {}
        # Re-check the shipped answer independently of the retry loop, so the
        # reported validity is a property of what the user would have seen.
        citation_failures = check_citations(answer, result.retrieved) if answer else []
        quote_failures = check_quotes(answer, result.retrieved) if answer else []
        quotes = answer.get("quotes") or []
        grounded_quotes = len(quotes) - len([f for f in quote_failures if "verbatim" in f.detail])
        acceptable = question.get("acceptable_modes") or []

        rows.append({
            "id": question["id"],
            "kind": question["kind"],
            "answerable": question["answerable"],
            "question": question["question"],
            "best_similarity": round(result.best_similarity, 4),
            "retrieved": [c["chunk_id"] for c in result.retrieved],
            "gold": question.get("gold_chunk_ids") or question.get("gold_document_ids") or [],
            "hit_at_5": hit_at_5(question, result.retrieved),
            "mode": result.mode,
            "acceptable_modes": acceptable,
            "mode_correct": result.mode in acceptable if acceptable else None,
            "passed_gates": result.passed,
            "first_attempt_passed": result.first_attempt_passed,
            "attempts": len(result.attempts),
            "citations": answer.get("citations", []),
            "citation_valid": not citation_failures,
            "quote_count": len(quotes),
            "grounded_quote_count": grounded_quotes,
            "carries_label": fold(EXTRAPOLATION_LABEL) in fold(str(answer.get("answer", ""))),
            "attempt_failures": [a.failures for a in result.attempts],
            "errors": [a.error for a in result.attempts if a.error],
        })
        status = "ok" if result.passed else "GATE FAIL"
        print(f"  {question['id']:<18} {result.mode:<14} sim {result.best_similarity:.3f}  "
              f"attempts {len(result.attempts)}  {status}")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    synthetic = [r for r in rows if r["kind"] == "synthetic"]
    handwritten = [r for r in rows if r["kind"] == "handwritten"]
    scored = [r for r in rows if r["hit_at_5"] is not None]
    extrapolations = [r for r in rows if r["mode"] == "extrapolation"]
    total_quotes = sum(r["quote_count"] for r in rows)

    metrics = {
        "n_questions": len(rows),
        "hit_at_5_all_gold": _rate(sum(bool(r["hit_at_5"]) for r in scored), len(scored)),
        "hit_at_5_synthetic": _rate(sum(bool(r["hit_at_5"]) for r in synthetic), len(synthetic)),
        "hit_at_5_handwritten": _rate(sum(bool(r["hit_at_5"]) for r in handwritten), len(handwritten)),
        "citation_validity_rate": _rate(sum(r["citation_valid"] for r in rows), len(rows)),
        "quote_grounding_rate": _rate(sum(r["grounded_quote_count"] for r in rows), total_quotes),
        "total_quotes": total_quotes,
        "correct_abstention_rate": _rate(sum(bool(r["mode_correct"]) for r in unanswerable), len(unanswerable)),
        "extrapolation_label_rate": _rate(sum(r["carries_label"] for r in extrapolations), len(extrapolations)),
        "n_extrapolations": len(extrapolations),
        "gate_pass_rate": _rate(sum(r["passed_gates"] for r in rows), len(rows)),
        "first_attempt_pass_rate": _rate(sum(r["first_attempt_passed"] for r in rows), len(rows)),
        "passed_only_after_retry": sum(r["passed_gates"] and not r["first_attempt_passed"] for r in rows),
        "never_passed": sum(not r["passed_gates"] for r in rows),
        "api_errors": sum(len(r["errors"]) for r in rows),
    }

    return {
        "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "chat_model": model,
        "embed_model": index.meta["embed_model"],
        "persona_prompt_sha256_12": prompt_hash("persona.md"),
        "threshold": threshold,
        "threshold_meta": {k: v for k, v in threshold_meta.items() if k != "per_question"},
        "chunk_count": len(index.chunks),
        "metrics": metrics,
        "rows": rows,
    }


def write_markdown(report: dict) -> None:
    m = report["metrics"]
    counts = {}
    for row in report["rows"]:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    composition = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))

    lines = [
        "# Eval results",
        "",
        f"One real run, {report['run_at_utc']}. Every number below was produced by",
        "`python eval/run_eval.py` against the live corpus and API. Nothing here is estimated.",
        "",
        "| | |",
        "|---|---|",
        f"| Chat model | `{report['chat_model']}` |",
        f"| Embedding model | `{report['embed_model']}` |",
        f"| Persona prompt | `{report['persona_prompt_sha256_12']}` |",
        f"| Chunks indexed | {report['chunk_count']} |",
        f"| Abstention threshold | {report['threshold']} (calibrated, see below) |",
        f"| Questions | {m['n_questions']} ({composition}) |",
        "",
        "## Metrics",
        "",
        "| Metric | Value | What it counts |",
        "|---|---|---|",
        f"| Retrieval hit@5 (all gold-labelled) | {_pct(m['hit_at_5_all_gold'])} | gold chunk, or any chunk of the gold document, in the top 5 |",
        f"| — on synthetic questions | {_pct(m['hit_at_5_synthetic'])} | upper bound: the question was written from the gold chunk |",
        f"| — on hand-written questions | {_pct(m['hit_at_5_handwritten'])} | the harder and more honest number |",
        f"| Citation validity | {_pct(m['citation_validity_rate'])} | answers whose every cited chunk_id was retrieved |",
        f"| Quote grounding | {_pct(m['quote_grounding_rate'])} | of {m['total_quotes']} quotes shipped, share verbatim in the cited chunk |",
        f"| Correct abstention (unanswerable set) | {_pct(m['correct_abstention_rate'])} | mode was abstain, or a labelled extrapolation where that is acceptable |",
        f"| Extrapolation labelling | {_pct(m['extrapolation_label_rate'])} | of {m['n_extrapolations']} extrapolations, share carrying the exact required sentence |",
        f"| Gate pass rate | {_pct(m['gate_pass_rate'])} | answers that passed every gate within the retry budget |",
        f"| First-attempt pass rate | {_pct(m['first_attempt_pass_rate'])} | passed with no retry |",
        "",
        f"Passed only after a retry: **{m['passed_only_after_retry']}**. "
        f"Never passed within 2 retries: **{m['never_passed']}**. "
        f"API errors during the run: **{m['api_errors']}**.",
        "",
        "## Biases and limits of this evaluation",
        "",
        "These apply to every number above and are not incidental.",
        "",
        f"1. **n = {m['n_questions']}.** Far too small for a confidence interval worth printing. "
        "A single question changes any rate here by about four percentage points.",
        "2. **The synthetic questions are easier by construction.** The model wrote each one "
        "from the gold chunk, so the question's vocabulary already overlaps the passage that "
        "should be retrieved. The synthetic hit@5 is an upper bound on real performance; "
        "the hand-written row is the more honest number, and it is computed over very few questions.",
        f"3. **The threshold was tuned on this same set.** {report['threshold_meta'].get('bias_warning', '')} "
        "No held-out split was possible at this n. The abstention rate is therefore optimistic.",
        "4. **Citation validity and quote grounding are near-tautological on passing answers.** "
        "The same checks are enforced as gates during generation, so these rates mostly measure "
        "whether the retry loop converged, not whether the model is naturally well behaved. "
        "The first-attempt pass rate is the number that says something about the raw model.",
        "5. **Persona fidelity is not measured at all.** Nothing here tests whether the voice "
        "resembles Lee Kuan Yew's. That would need a human judge and a rubric.",
        "6. **English only, speeches only.** See the README's Limitations section.",
        "",
        "## Per-question results",
        "",
        "| id | kind | sim | mode | ok | hit@5 | tries | gates |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        hit = {True: "yes", False: "no", None: "—"}[row["hit_at_5"]]
        ok = {True: "yes", False: "NO", None: "—"}[row["mode_correct"]]
        lines.append(
            f"| `{row['id']}` | {row['kind']} | {row['best_similarity']:.3f} | {row['mode']} | "
            f"{ok} | {hit} | {row['attempts']} | {'pass' if row['passed_gates'] else 'FAIL'} |"
        )
    lines += ["", "Raw per-question output, including every gate failure and retry, is in "
                  "`eval/results.json`.", ""]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = evaluate()
    RESULTS_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report)
    m = report["metrics"]
    print()
    print(f"hit@5 {_pct(m['hit_at_5_all_gold'])} | citations {_pct(m['citation_validity_rate'])} | "
          f"quotes {_pct(m['quote_grounding_rate'])} | abstention {_pct(m['correct_abstention_rate'])}")
    print(f"gates {_pct(m['gate_pass_rate'])} | first attempt {_pct(m['first_attempt_pass_rate'])}")
    print("-> eval/results.json, eval/results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
