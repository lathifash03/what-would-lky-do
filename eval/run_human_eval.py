"""Run ONLY the held-out human-written set (eval/human_written.json).

Separate from eval/run_eval.py on purpose: this set is held out. The threshold
was frozen at 0.4317 before it was ever run, and nothing is tuned on it. It is
run once, and whatever it reports is what gets reported.

It is the only set in this project whose questions AND gold documents were both
chosen by a human.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, chat_model, prompt_hash
from src.embed import Index
from src.rag import answer_question, load_threshold, log_turn

QUESTIONS = ROOT / "eval" / "human_written.json"
OUT_MD = ROOT / "eval" / "results_human.md"
OUT_JSON = ROOT / "eval" / "results_human.json"


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _rate(num, den):
    return round(num / den, 4) if den else None


def main() -> int:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    threshold, tmeta = load_threshold()
    if not tmeta.get("frozen"):
        print("refusing to run: the threshold is not frozen", file=sys.stderr)
        return 1
    index = Index()
    docs_indexed = {c["document_id"] for c in index.chunks}

    rows = []
    for q in questions:
        result = answer_question(index, q["question"])
        log_turn(result)
        answer = result.answer or {}
        got_docs = {c["document_id"] for c in result.retrieved}
        gold = set(q["gold_document_ids"])
        hit = bool(gold & got_docs) if gold else None
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "gold_document_ids": sorted(gold),
            "best_similarity": round(result.best_similarity, 4),
            "retrieved_documents": sorted(got_docs),
            "hit_at_5": hit,
            "mode": result.mode,
            "acceptable_modes": q["acceptable_modes"],
            "mode_correct": result.mode in q["acceptable_modes"],
            "passed_gates": result.passed,
            "first_attempt_passed": result.first_attempt_passed,
            "attempts": len(result.attempts),
            "auto_trimmed": int(answer.get("_auto_trimmed", 0) or 0),
            "quote_limit_broken_first_attempt": bool(
                (result.attempts[0].answer or {}).get("_quote_limit_broken", False)
                if result.attempts else False),
            "citations": answer.get("citations", []),
            "attempt_failures": [a.failures for a in result.attempts],
            "errors": [a.error for a in result.attempts if a.error],
        })
        print(f"  {q['id']:<10} {result.mode:<14} sim {result.best_similarity:.3f}  "
              f"tries {len(result.attempts)}  {'ok' if result.passed else 'GATE FAIL'}")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    scored = [r for r in rows if r["hit_at_5"] is not None]

    m = {
        "n_questions": len(rows),
        "hit_at_5_answerable": _rate(sum(bool(r["hit_at_5"]) for r in scored), len(scored)),
        "n_gold_labelled": len(scored),
        "answerable_answered": sum(1 for r in answerable if r["mode"] != "abstain"),
        "answerable_refused": sum(1 for r in answerable if r["mode"] == "abstain"),
        "n_answerable": len(answerable),
        "unanswerable_correctly_refused": sum(1 for r in unanswerable if r["mode_correct"]),
        "n_unanswerable": len(unanswerable),
        "first_attempt_pass_rate": _rate(sum(r["first_attempt_passed"] for r in rows), len(rows)),
        "model_compliance_first_attempt": _rate(
            sum(not r["quote_limit_broken_first_attempt"] for r in rows), len(rows)),
        "auto_trimmed_quotes": sum(r["auto_trimmed"] for r in rows),
        "answers_with_an_auto_trim": sum(1 for r in rows if r["auto_trimmed"]),
        "gate_pass_rate": _rate(sum(r["passed_gates"] for r in rows), len(rows)),
        "api_errors": sum(len(r["errors"]) for r in rows),
    }

    report = {
        "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "chat_model": chat_model(),
        "embed_model": index.meta["embed_model"],
        "persona_prompt_sha256_12": prompt_hash("persona.md"),
        "threshold": threshold,
        "threshold_frozen": True,
        "chunk_count": len(index.chunks),
        "metrics": m,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Held-out human-written evaluation (n=8)",
        "",
        f"One real run, {report['run_at_utc']}. Model `{report['chat_model']}`, "
        f"embeddings `{report['embed_model']}`, {report['chunk_count']} chunks indexed, "
        f"threshold **{threshold}** (frozen).",
        "",
        "**n = 8.** This is the only set in this project whose questions *and* gold",
        "documents were both chosen by a human, the project's author. Six are answerable",
        "with author-selected gold documents; two are unanswerable by design.",
        "",
        "**Nothing was tuned on this set.** The threshold was frozen at 0.4317 before this",
        "set was ever run, the gates and the persona prompt were already committed, the set",
        "was run once, and no code, prompt or threshold was changed after seeing these",
        "results. At n=8 a single question moves any rate here by 12.5 points.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Retrieval hit@5 (answerable) | **{_pct(m['hit_at_5_answerable'])}** over {m['n_gold_labelled']} gold-labelled questions |",
        f"| Answerable: answered | **{m['answerable_answered']} of {m['n_answerable']}** |",
        f"| Answerable: refused (abstained) | **{m['answerable_refused']} of {m['n_answerable']}** |",
        f"| Unanswerable: correctly refused | **{m['unanswerable_correctly_refused']} of {m['n_unanswerable']}** |",
        f"| First-attempt pass rate | **{_pct(m['first_attempt_pass_rate'])}** |",
        f"| Model compliance on first attempt | **{_pct(m['model_compliance_first_attempt'])}** (no quote over 25 words before trimming) |",
        f"| Auto-trimmed quotes | **{m['auto_trimmed_quotes']}** in {m['answers_with_an_auto_trim']} answers |",
        f"| Gate pass rate | **{_pct(m['gate_pass_rate'])}** |",
        f"| API errors | {m['api_errors']} |",
        "",
        "## Per question",
        "",
        "| id | sim | mode | gates | tries | hit@5 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        hit = {True: "yes", False: "**no**", None: "— (unanswerable)"}[r["hit_at_5"]]
        lines.append(
            f"| `{r['id']}` | {r['best_similarity']:.3f} | {r['mode']} | "
            f"{'pass' if r['passed_gates'] else '**FAIL**'} | {r['attempts']} | {hit} |")
    lines += ["", "Per-question detail, including every gate failure and retry, is in "
                  "`eval/results_human.json`.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"hit@5 {_pct(m['hit_at_5_answerable'])} | answered {m['answerable_answered']}/{m['n_answerable']} "
          f"| refused-correctly {m['unanswerable_correctly_refused']}/{m['n_unanswerable']}")
    print(f"first attempt {_pct(m['first_attempt_pass_rate'])} | compliance {_pct(m['model_compliance_first_attempt'])} "
          f"| trimmed {m['auto_trimmed_quotes']} | gates {_pct(m['gate_pass_rate'])}")
    print("-> eval/results_human.md, eval/results_human.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
