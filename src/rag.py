"""Step 4+5: retrieve, generate, gate, retry.

The model proposes; the gates dispose. Nothing the model returns reaches the
user until src/gates.py has checked every citation, every quote, every figure
and the mode label against the retrieved text. On failure the exact failure
list and the model's own previous answer go back to it, at most twice. This is
the retry pattern from rumah123-ig-weekly-generator/src/retry.py.

If the answer still fails after the last attempt, the application says so. A
failing answer is never shown as if it had passed.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT, chat_model, prompt_hash, prompt_text
from .gates import EXTRAPOLATION_LABEL, Failure, format_failures, run_gates

DISCLOSURE = "AI simulation based on archived speeches, not Lee Kuan Yew."
THRESHOLD_FILE = ROOT / "config" / "threshold.json"
LOGS = ROOT / "logs"
MAX_RETRIES = 2
PERSONA = "persona.md"


class NotCalibrated(RuntimeError):
    pass


def load_threshold() -> tuple[float, dict]:
    """The abstention threshold, which is calibrated and never guessed.

    src/calibrate.py derives it from retrieval scores on eval/questions.json.
    Until that has been run there is no honest default, so this raises.
    """
    if not THRESHOLD_FILE.exists():
        raise NotCalibrated(
            "config/threshold.json missing. The abstention threshold must be calibrated, "
            "not guessed: run `python -m src.calibrate`."
        )
    data = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
    return float(data["threshold"]), data


def format_context(retrieved: list[dict]) -> str:
    """Chunks as the model sees them, each labelled with the id it must cite."""
    blocks = []
    for chunk in retrieved:
        blocks.append(
            f"[chunk_id: {chunk['chunk_id']}]\n"
            f"Source: \"{chunk['title']}\", {chunk['date']} (similarity {chunk['score']:.3f})\n"
            f"---\n{chunk['text']}"
        )
    return "\n\n========\n\n".join(blocks)


def build_messages(question: str, retrieved: list[dict], best: float, threshold: float,
                   previous: dict | None = None, failures: list[Failure] | None = None) -> list[dict]:
    system = prompt_text(PERSONA)
    user = (
        f"RETRIEVED CHUNKS (the only material you may use):\n\n{format_context(retrieved)}\n\n"
        f"========\n\n"
        f"Best similarity score: {best:.3f}. Calibrated abstention threshold: {threshold:.3f}.\n"
        f"{'This is BELOW the threshold, so mode must be abstain.' if best < threshold else ''}\n\n"
        f"QUESTION: {question}\n\n"
        f"Reply with the JSON object described in your instructions."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if previous is not None and failures:
        messages.append({"role": "assistant", "content": json.dumps(previous)})
        messages.append({"role": "user", "content": (
            "Your answer failed these automated checks:\n\n"
            f"{format_failures(failures)}\n\n"
            "Fix every one of them and return the corrected JSON object. "
            "Do not explain the fixes; return only the JSON."
        )})
    return messages


@dataclass
class Attempt:
    index: int
    answer: dict | None
    failures: list[str]
    error: str | None = None


@dataclass
class Result:
    question: str
    retrieved: list[dict]
    best_similarity: float
    threshold: float
    passed: bool
    answer: dict | None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def first_attempt_passed(self) -> bool:
        return bool(self.attempts) and not self.attempts[0].failures and self.attempts[0].error is None

    @property
    def mode(self) -> str:
        return (self.answer or {}).get("mode", "none")


def _call(client, model: str, messages: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def answer_question(index, question: str, client=None, top_k: int = 5,
                    max_retries: int = MAX_RETRIES) -> Result:
    from openai import OpenAI

    client = client or OpenAI()
    model = chat_model()
    threshold, _ = load_threshold()

    retrieved = index.search(question, top_k=top_k)
    best = retrieved[0]["score"] if retrieved else 0.0

    result = Result(question=question, retrieved=retrieved, best_similarity=best,
                    threshold=threshold, passed=False, answer=None)

    previous: dict | None = None
    failures: list[Failure] = []
    for attempt_index in range(max_retries + 1):
        messages = build_messages(question, retrieved, best, threshold, previous, failures)
        try:
            candidate = _call(client, model, messages)
        except Exception as exc:                      # recorded, never silently swallowed
            result.attempts.append(Attempt(attempt_index, None, [], error=f"{type(exc).__name__}: {exc}"))
            continue
        failures = run_gates(candidate, retrieved, best, threshold)
        result.attempts.append(Attempt(attempt_index, candidate, [str(f) for f in failures]))
        if not failures:
            result.passed, result.answer = True, candidate
            return result
        previous = candidate
    result.answer = previous
    return result


def log_turn(result: Result, path: Path | None = None) -> Path:
    """Append one turn to a run log. Written whether or not the gates passed."""
    LOGS.mkdir(exist_ok=True)
    path = path or LOGS / f"chat-{dt.date.today().isoformat()}.jsonl"
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "question": result.question,
        "chat_model": chat_model(),
        "persona_prompt_sha256_12": prompt_hash(PERSONA),
        "threshold": result.threshold,
        "best_similarity": round(result.best_similarity, 4),
        "retrieved": [{"chunk_id": c["chunk_id"], "score": round(c["score"], 4)} for c in result.retrieved],
        "attempts": [{"index": a.index, "failures": a.failures, "error": a.error} for a in result.attempts],
        "attempt_count": len(result.attempts),
        "first_attempt_passed": result.first_attempt_passed,
        "passed": result.passed,
        "mode": result.mode,
        "citations": (result.answer or {}).get("citations", []),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return path
