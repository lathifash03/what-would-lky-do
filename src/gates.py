"""Step 5: the deterministic gates. The model's answer is never trusted.

Each gate is a pure function of (answer, retrieved chunks, threshold). No model
is consulted, so the whole module is testable with no API key — that is the
point. On failure the exact reasons are formatted and handed back to the model
together with its own previous answer, the retry pattern reused from
rumah123-ig-weekly-generator/src/retry.py.

Fail-closed everywhere: a malformed answer fails, an unparseable quote fails,
an unknown mode fails.
"""
from __future__ import annotations

from dataclasses import dataclass

from .textutil import fold, normalize_ws, numbers_in, quote_is_grounded, word_count

EXTRAPOLATION_LABEL = "Hypothetical extrapolation, not something he said."
MAX_QUOTE_WORDS = 25
MODES = {"documented", "extrapolation", "abstain"}


@dataclass(frozen=True)
class Failure:
    gate: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.gate}] {self.detail}"


def format_failures(failures: list[Failure]) -> str:
    return "\n".join(f"- {f}" for f in failures)


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def check_shape(answer: dict) -> list[Failure]:
    """The answer must be a dict with the four fields and a legal mode."""
    out: list[Failure] = []
    if not isinstance(answer, dict):
        return [Failure("shape", "answer is not a JSON object")]
    if not isinstance(answer.get("answer"), str) or not answer["answer"].strip():
        out.append(Failure("shape", "field 'answer' must be a non-empty string"))
    mode = answer.get("mode")
    if mode not in MODES:
        out.append(Failure("shape", f"field 'mode' must be one of {sorted(MODES)}, got {mode!r}"))
    if not isinstance(answer.get("citations"), list):
        out.append(Failure("shape", "field 'citations' must be a list"))
    if not isinstance(answer.get("quotes"), list):
        out.append(Failure("shape", "field 'quotes' must be a list"))
    return out


def check_citations(answer: dict, retrieved: list[dict]) -> list[Failure]:
    """Every cited chunk_id must be one that was actually retrieved."""
    allowed = {c["chunk_id"] for c in retrieved}
    out: list[Failure] = []
    for cid in _as_list(answer.get("citations")):
        if cid not in allowed:
            out.append(Failure(
                "citation",
                f"cited chunk_id '{cid}' was not in the retrieved set. "
                f"You may only cite: {sorted(allowed)}",
            ))
    if answer.get("mode") == "documented" and not _as_list(answer.get("citations")):
        out.append(Failure("citation", "mode is 'documented' but no chunk was cited"))
    return out


def check_quotes(answer: dict, retrieved: list[dict]) -> list[Failure]:
    """Each quote must be verbatim from the chunk it names, and <= 25 words."""
    by_id = {c["chunk_id"]: c["text"] for c in retrieved}
    out: list[Failure] = []
    for i, quote in enumerate(_as_list(answer.get("quotes"))):
        where = f"quotes[{i}]"
        if not isinstance(quote, dict):
            out.append(Failure("quote", f"{where} is not an object"))
            continue
        text = quote.get("text")
        cid = quote.get("chunk_id")
        if not isinstance(text, str) or not text.strip():
            out.append(Failure("quote", f"{where} has no text"))
            continue
        if cid not in by_id:
            out.append(Failure("quote", f"{where} names chunk_id '{cid}', which was not retrieved"))
            continue
        n = word_count(text)
        if n > MAX_QUOTE_WORDS:
            out.append(Failure(
                "quote",
                f"{where} is {n} words, the limit is {MAX_QUOTE_WORDS}. Shorten it to the essential span.",
            ))
        if not quote_is_grounded(text, by_id[cid]):
            out.append(Failure(
                "quote",
                f"{where} is not a verbatim span of {cid}. Copy the words exactly as they appear, "
                f"or drop the quote. Paraphrase is not allowed in 'quotes'.",
            ))
    if answer.get("mode") == "abstain" and _as_list(answer.get("quotes")):
        out.append(Failure("quote", "mode is 'abstain' but quotes were supplied; abstentions quote nothing"))
    return out


def check_numbers(answer: dict, retrieved: list[dict]) -> list[Failure]:
    """Figures and years in the answer must appear in the retrieved text.

    Skipped for 'extrapolation', where the answer is explicitly reasoning
    beyond the record and may refer to the present day.
    """
    if answer.get("mode") == "extrapolation":
        return []
    corpus = " ".join(c["text"] for c in retrieved)
    ungrounded = numbers_in(str(answer.get("answer", ""))) - numbers_in(corpus)
    if not ungrounded:
        return []
    return [Failure(
        "number",
        f"these figures or years are in your answer but not in the retrieved chunks: "
        f"{sorted(ungrounded)}. Remove them, or use a figure that is in the text.",
    )]


def check_label(answer: dict) -> list[Failure]:
    """Extrapolations must carry the exact label; other modes must not."""
    text = normalize_ws(str(answer.get("answer", "")))
    has_label = fold(EXTRAPOLATION_LABEL) in fold(text)
    if answer.get("mode") == "extrapolation" and not has_label:
        return [Failure("label", f"mode is 'extrapolation', so the answer must contain exactly: {EXTRAPOLATION_LABEL}")]
    if answer.get("mode") != "extrapolation" and has_label:
        return [Failure("label", f"the answer carries the extrapolation label but mode is {answer.get('mode')!r}")]
    return []


def check_abstention(answer: dict, best_similarity: float, threshold: float) -> list[Failure]:
    """Below the calibrated threshold the only honest mode is abstain."""
    if best_similarity < threshold and answer.get("mode") != "abstain":
        return [Failure(
            "abstain",
            f"the best retrieved chunk scored {best_similarity:.3f}, below the calibrated "
            f"threshold {threshold:.3f}. The archive does not cover this. Set mode to 'abstain'.",
        )]
    return []


def run_gates(answer: dict, retrieved: list[dict], best_similarity: float,
              threshold: float) -> list[Failure]:
    """All gates. Shape failures short-circuit: the rest would be noise."""
    shape = check_shape(answer)
    if shape:
        return shape
    return (
        check_citations(answer, retrieved)
        + check_quotes(answer, retrieved)
        + check_numbers(answer, retrieved)
        + check_label(answer)
        + check_abstention(answer, best_similarity, threshold)
    )
