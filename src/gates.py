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

from .process import looks_like_heading
from .textutil import fold, normalize_ws, numbers_in, quote_is_grounded, word_count

EXTRAPOLATION_LABEL = "Hypothetical extrapolation, not something he said."
TRANSLATION_PHRASE = "in the official translation"
MAX_QUOTE_WORDS = 25
MODES = {"documented", "extrapolation", "abstain"}

# Questions about his family, his health or his wife's illness are answered
# only from the 2010 NYT interview, where he spoke about them himself, and
# never by extrapolation.
PERSONAL_DOC = "lky_2010-09-01_interview-nyt-seth-mydans"
PERSONAL_PATTERNS = (
    "your wife", "his wife", "mrs lee", "kwa geok choo", "her illness",
    "your family", "his family", "your children", "his children",
    "your son", "his son", "your daughter", "his daughter",
    "your health", "his health", "your death", "his death", "dying",
    "mortality", "grief", "bereave", "widow", "marriage", "funeral",
)


def is_personal_question(question: str) -> bool:
    low = fold(question)
    return any(p in low for p in PERSONAL_PATTERNS)


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


def check_quotes(answer: dict, retrieved: list[dict],
                 titles: list[str] | None = None) -> list[Failure]:
    """Each quote must be verbatim, <= 25 words, and from a quotable document.

    Quotable excludes two kinds of document. A summary is somebody else's
    paraphrase of what Lee said. A record carrying long third-party extracts
    cannot be certified word by word as his. Both may still be cited.

    A quote matching an archive heading is also refused: the heading describes
    the record, it is not speech.
    """
    by_id = {c["chunk_id"]: c["text"] for c in retrieved}
    quotable = {c["chunk_id"]: bool(c.get("quotable", c.get("verbatim", True)))
                for c in retrieved}
    doc_of = {c["chunk_id"]: c.get("document_id", c["chunk_id"])
              for c in retrieved}
    folded_titles = [fold(t) for t in (titles or []) if t]
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
        if not quotable.get(cid, True):
            out.append(Failure(
                "quote",
                f"{where} quotes {cid}, which belongs to "
                f"{doc_of.get(cid)}. That document may be cited but never "
                f"quoted: it is a summary or it carries third-party extracts, "
                f"so its words cannot be certified as his. Cite it instead.",
            ))
            continue
        folded_quote = fold(text)
        if looks_like_heading(text) or any(
                folded_quote in t or t in folded_quote for t in folded_titles):
            out.append(Failure(
                "quote",
                f"{where} is an archive heading or document title, not "
                f"speech. Quote something he said.",
            ))
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


def check_translation_disclosure(answer: dict,
                                 retrieved: list[dict]) -> list[Failure]:
    """An answer resting on a translated record must say so.

    Two 1965 records are official translations of speeches given in Hokkien.
    The wording is a translator's, so presenting it as his English is wrong.
    """
    if answer.get("mode") == "abstain":
        return []
    cited = set(_as_list(answer.get("citations")))
    quoted = {q.get("chunk_id") for q in _as_list(answer.get("quotes"))
              if isinstance(q, dict)}
    used = cited | quoted
    translated = [c["chunk_id"] for c in retrieved
                  if c["chunk_id"] in used and c.get("translated")]
    if not translated:
        return []
    if TRANSLATION_PHRASE in fold(str(answer.get("answer", ""))):
        return []
    return [Failure(
        "translation",
        f"this answer uses translated records ({sorted(translated)}), so it "
        f"must contain the phrase \"{TRANSLATION_PHRASE}\".",
    )]


def check_personal_topic(answer: dict, retrieved: list[dict],
                         question: str) -> list[Failure]:
    """Family, health and bereavement are answered only from the interview.

    He spoke about his wife's illness and his own age in the 2010 NYT
    interview. Everything else in the archive is public policy, and inferring
    a private matter from a speech about housing would be invention.
    """
    if not is_personal_question(question):
        return []
    out: list[Failure] = []
    if answer.get("mode") == "extrapolation":
        out.append(Failure(
            "personal",
            "this is a question about his family, health or bereavement. "
            "Extrapolation is not allowed here: answer from the 2010 "
            "interview or abstain.",
        ))
    used = set(_as_list(answer.get("citations"))) | {
        q.get("chunk_id") for q in _as_list(answer.get("quotes"))
        if isinstance(q, dict)}
    doc_of = {c["chunk_id"]: c.get("document_id", c["chunk_id"])
              for c in retrieved}
    outside = sorted({doc_of.get(cid, cid) for cid in used
                      if doc_of.get(cid) != PERSONAL_DOC})
    if outside:
        out.append(Failure(
            "personal",
            f"a question about his family or health may only use "
            f"{PERSONAL_DOC}. These were used instead: {outside}.",
        ))
    return out


def run_gates(answer: dict, retrieved: list[dict], best_similarity: float,
              threshold: float, question: str = "",
              titles: list[str] | None = None) -> list[Failure]:
    """All gates. Shape failures short-circuit: the rest would be noise."""
    shape = check_shape(answer)
    if shape:
        return shape
    return (
        check_citations(answer, retrieved)
        + check_quotes(answer, retrieved, titles)
        + check_numbers(answer, retrieved)
        + check_label(answer)
        + check_abstention(answer, best_similarity, threshold)
        + check_translation_disclosure(answer, retrieved)
        + check_personal_topic(answer, retrieved, question)
    )
