"""Normalisation and grounding primitives.

These are the deterministic checks. No model output is trusted here: a quote is
grounded only if it is literally present in the chunk it claims to come from,
and a figure is grounded only if it is literally present in the retrieved text.

Adapted from the claimcheck/textutil pair in rumah123-ig-weekly-generator.
"""
from __future__ import annotations

import re
import unicodedata

# Characters archives and word processors use that break naive substring checks.
_SMART = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "…": "...",
}


def normalize_ws(text: str) -> str:
    """Collapse whitespace and unify quotes/dashes, preserving word order.

    Used on both sides of every substring check. Without it a quote copied
    from a chunk fails the check purely because the archive used a curly
    apostrophe and the model emitted a straight one.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _SMART.items():
        text = text.replace(bad, good)
    return re.sub(r"\s+", " ", text).strip()


def fold(text: str) -> str:
    """normalize_ws plus case folding, for grounding comparisons only."""
    return normalize_ws(text).casefold()


def word_count(text: str) -> int:
    return len(normalize_ws(text).split())


def quote_is_grounded(quote: str, source: str) -> bool:
    """Whether the quote appears verbatim in the source, whitespace normalised.

    Case-insensitive, because the archives' own transcripts are inconsistent
    about capitalisation after an ellipsis. Word order and wording are not
    negotiable: a paraphrase is not a quote.
    """
    q, s = fold(quote), fold(source)
    if not q:
        return False
    return q in s


# A year is any standalone 4-digit number in 1800-2099. Everything else that
# contains a digit is treated as a figure.
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
# Figures: 12, 12.5, 1,200, 85%, $3, 3m. Captured with their surrounding digits
# so "1,200" is one token, not two.
_NUMBER_RE = re.compile(r"\d[\d,.]*")


def years_in(text: str) -> set[str]:
    return set(_YEAR_RE.findall(normalize_ws(text)))


def _canonical_number(token: str) -> str:
    """Strip grouping commas and trailing punctuation so 1,200. == 1200."""
    token = token.replace(",", "").rstrip(".")
    if token.startswith("."):
        token = token.lstrip(".")
    return token


def numbers_in(text: str) -> set[str]:
    """Every figure in the text, canonicalised. Years are included."""
    out = set()
    for raw in _NUMBER_RE.findall(normalize_ws(text)):
        canon = _canonical_number(raw)
        if canon:
            out.add(canon)
    return out


def ungrounded_numbers(answer: str, sources: str) -> set[str]:
    """Figures asserted in the answer that do not appear in the sources.

    Fail-closed: anything with a digit must be findable. A model that invents
    "Singapore's GDP grew 7 per cent" is caught by the 7, not by its wording.
    """
    return numbers_in(answer) - numbers_in(sources)


def sha256_of(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
