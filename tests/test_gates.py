"""Gate tests. No API key, no network — the gates are pure functions.

Every gate gets a passing fixture and a failing fixture, because a gate that
only ever sees good input is not known to reject anything.
"""
import pytest

from src.gates import (
    EXTRAPOLATION_LABEL,
    check_abstention,
    check_citations,
    check_label,
    check_numbers,
    check_quotes,
    check_shape,
    run_gates,
)

CHUNK_A = {
    "chunk_id": "19650809-independence#001",
    "text": "We are going to have a multi-racial nation in Singapore. "
            "This is not a Malay nation; this is not a Chinese nation; this is not an Indian nation. "
            "In 1965 we had 1,900,000 people and no hinterland.",
}
CHUNK_B = {
    "chunk_id": "19780211-education#004",
    "text": "Education is the one investment that compounds. "
            "You cannot buy a trained mind in the market.",
}
RETRIEVED = [CHUNK_A, CHUNK_B]

GOOD = {
    "answer": "Singapore had no hinterland and 1,900,000 people in 1965. "
              "The answer was to make ourselves useful to the world.",
    "citations": ["19650809-independence#001"],
    "quotes": [{"text": "this is not a Chinese nation", "chunk_id": "19650809-independence#001"}],
    "mode": "documented",
}


def test_good_answer_passes_every_gate():
    assert run_gates(GOOD, RETRIEVED, best_similarity=0.55, threshold=0.30) == []


# --- shape -----------------------------------------------------------------

def test_shape_accepts_valid_object():
    assert check_shape(GOOD) == []


@pytest.mark.parametrize("bad", [
    "not an object",
    {"answer": "", "citations": [], "quotes": [], "mode": "documented"},
    {"answer": "x", "citations": [], "quotes": [], "mode": "speculation"},
    {"answer": "x", "citations": "nope", "quotes": [], "mode": "documented"},
])
def test_shape_rejects_malformed(bad):
    assert check_shape(bad)


def test_shape_failure_short_circuits_other_gates():
    failures = run_gates({"mode": "nonsense"}, RETRIEVED, 0.9, 0.3)
    assert all(f.gate == "shape" for f in failures)


# --- citations -------------------------------------------------------------

def test_citation_subset_passes():
    assert check_citations(GOOD, RETRIEVED) == []


def test_citation_outside_retrieved_set_fails():
    answer = {**GOOD, "citations": ["19991231-invented#001"]}
    failures = check_citations(answer, RETRIEVED)
    assert len(failures) == 1 and failures[0].gate == "citation"


def test_documented_mode_requires_a_citation():
    answer = {**GOOD, "citations": [], "quotes": []}
    assert any("no chunk was cited" in f.detail for f in check_citations(answer, RETRIEVED))


# --- quotes ----------------------------------------------------------------

def test_verbatim_quote_passes():
    assert check_quotes(GOOD, RETRIEVED) == []


def test_quote_survives_smart_punctuation_and_case():
    answer = {**GOOD, "quotes": [
        {"text": "You Cannot Buy A Trained Mind", "chunk_id": "19780211-education#004"},
    ], "citations": ["19780211-education#004"]}
    assert check_quotes(answer, RETRIEVED) == []


def test_paraphrased_quote_fails():
    answer = {**GOOD, "quotes": [
        {"text": "this is not a Chinese country", "chunk_id": "19650809-independence#001"},
    ]}
    failures = check_quotes(answer, RETRIEVED)
    assert len(failures) == 1 and "verbatim" in failures[0].detail


def test_quote_attributed_to_the_wrong_chunk_fails():
    answer = {**GOOD, "quotes": [
        {"text": "this is not a Chinese nation", "chunk_id": "19780211-education#004"},
    ]}
    assert check_quotes(answer, RETRIEVED)


def test_quote_over_25_words_fails():
    long_span = " ".join(["word"] * 26)
    chunk = {"chunk_id": "x#001", "text": long_span}
    answer = {**GOOD, "quotes": [{"text": long_span, "chunk_id": "x#001"}]}
    failures = check_quotes(answer, [chunk])
    assert any("26 words" in f.detail for f in failures)


def test_quote_of_exactly_25_words_passes():
    span = " ".join(["word"] * 25)
    chunk = {"chunk_id": "x#001", "text": span}
    answer = {**GOOD, "quotes": [{"text": span, "chunk_id": "x#001"}]}
    assert check_quotes(answer, [chunk]) == []


def test_abstention_may_not_carry_quotes():
    answer = {**GOOD, "mode": "abstain"}
    assert any(f.gate == "quote" for f in check_quotes(answer, RETRIEVED))


# --- numbers ---------------------------------------------------------------

def test_grounded_numbers_pass():
    assert check_numbers(GOOD, RETRIEVED) == []


def test_comma_grouping_does_not_count_as_ungrounded():
    answer = {**GOOD, "answer": "There were 1900000 people."}
    assert check_numbers(answer, RETRIEVED) == []


def test_invented_figure_fails():
    answer = {**GOOD, "answer": "Unemployment stood at 14 per cent in 1965."}
    failures = check_numbers(answer, RETRIEVED)
    assert len(failures) == 1 and "'14'" in failures[0].detail


def test_invented_year_fails():
    answer = {**GOOD, "answer": "By 1990 the policy had worked."}
    assert check_numbers(answer, RETRIEVED)


def test_numbers_are_not_checked_for_extrapolation():
    answer = {**GOOD, "mode": "extrapolation",
              "answer": f"{EXTRAPOLATION_LABEL} By 2026 the problem is different."}
    assert check_numbers(answer, RETRIEVED) == []


# --- label -----------------------------------------------------------------

def test_extrapolation_with_label_passes():
    answer = {**GOOD, "mode": "extrapolation",
              "answer": f"{EXTRAPOLATION_LABEL} He never addressed this."}
    assert check_label(answer) == []


def test_extrapolation_without_label_fails():
    answer = {**GOOD, "mode": "extrapolation", "answer": "He would probably have agreed."}
    failures = check_label(answer)
    assert len(failures) == 1 and failures[0].gate == "label"


def test_near_miss_label_is_not_accepted():
    answer = {**GOOD, "mode": "extrapolation",
              "answer": "Hypothetical extrapolation - not something he said."}
    assert check_label(answer)


def test_documented_answer_may_not_claim_to_be_an_extrapolation():
    answer = {**GOOD, "answer": f"{EXTRAPOLATION_LABEL} But it is documented."}
    assert check_label(answer)


# --- abstention threshold --------------------------------------------------

def test_above_threshold_any_mode_is_allowed():
    assert check_abstention(GOOD, best_similarity=0.42, threshold=0.30) == []


def test_below_threshold_must_abstain():
    failures = check_abstention(GOOD, best_similarity=0.11, threshold=0.30)
    assert len(failures) == 1 and failures[0].gate == "abstain"


def test_below_threshold_abstaining_passes():
    answer = {**GOOD, "mode": "abstain", "quotes": [], "citations": []}
    assert check_abstention(answer, best_similarity=0.11, threshold=0.30) == []
