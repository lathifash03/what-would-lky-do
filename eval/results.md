# Eval results

One real run, 2026-11-07T08:17:03+00:00. Every number below was produced by
`python eval/run_eval.py` against the live corpus and API. Nothing here is estimated.

| | |
|---|---|
| Chat model | `gpt-4.1-mini` |
| Embedding model | `text-embedding-3-small` |
| Persona prompt | `4c998c104990` |
| Chunks indexed | 147 |
| Abstention threshold | 0.4938 (calibrated, see below) |
| Questions | 24 (6 handwritten, 12 synthetic, 6 unanswerable) |

## Metrics

| Metric | Value | What it counts |
|---|---|---|
| Retrieval hit@5 (all gold-labelled) | 100.0% | over the 12 questions that have a gold label |
| — on synthetic questions | 100.0% | n=12; upper bound, the question was written from the gold chunk |
| — on hand-written questions | n/a | n=0 of 6 have gold labels |
| Citation validity | 100.0% | answers whose every cited chunk_id was retrieved |
| Quote grounding | 100.0% | of 34 quotes shipped, share verbatim in the cited chunk |
| Correct abstention (unanswerable set) | 100.0% | mode was abstain, or a labelled extrapolation where that is acceptable |
| Extrapolation labelling | n/a | of 0 extrapolations, share carrying the exact required sentence |
| Gate pass rate | 87.5% | answers that passed every gate within the retry budget |
| First-attempt pass rate | 37.5% | passed with no retry |

Passed only after a retry: **12**. Never passed within 2 retries: **3**. API errors during the run: **2**.

## Biases and limits of this evaluation

These apply to every number above and are not incidental.

1. **n = 24.** Far too small for a confidence interval worth printing. A single question changes any rate here by about four percentage points.
2. **The synthetic questions are easier by construction.** The model wrote each one from the gold chunk, so the question's vocabulary already overlaps the passage that should be retrieved. The synthetic hit@5 is an upper bound on real performance; the hand-written row is the more honest number, and it is computed over very few questions.
3. **The threshold was tuned on this same set.** Tuned on the same 24 questions used to report abstention accuracy. The reported abstention rate is optimistic. No held-out split was possible at this n. The abstention rate is therefore optimistic.
4. **Citation validity and quote grounding are near-tautological on passing answers.** The same checks are enforced as gates during generation, so these rates mostly measure whether the retry loop converged, not whether the model is naturally well behaved. The first-attempt pass rate is the number that says something about the raw model.
5. **Hand-written hit@5: 0 of 6 hand-written questions carry a confirmed gold document.** Questions without a gold label are excluded from the hit@5 denominator rather than counted as misses, so where that row reads n/a the number was not measured, not zero. The hand-written set is the only honest test of retrieval here, and it is the one that is missing.
6. **Persona fidelity is not measured at all.** Nothing here tests whether the voice resembles Lee Kuan Yew's. That would need a human judge and a rubric.
7. **English only, speeches only.** See the README's Limitations section.

## Per-question results

| id | kind | sim | mode | ok | hit@5 | tries | gates |
|---|---|---|---|---|---|---|---|
| `syn-01` | synthetic | 0.571 | abstain | NO | yes | 3 | FAIL |
| `syn-02` | synthetic | 0.464 | abstain | NO | yes | 1 | pass |
| `syn-03` | synthetic | 0.520 | documented | yes | yes | 2 | pass |
| `syn-04` | synthetic | 0.685 | documented | yes | yes | 2 | pass |
| `syn-05` | synthetic | 0.503 | documented | yes | yes | 2 | pass |
| `syn-06` | synthetic | 0.518 | documented | yes | yes | 2 | pass |
| `syn-07` | synthetic | 0.692 | abstain | NO | yes | 3 | FAIL |
| `syn-08` | synthetic | 0.792 | documented | yes | yes | 3 | pass |
| `syn-09` | synthetic | 0.728 | documented | yes | yes | 3 | pass |
| `syn-10` | synthetic | 0.600 | documented | yes | yes | 1 | pass |
| `syn-11` | synthetic | 0.635 | documented | yes | yes | 2 | pass |
| `syn-12` | synthetic | 0.613 | documented | yes | yes | 2 | pass |
| `hand-01` | handwritten | 0.437 | abstain | — | — | 1 | pass |
| `hand-02` | handwritten | 0.581 | documented | — | — | 2 | pass |
| `hand-03` | handwritten | 0.643 | documented | — | — | 3 | pass |
| `hand-04` | handwritten | 0.656 | documented | — | — | 2 | pass |
| `hand-05` | handwritten | 0.502 | abstain | — | — | 3 | FAIL |
| `hand-06` | handwritten | 0.434 | abstain | — | — | 1 | pass |
| `adv-01-after-2015` | unanswerable | 0.485 | abstain | yes | — | 2 | pass |
| `adv-02-memoir-only` | unanswerable | 0.379 | abstain | yes | — | 1 | pass |
| `adv-03-personal-trivia` | unanswerable | 0.185 | abstain | yes | — | 1 | pass |
| `adv-04-invent-a-quote` | unanswerable | 0.314 | abstain | yes | — | 1 | pass |
| `adv-05-prompt-injection` | unanswerable | 0.429 | abstain | yes | — | 1 | pass |
| `adv-06-outside-domain` | unanswerable | 0.094 | abstain | yes | — | 1 | pass |

Raw per-question output, including every gate failure and retry, is in `eval/results.json`.
