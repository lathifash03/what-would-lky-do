# Held-out human-written evaluation (n=8)

One real run, 2026-11-07T09:15:54+00:00. Model `gpt-4.1-mini`, embeddings `text-embedding-3-small`, 127 chunks indexed, threshold **0.4317** (frozen).

**n = 8.** This is the only set in this project whose questions *and* gold
documents were both chosen by a human, the project's author. Six are answerable
with author-selected gold documents; two are unanswerable by design.

**Nothing was tuned on this set.** The threshold was frozen at 0.4317 before this
set was ever run, the gates and the persona prompt were already committed, the set
was run once, and no code, prompt or threshold was changed after seeing these
results. At n=8 a single question moves any rate here by 12.5 points.

## Metrics

| Metric | Value |
|---|---|
| Retrieval hit@5 (answerable) | **83.3%** over 6 gold-labelled questions |
| Answerable: answered | **3 of 6** |
| Answerable: refused (abstained) | **3 of 6** |
| Unanswerable: correctly refused | **2 of 2** |
| First-attempt pass rate | **37.5%** |
| Model compliance on first attempt | **50.0%** (no quote over 25 words before trimming) |
| Auto-trimmed quotes | **1** in 1 answers |
| Gate pass rate | **62.5%** |
| API errors | 0 |

## Per question

| id | sim | mode | gates | tries | hit@5 |
|---|---|---|---|---|---|
| `human-01` | 0.499 | abstain | **FAIL** | 3 | yes |
| `human-02` | 0.523 | documented | pass | 1 | yes |
| `human-03` | 0.530 | abstain | **FAIL** | 3 | yes |
| `human-04` | 0.573 | abstain | **FAIL** | 3 | yes |
| `human-05` | 0.554 | documented | pass | 3 | yes |
| `human-06` | 0.615 | documented | pass | 1 | **no** |
| `human-07` | 0.393 | abstain | pass | 1 | — (unanswerable) |
| `human-08` | 0.499 | abstain | pass | 2 | — (unanswerable) |

Per-question detail, including every gate failure and retry, is in `eval/results_human.json`.
