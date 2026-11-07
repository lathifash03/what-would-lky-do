# Sample conversations
Captured 2026-11-07T09:08:14+00:00 from one real run of
`app/chat.py`'s code path (`answer_question` then `render`), pasted unedited.
**Outputs vary between runs.** The model is called at temperature 0.2 and the retry
loop depends on what the first attempt produces, so re-running these four questions
will not reproduce this text word for word. The gates are deterministic; the prose is not.

> AI simulation based on archived speeches, not Lee Kuan Yew.

model `gpt-4.1-mini` | persona prompt `4c998c104990` | 127 chunks | abstention threshold 0.4317

---

## a. Why did Singapore separate from Malaysia?
*Designed behaviour: documented with citations.* Actual mode: `abstain`, gates FAILED, attempts 3.
```
GATE FAILURE — this answer did not pass verification and is not shown.
  [quote] quotes[2] is 31 words, the limit is 25. Shorten it to the essential span.
  [quote] quotes[2] is not a verbatim span of lky_1965-10-17_leong-khay-huay-kuan-TRANSLATION#0006. Copy the words exactly as they appear, or drop the quote. Paraphrase is not allowed in 'quotes'.
  [translation] this answer uses translated records (['lky_1965-10-17_leong-khay-huay-kuan-TRANSLATION#0003', 'lky_1965-10-17_leong-khay-huay-kuan-TRANSLATION#0004', 'lky_1965-10-17_leong-khay-huay-kuan-TRANSLATION#0006', 'lky_1965-10-17_leong-khay-huay-kuan-TRANSLATION#0013']), so it must contain the phrase "in the official translation".
  attempts: 3
```

---

## b. What do you think of cryptocurrency?
*Designed behaviour: abstain.* Actual mode: `abstain`, gates passed, attempts 1.
```
The archive does not cover Lee Kuan Yew's views on cryptocurrency.

mode: abstain — the archive does not cover this
best similarity: 0.205 (threshold 0.432)
attempts: 1 (passed first try)
```

---

## c. What would you say about housing prices in Singapore today?
*Designed behaviour: labelled extrapolation.* Actual mode: `abstain`, gates passed, attempts 1.
```
The archive does not cover housing prices in Singapore today. There is no information in the retrieved material about current housing prices or policies.

mode: abstain — the archive does not cover this
best similarity: 0.532 (threshold 0.432)
attempts: 1 (passed first try)
```

---

## d. How did you keep the government free of corruption?
*Designed behaviour: documented; in run 1 this failed the gate three times on a 31-word quote.* Actual mode: `abstain`, gates FAILED, attempts 3.
```
GATE FAILURE — this answer did not pass verification and is not shown.
  [quote] quotes[2] is 31 words, the limit is 25, and no prefix within the limit ends at a sentence or clause boundary. Choose a shorter span that ends cleanly.
  [quote] quotes[4] is 29 words, the limit is 25. Shorten it to the essential span.
  [quote] quotes[4] is not a verbatim span of lky_2005-04-28_ethical-leadership-kl#0005. Copy the words exactly as they appear, or drop the quote. Paraphrase is not allowed in 'quotes'.
  attempts: 3
```
