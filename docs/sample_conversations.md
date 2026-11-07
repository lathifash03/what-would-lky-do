# Sample conversations
Captured 2026-11-07T08:20:40+00:00 from one real run of
`app/chat.py`'s code path (`answer_question` then `render`), pasted unedited.
**Outputs vary between runs.** The model is called at temperature 0.2 and the retry
loop depends on what the first attempt produces, so re-running these three questions
will not reproduce this text word for word. The gates are deterministic; the prose is not.

> AI simulation based on archived speeches, not Lee Kuan Yew.

model `gpt-4.1-mini` | persona prompt `4c998c104990` | 147 chunks | abstention threshold 0.4938

---

## a. Why did Singapore separate from Malaysia?
*Designed behaviour: documented with citations.* Actual mode: `documented`, gates passed, attempts 2.

> Gates passed, but see README Known limitations: the content may reflect third-party text quoted in the source speech.
```
Singapore separated from Malaysia because the British had originally conceived and implemented a policy that treated Singapore as a different community with lesser political rights, separating it from the rest of the Malay Peninsula. This separation was a feature of the abortive Malayan Union and was detested by the Chinese community. The British feared that adding Singapore to the Federation would give the Chinese community a majority, altering the political balance. The opposition of the Chinese, especially the younger generation, was likely to continue or intensify unless Malaysia became a progressive and democratic society rather than an aristocratic one dominated by Malay feudal families. The separation was also influenced by political and racial tensions, and the British played a role in the process, with some viewing the separation as a British-inspired arrangement rather than solely a matter between local leaders.

mode: documented — grounded in the retrieved speeches
best similarity: 0.640 (threshold 0.494)
sources:
  lky_1965-12-14_parliament-motion-of-thanks  1965-12-14  lky_1965-12-14_parliament-motion-of-thanks#0018  [cite only, not quotable]
      "Speech moving the Motion of Thanks to the Yang di-Pertuan Negara on the opening of Parliament"
  lky_1965-12-14_parliament-motion-of-thanks  1965-12-14  lky_1965-12-14_parliament-motion-of-thanks#0014  [cite only, not quotable]
      "Speech moving the Motion of Thanks to the Yang di-Pertuan Negara on the opening of Parliament"
attempts: 2 (passed after retry)
```

---

## b. What do you think of cryptocurrency?
*Designed behaviour: abstain.* Actual mode: `abstain`, gates passed, attempts 1.
```
The archive does not cover Lee Kuan Yew's views on cryptocurrency.

mode: abstain — the archive does not cover this
best similarity: 0.205 (threshold 0.494)
attempts: 1 (passed first try)
```

---

## c. What would you say about housing prices in Singapore today?
*Designed behaviour: labelled extrapolation.* Actual mode: `abstain`, gates passed, attempts 1.
```
The archive does not cover housing prices in Singapore today. I have no documented material on this topic.

mode: abstain — the archive does not cover this
best similarity: 0.507 (threshold 0.494)
attempts: 1 (passed first try)
```
