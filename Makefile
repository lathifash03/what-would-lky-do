.PHONY: install collect process embed calibrate questions eval test chat check
PY := ./.venv/bin/python

install:
	python3 -m venv .venv && ./.venv/bin/pip install -q -r requirements.txt

collect:    ; $(PY) -m src.collect
process:    ; $(PY) -m src.process
embed:      ; $(PY) -m src.embed
questions:  ; $(PY) eval/make_questions.py
calibrate:  ; $(PY) -m src.calibrate
eval:       ; $(PY) eval/run_eval.py
test:       ; $(PY) -m pytest tests/ -q
chat:       ; $(PY) app/chat.py

# Full rebuild, in the order the pipeline depends on.
all: collect process embed questions calibrate eval

# Pre-commit safety: no key, no corpus text.
check:
	@git diff --cached --name-only | grep -E '^data/(raw|processed)/|\.env$$|\.npy$$' && \
	  (echo "REFUSING: corpus text, key or vectors staged" && exit 1) || echo "staged paths OK"
	@git diff --cached | grep -E 'sk-[A-Za-z0-9_-]{20,}' && \
	  (echo "REFUSING: key-shaped string in staged diff" && exit 1) || echo "no key in staged diff"
