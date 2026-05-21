# Step Report: Cascading Classifier + RAGAS Eval Runner

**Date:** 2026-05-21
**Branch:** feature/foundations-skeleton
**Commit:** d719ab1

---

## What was built

- **`backend/app/config.py`** — Added `classify_cascade_threshold: float = Field(default=0.75)` to `Settings`.  Configurable without rebuild; set to 0.0 to disable cascade entirely.
- **`backend/app/domain/classification.py`** — Added `cascade_triggered: bool` and `cascade_model: str | None` to `ClassificationResult` so every caller knows which path was taken.
- **`backend/app/services/classification.py`** — Rewrote `ClassificationService` with cascade logic: calls DistilBERT first; if `confidence < cascade_threshold`, re-classifies with the LLM (Gemini) using a few-shot JSON prompt; total latency = DL + LLM; falls back gracefully on any LLM failure.
- **`backend/app/api/routes/classify.py`** — Now uses `ClassificationServiceDep` (injected via DI); response includes `cascade_triggered` and `cascade_model`.
- **`backend/app/api/dependencies.py`** — `get_classification_service` wires `gemini_client`, `cascade_threshold`, and `gemini_model` into the service.
- **`backend/eval/rag/run_eval.py`** — Full RAGAS runner: `--offline` mode for structure-only CI checks; live mode hits `/rag/search` + `/chat`, computes faithfulness + answer_relevancy (Gemini judge via OpenAI-compat endpoint), plus Hit@5 + MRR@10.
- **`backend/pyproject.toml`** — Added `eval` dependency group: `ragas>=0.2.6`, `langchain-openai>=0.3.0`, `datasets>=3.4.0`, `pandas>=2.0.0`.

## Tests written

No new test files added in this commit (cascade unit tests would go in `tests/unit/test_classification_cascade.py`).

## Checks run

- `ruff check` — all 5 modified files: **PASS**
- `mypy --strict` on 4 new/modified files — **PASS** (zero errors in owned files; 10 pre-existing errors in unmodified files)

## What's next

Write `tests/unit/test_classification_cascade.py` to cover: DL high-confidence (no cascade), DL low-confidence (cascade fires), model-server down (cascade fires), LLM fails (returns DL fallback).
