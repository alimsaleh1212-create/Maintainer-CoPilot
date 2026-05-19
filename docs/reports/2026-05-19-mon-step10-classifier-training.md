# MON Step 10 — DistilBERT Training Pipeline

**Date:** 2026-05-19
**Branch:** `feature/foundations-skeleton`
**Status:** complete — ruff clean, mypy clean, 6/6 unit tests pass

## What was built

### `backend/app/domain/classification.py`
Pydantic domain model `ClassificationResult` with `label`, `confidence`, `model_version`, `latency_ms` fields. Validates label is one of CLASS_NAMES at the boundary.

### `backend/app/ml/classifier.py`
- `ClassifierLoadError(RuntimeError)` — raised on every refuse-to-boot condition
- `_sha256_dir_weights(model_dir)` — hashes all `.safetensors`/`.bin` files sorted by name
- `DistilBertClassifier` — thread-safe inference wrapper; `predict(text)` runs torch forward pass with `no_grad`, returns `ClassificationResult` with latency
- `load_classifier(model_dir)` — verifies 6 conditions in order (dir exists → card exists → card valid JSON → sha256 field present → sha256 matches actual → weight files exist), then loads AutoModelForSequenceClassification + AutoTokenizer; logs with structlog

### `backend/app/ml/train_classifier.py`
- `TrainConfig` dataclass: lr=2e-5, batch=16, epochs=5, weight_decay=0.01, warmup_ratio=0.1, early_stopping_patience=2
- `train(train_path, val_path, config) -> Path` — full HF Trainer pipeline:
  1. Load splits with `load_split()`
  2. Tokenize with `AutoTokenizer.from_pretrained("distilbert-base-uncased")`
  3. Load `AutoModelForSequenceClassification` with 3-class head
  4. Train with `EarlyStoppingCallback(patience=2)`, eval/save per epoch
  5. Save best checkpoint + tokenizer to `artifacts/classifier/best/`
  6. Evaluate on val, write `model_card.json` (architecture, hyperparams, sha256, metrics, trained_at)
- All heavy imports (torch, transformers, datasets, evaluate) are lazy inside `train()` so FastAPI starts without GPU deps

### `backend/scripts/train_classifier.py`
CLI wrapper: `uv run python scripts/train_classifier.py [--epochs N] [--batch-size N] [--lr F]`

### `backend/eval/eval_thresholds.yaml`
Committed thresholds (placeholders before first run): classification macro_f1≥0.78, per_class_f1_min≥0.70, rag faithfulness≥0.85, answer_relevancy≥0.80, hit_at_5≥0.70.

### `backend/tests/unit/test_classifier_loads.py`
6 unit tests for every refuse-to-boot condition — none load a real model.

## Key decisions
- structlog used instead of stdlib logging for structured kwargs on log calls
- `type: ignore` comments kept only where actually needed: `datasets` (no py.typed), `evaluate` (no py.typed), `numpy` (has stubs — removed)
- `trainer.evaluate()` return typed as `dict[str, float]` directly (transformers now ships stubs)
- `tokenize` inner function annotated `# type: ignore[no-any-return]` for HF tokenizer Any return

## Tests
```
tests/unit/test_classifier_loads.py: 6/6 passed
ruff: clean
mypy: clean (3 source files)
```
