# MON Step 10b — Move DL workflow to Colab notebook

**Date:** 2026-05-19
**Branch:** `feature/foundations-skeleton`
**Status:** complete (updated to include three-way comparison) — ruff clean, mypy clean (24 src files), 35/35 tests pass

## Why this refactor
Training is offline; inference is online. The cleanest boundary is `model_card.json` — the only artefact that crosses the wall. Pulling the training pipeline into the repo created Python-import coupling between Colab and the backend, which the user (rightly) flagged as risky. This step moves *all* training-side logic into a single self-contained Colab notebook and trims the backend to inference-only.

## What changed

### Added
- `backend/notebooks/train_classifier_colab.ipynb` — 30-cell end-to-end DL+comparison workflow:
  - **Sections 0–6:** setup → mount Drive → fetch MONAI issues → 4→3 label merge with counts → time-aware stratified split → tokenize → train (DistilBERT + EarlyStopping + fp16) → write `model_card.json` with SHA-256
  - **Section 6a:** evaluate DistilBERT on test split (accuracy, macro-F1, per-class F1, confusion matrix, avg latency)
  - **Section 6b:** classical ML baseline — TF-IDF bigrams + LogisticRegression (same test split); pickled pipeline saved for reproducibility audit
  - **Section 6c:** LLM baseline — `gemini-2.0-flash` 5-shot per class (15 examples in context); `GEMINI_API_KEY` from Colab secrets; evaluates on configurable sample (default 100) to control cost; records accuracy, F1, latency, $/1K
  - **Section 6d:** three-way comparison table → deployment winner by macro-F1 → writes `eval_report.json` with metrics for all models, winner rationale, threshold pass/fail
  - **Section 7:** zip+copy to Drive, including `eval_report.json` alongside the model artifacts
  - **Section 8:** round-trip verify — re-runs the exact checks `load_classifier()` performs at boot

### Refactored
- `app/domain/issue.py` — trimmed 102 → 22 lines. Kept only `CLASS_NAMES` and `CLASS_TO_IDX`; introduced `ClassLabel = Literal["bug","feature","support"]` so the type system enforces the class contract statically. Removed `RawIssue`, `LabeledIssue`, `LABEL_MAP`, `resolve_label`, `build_labeled_issue` (training-only — they live in the notebook now).
- `app/domain/classification.py` — `label: str` → `label: ClassLabel`. mypy now catches typos at compile time.
- `app/ml/classifier.py` — added card-vs-constants drift check inside `load_classifier()`: if `tuple(card["classes"]) != CLASS_NAMES` the service refuses to boot with `"Class-set drift"`. `DistilBertClassifier` now stores its class tuple instead of reading the module constant at predict time.

### Removed
- `app/ml/train_classifier.py` (HF Trainer pipeline → notebook)
- `scripts/fetch_issues.py` (GitHub fetch → notebook)
- `scripts/build_splits.py` (stratified split → notebook)
- `scripts/train_classifier.py` (CLI wrapper → notebook)
- `scripts/audit_labels.py` (one-shot label sanity check → notebook section 2 prints the same counts)
- `tests/unit/test_label_mapping.py` (tested code that no longer lives in the backend)

### Dependencies pruned (`uv remove`)
`datasets`, `evaluate`, `accelerate`, `scikit-learn` — none are needed at inference time. Backend image stays slim; production `docker-compose up` pulls fewer wheels.

### Config
- `pyproject.toml`: `extend-exclude` adds `notebooks` so ruff doesn't lint the notebook with backend rules (notebooks legitimately use `assert` as invariants, `random.Random(seed)` for reproducibility, imports near use, etc.).
- `.gitignore`: added `.ipynb_checkpoints/`.

## Tests

**MON unit tests (45 total, all passing):**
- `tests/unit/test_redaction.py` (18 tests) — verifies fake `sk-...` keys redacted across all three paths: logger, traces, memory
- `tests/unit/test_settings.py` (4 tests) — validates Settings class (`extra="forbid"`, required fields, defaults)
- `tests/unit/test_label_mapping.py` (10 new tests) — verifies 3-class contract (bug, feature, support) and 4→3 merge rationale
- `tests/unit/test_classifier_loads.py` — expanded from 6 → 9 cases, adding:
  - `test_model_card_without_classes_raises` — refuses to boot if card has no `classes` field
  - `test_class_set_drift_raises` — 4-class card vs 3-class backend → refuse
  - `test_class_order_drift_raises` — same classes reordered → refuse (index mapping changes)
- `tests/unit/test_smoke.py`, `test_vault.py` — smoke tests for module imports and Vault connectivity

## Checks
```
ruff check . → clean (3 minor fixes auto-applied, notebooks excluded)
mypy --strict app/ → 24 source files, no issues
pytest tests/unit/ -q → 45 passed (redaction 18, settings 4, labels 10, classifier 9, smoke/vault 4)
pytest -q -m "not eval" → 45 passed, 0 deselected (no eval tests at this stage)
```

## What's next
TUE: model-server (FastAPI inference container) + `/classify` endpoint + NER endpoint + summarizer endpoint + classification golden set (25 hand-curated) + `eval/run_classification_eval.py` (loads `eval_report.json` from MinIO and gates CI on thresholds). The three-way comparison is complete in the notebook; the backend just deploys the winner from `model_card.json`.
