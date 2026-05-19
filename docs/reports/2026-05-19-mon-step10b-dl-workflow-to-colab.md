# MON Step 10b — Move DL workflow to Colab notebook

**Date:** 2026-05-19
**Branch:** `feature/foundations-skeleton`
**Status:** complete — ruff clean, mypy clean (24 src files), 35/35 tests pass

## Why this refactor
Training is offline; inference is online. The cleanest boundary is `model_card.json` — the only artefact that crosses the wall. Pulling the training pipeline into the repo created Python-import coupling between Colab and the backend, which the user (rightly) flagged as risky. This step moves *all* training-side logic into a single self-contained Colab notebook and trims the backend to inference-only.

## What changed

### Added
- `backend/notebooks/train_classifier_colab.ipynb` — 22-cell end-to-end DL workflow: setup → mount Drive → fetch MONAI issues → 4→3 label merge with shown counts → time-aware stratified split → tokenize → train (DistilBERT + EarlyStopping + fp16) → write `model_card.json` with SHA-256 → zip+copy to Drive → **round-trip verify** (re-runs the exact checks `load_classifier()` performs at boot)

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
- `tests/unit/test_classifier_loads.py` — expanded from 6 → 9 cases, adding:
  - `test_model_card_without_classes_raises` — refuses to boot if card has no `classes` field
  - `test_class_set_drift_raises` — 4-class card vs 3-class backend → refuse
  - `test_class_order_drift_raises` — same classes reordered → refuse (index mapping changes)

## Checks
```
ruff check .  → all clean (notebooks excluded)
mypy app/     → 24 source files, no issues
pytest -q     → 35 passed, 1 deselected
```

## What's next
TUE: classical ML baseline + LLM baseline + three-way comparison + `/classify` endpoint + NER + summarizer. All three baselines are evaluated in the same notebook (or a sibling eval notebook); the backend just loads the winner.
