# MON Step 9 — Dataset Fetch, Label Mapping, and Splits

**Date:** 2026-05-19
**Branch:** feature/foundations-skeleton

## What was built

- **`backend/app/domain/issue.py`** — `RawIssue` and `LabeledIssue` Pydantic models. `LABEL_MAP` (4 GitHub labels → 3 canonical classes). `resolve_label()` returns None on empty, unknown-only, or ambiguous inputs. `build_labeled_issue()` assembles classifier text as `title + "\n\n" + body` and resolves the label in one step.
- **`backend/scripts/fetch_issues.py`** — Paginates GitHub REST API (`/repos/Project-MONAI/MONAI/issues?state=closed`), filters PRs and non-target labels, saves to `data/raw_issues.jsonl`. Respects rate limits with 0.5s inter-page delay; shows warning if `GITHUB_TOKEN` is unset.
- **`backend/scripts/audit_labels.py`** — Prints per-label and per-class counts from the raw JSONL to confirm the 3-class merge decision numbers.
- **`backend/scripts/build_splits.py`** — Time-aware stratified split (seed=42): test = most recent 15% chronologically (no leakage), then stratified 15/85 val/train from the remainder. Writes `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`, `data/splits_summary.json`. An `assert` guards the temporal invariant.

## Tests written

- **`tests/unit/test_label_mapping.py`** — 17 tests across 3 classes:
  - `TestResolveLabel` — each of the 4 GitHub labels, the documentation+questions merge, unknown labels, empty list, extra non-target labels ignored, two ambiguous cases return None.
  - `TestClassCoverage` — all 3 canonical classes are reachable from LABEL_MAP; CLASS_TO_IDX covers exactly CLASS_NAMES.
  - `TestBuildLabeledIssue` — happy path, None body, missing closed_at, no target label, ambiguous label.

## Checks run

- `uv run ruff check .` → all checks passed
- `uv run mypy app/` → no issues (22 source files)
- `uv run pytest -q` → 43 passed, 1 deselected

## What's next

MON Step 10 — Start fine-tuning DistilBERT 3-class classifier (`app/ml/train_classifier.py`). Dataset fetch must be run first: `GITHUB_TOKEN=<token> uv run python scripts/fetch_issues.py && uv run python scripts/build_splits.py`.
