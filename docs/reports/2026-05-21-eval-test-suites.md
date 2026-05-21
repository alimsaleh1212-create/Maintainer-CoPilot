# Step Report: Eval Test Suites

**Date:** 2026-05-21
**Branch:** `feature/foundations-skeleton`

## What was built

- **`tests/eval/test_classification_thresholds.py`** — Loads `eval_report.json` and asserts the winning model's `macro_f1` and `per_class_f1` clear the committed thresholds. Also checks class completeness (`bug/feature/support`), no-zero threshold guard, and non-trivial sample size (≥50).
- **`tests/eval/test_rag_thresholds.py`** — Validates `eval_thresholds.yaml` config (all three thresholds present, none zero, faithfulness ≥0.70), golden set structure (≥20 items, required fields, unique IDs, non-empty `ground_truth_chunks`), and `run_rag_eval.py` exits 0.

## Tests written

| File | Count | Key assertions |
|------|-------|----------------|
| `test_classification_thresholds.py` | 5 | macro_f1≥0.75, per_class≥0.55, all 3 classes, no-zero guard, sample≥50 |
| `test_rag_thresholds.py` | 8 | All required thresholds, faithfulness≥0.70, golden ≥20 items, fields, IDs unique, runner exits 0 |

**Total: 13 new eval tests. Full suite: 142 passing (23 deselected non-eval marks).**

## Checks run

```
uv run ruff check . → All checks passed!
uv run pytest -q → 142 passed, 23 deselected in 3.42s
uv run pytest tests/eval/ -v -m eval → 13 passed in 0.16s
```

## What's next

Run the full acceptance checklist (items 1–6 of the 13-item list): fresh-clone smoke, static checks, secret hygiene, unit+integration, eval suites, and redaction proof.
