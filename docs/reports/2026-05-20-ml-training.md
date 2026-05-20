# 2026-05-20 — ML Training Pipeline

## What was built

- **`ml/`** — separate training directory (outside `backend/`) with its own `pyproject.toml`, `uv` venv (torch+cu124), `data/`, `artifacts/`, `notebooks/`.
- **`ml/notebooks/train_local.ipynb`** — 27-cell local GPU notebook: fetch with cache-check, LLM auto-labeling, stratified split, DistilBERT fine-tune, TF-IDF baseline, Gemini few-shot baseline, three-way comparison, round-trip verify.
- **`ml/notebooks/train_pipeline.py`** — script mirror of the notebook; runs end-to-end with `uv run`.
- **`ml/data/llm_labeled.jsonl`** — 1 457 Gemini-2.5-Flash labels for previously unlabeled issues (committed; expensive to reproduce).
- **`ml/artifacts/classifier/best/`** — DistilBERT weights + tokenizer + `model_card.json` with SHA-256.
- **`ml/artifacts/classical_ml/pipeline.joblib`** — TF-IDF + LogReg sklearn pipeline (compress=3).
- **`ml/artifacts/eval_report.json`** — three-way comparison; DistilBERT declared winner.

## Dataset expansion

| Source | Count |
|---|---|
| GitHub label mapping | 1 471 |
| Gemini-2.5-Flash auto-label | 1 457 (0 errors) |
| Labeled "other" (excluded) | 107 |
| **Training-eligible total** | **2 826** |

## Training results (test split, n=424, time-aware)

| Model | Accuracy | Macro-F1 | Latency | $/1K |
|---|---|---|---|---|
| **DistilBERT** | **0.823** | **0.764** | 5.6 ms | $0.00 |
| TF-IDF + LogReg | 0.788 | 0.723 | 0.4 ms | $0.00 |
| Gemini-2.5-Flash | 0.770 | 0.644 | 703 ms | $0.18 |

Per-class F1 (DistilBERT): bug=0.863, feature=0.847, support=0.583.  
`support` is below the 0.70 per-class threshold — driven by class imbalance (support=426 vs bug=1012, feature=1388) and harder decision boundary.

## Tests written

None at this step — training pipeline is offline (not imported by backend). The backend `/classify` endpoint integration test is scoped to the Tuesday branch.

## Checks run

- Pipeline ran end-to-end with 0 errors on RTX 4050 (CUDA 12.4, torch 2.6.0+cu124)
- `eval_report.json` written and committed
- Model SHA-256 stored in `model_card.json`

## What's next

Tuesday branch `feature/dl-track-classifiers`: wire DistilBERT weights into `backend/model_server/`, add `/classify` endpoint, write classification golden set, meet per-class F1 thresholds (may need class-weighted loss or oversampling for `support`).
