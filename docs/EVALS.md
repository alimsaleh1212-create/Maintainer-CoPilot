# Evaluations — Maintainer's Copilot

## Classification eval

### Golden set

`backend/eval/golden_classification.jsonl` — 25 hand-curated issues from Project-MONAI/MONAI that are:
- NOT in the training or test split (held out)
- Prioritized for edge cases: multi-label ambiguity, very short titles, non-English code snippets, old vs new format issues

### Three-way model comparison (on 424-issue held-out test set)

| Model | Macro-F1 | Per-class F1 (bug/feature/support) | p50 latency | Cost/1000 |
|---|---|---|---|---|
| DistilBERT (fine-tuned) | **0.764** | 0.863 / 0.847 / 0.583 | 45ms | ~$0 |
| TF-IDF + LogReg (baseline) | 0.723 | 0.791 / 0.812 / 0.567 | 2ms | ~$0 |
| Gemini 2.5 Flash (few-shot) | 0.644 | 0.723 / 0.689 / 0.521 | 1.2s | ~$0.15 |

**Winner: DistilBERT** — best macro-F1, zero inference cost, deterministic.  
Threshold: `macro_f1 >= 0.75` (committed in `eval_thresholds.yaml`).

Note: `support` class has lower F1 because it merges `documentation` + `questions` — the boundary is genuinely ambiguous (some questions are feature requests in disguise).

### Hand-label vs golden set agreement

5 issues from the golden set were independently classified by both human rater and the eval harness. Agreement: **4/5 (80%)** on the 3-class problem. The disagreement was on a `support` issue that was borderline `feature` — both labels are defensible.

## RAG eval

### Golden set

`backend/eval/golden_rag.jsonl` — 25 question/ideal-answer/ground-truth-chunks triples from MONAI docs. Questions cover:
- Installation and environment setup
- API usage (transforms, datasets, networks)
- Error diagnosis ("why does X fail when Y")
- Cross-topic synthesis ("how does MONAI handle X compared to Y")

### RAGAS metrics (on golden set)

| Metric | Score | Threshold |
|---|---|---|
| Faithfulness | 0.87 | ≥ 0.85 |
| Answer relevancy | 0.83 | ≥ 0.80 |
| Context precision | 0.79 | ≥ 0.75 |
| Context recall | 0.76 | ≥ 0.70 |

### Retrieval metrics

| Metric | Score | Threshold |
|---|---|---|
| Hit@5 | 0.84 | ≥ 0.70 |
| MRR@10 | 0.76 | ≥ 0.65 |

### Hand-label judge agreement

5 of 25 RAG responses were hand-labeled for faithfulness (1 = faithful, 0 = hallucinated/unfaithful). Agreement with RAGAS judge: **5/5 (100%)** on this sample. RAGAS faithfulness scores correlated well with human judgment.

## How to run evals

```bash
cd backend

# Classification eval (runs all three models on test + golden set)
uv run python eval/run_classification_eval.py

# RAG eval (requires running API stack for embeddings)
uv run python eval/run_rag_eval.py

# Both via pytest (slow tests, tagged @pytest.mark.eval)
uv run pytest -q -m eval
```

## Threshold policy

Thresholds are committed in `backend/eval/eval_thresholds.yaml`. The API refuses to boot if any threshold is at zero/disabled. CI runs the eval suite on every push to main; thresholds gate merges.

To lower a threshold (e.g., after a model change degrades one metric), open a dedicated PR with:
1. The new `eval_report.json` diff attached
2. A written justification (why the regression is acceptable)
3. The threshold change in `eval_thresholds.yaml`

Never lower a threshold without a PR.
