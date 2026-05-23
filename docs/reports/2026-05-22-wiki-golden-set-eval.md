# Wiki Golden Set — RAG Evaluation Report
**Date:** 2026-05-22  
**Branch:** feature/foundations-skeleton

## What was built

- **`backend/eval/rag/wiki_golden_set.jsonl`** — 25 Q/A pairs grounded in actual MONAI wiki corpus chunks across five files: FAQ, Developer-Guide-Transforms, Overview, Evaluation-metrics-task-force, and MONAI_Preprocessors_and_Transforms_Design_Discussion. Ground truth uses real section-header substrings that appear verbatim in retrieved chunk text (e.g., `"Transforms > shape convention"`, `"Project MONAI > Vision"`).
- **`backend/eval/rag/run_eval.py`** — Added `--golden-set <path>` CLI argument so any JSONL golden set can be evaluated; results are saved alongside the golden set as `<name>_results.json` rather than overwriting `last_results.json`.
- **`backend/eval/rag/wiki_golden_set_results.json`** — Evaluation output for the wiki golden set.

## Evaluation results

| Metric | Score | Threshold | Result |
|--------|-------|-----------|--------|
| Faithfulness | 0.893 | ≥ 0.80 | ✅ PASS |
| Answer relevancy | 0.751 | ≥ 0.75 | ✅ PASS |
| Hit@5 | 0.760 | ≥ 0.70 | ✅ PASS |

All three thresholds pass. This is a significant improvement over the issue-corpus golden set (faithfulness 0.783, answer_relevancy 0.436, Hit@5 0.640 — two of three failing). The wiki corpus contains structured documentation rather than specific bug reports, which produces more faithful and relevant answers.

## Debugging done this session

- **Embedding 501 error** (prior session): Switched RAGAS embeddings from Gemini OpenAI-compat endpoint (returns HTTP 501) to Ollama `nomic-embed-text`.
- **Hit@5 = 0.000** (prior session): Issue golden set used keyword labels as ground truth; fixed by building corpus-grounded excerpts.
- **Faithfulness 0.524→0.783** (prior session): Fixed system prompt to use injected Context block instead of re-calling `rag_search`.
- **Hit@5 = 0.200 on wiki set (this session)**: psql column-padding added trailing spaces to ground truth strings; fixed by using actual section-header text that appears verbatim in chunk headers.

## Usage

```bash
# Evaluate against wiki golden set (all RAGAS + retrieval metrics)
uv run --group eval python eval/rag/run_eval.py \
  --api-url http://localhost:8000 \
  --api-token <jwt> \
  --gemini-api-key <key> \
  --golden-set eval/rag/wiki_golden_set.jsonl

# Evaluate against original issue golden set (unchanged)
uv run --group eval python eval/rag/run_eval.py \
  --api-url http://localhost:8000 \
  --api-token <jwt> \
  --gemini-api-key <key>
```

## What's next

The RAG eval pipeline is complete with two golden sets. Next: wire up the full chatbot (`/chat` tool-calling) end-to-end and ensure cross-conversation memory recall works (Thursday tasks remaining).
