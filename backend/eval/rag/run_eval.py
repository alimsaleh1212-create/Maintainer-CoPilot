#!/usr/bin/env python
"""RAG evaluation runner: validate golden set against thresholds.

When the full retrieval pipeline is integrated (corpus ingested + RAGAS metrics
computed) this runner will assert actual faithfulness / answer_relevancy / Hit@5
values.  For now it validates golden set structure and threshold config.

Run inside the api container:
    docker exec docker-api-1 python eval/rag/run_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
_EVAL_DIR = Path(__file__).parent
_THRESHOLDS_PATH = _EVAL_DIR / "thresholds.yaml"
_GOLDEN_SET_PATH = _EVAL_DIR / "golden_set.jsonl"
_ABLATION_REPORT_PATH = _EVAL_DIR / "ablation_results.md"

_REQUIRED_GOLDEN_FIELDS = {"id", "question", "ideal_answer", "ground_truth_chunks"}


def load_thresholds() -> dict:
    """Load RAG thresholds from YAML.

    Returns:
        Dict of threshold name → value.
    """
    with open(_THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def load_golden_set() -> list[dict]:
    """Load golden RAG Q/A pairs from JSONL.

    Returns:
        List of dicts with id, question, ideal_answer, ground_truth_chunks.
    """
    qa_pairs = []
    with open(_GOLDEN_SET_PATH) as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    return qa_pairs


def run_evaluation() -> int:
    """Validate RAG golden set and threshold config.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    print("\n🧪 RAG Eval Runner")
    print(f"   Thresholds:      {_THRESHOLDS_PATH}")
    print(f"   Golden set:      {_GOLDEN_SET_PATH}")
    print(f"   Ablation report: {_ABLATION_REPORT_PATH}")

    thresholds = load_thresholds()
    golden_set = load_golden_set()

    print(f"\n📊 Golden set size: {len(golden_set)} Q/A pairs")

    # Validate golden set structure
    print("\n✅ Validating golden set structure:")
    for qa in golden_set:
        missing = _REQUIRED_GOLDEN_FIELDS - set(qa.keys())
        if missing:
            print(f"   ✗ {qa.get('id')}: missing fields {missing}")
            return 1
    print(f"   ✓ All {len(golden_set)} Q/A pairs have required fields")

    # Validate threshold config
    print("\n✅ Validating threshold config:")
    required_keys = {"faithfulness", "answer_relevancy", "hit_at_5"}
    missing_keys = required_keys - set(thresholds.keys())
    if missing_keys:
        print(f"   ✗ Missing thresholds: {missing_keys}")
        return 1

    for key in sorted(required_keys):
        val = thresholds[key]
        ok = val > 0
        print(f"   {'✓' if ok else '✗'} {key}: {val}")
        if not ok:
            return 1

    # Check ablation report is present
    if _ABLATION_REPORT_PATH.exists():
        print(f"\n📑 Ablation report present: {_ABLATION_REPORT_PATH.name}")
    else:
        print("\n⚠  Ablation report not found — run scripts/ablate_rag.py first")

    print("\n✨ RAG eval infrastructure ready.")
    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation())
