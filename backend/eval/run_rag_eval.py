#!/usr/bin/env python
"""RAG evaluation runner: validate golden set against thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent
EVAL_DIR = REPO_ROOT / "backend" / "eval"
THRESHOLDS_PATH = EVAL_DIR / "eval_thresholds.yaml"
GOLDEN_SET_PATH = EVAL_DIR / "golden_rag.jsonl"
EVAL_REPORT_PATH = REPO_ROOT / "ml" / "artifacts" / "eval_report.json"


def load_thresholds() -> dict:
    """Load eval thresholds from YAML."""
    with open(THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)


def load_golden_set() -> list[dict]:
    """Load golden RAG set from JSONL."""
    qa_pairs = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    return qa_pairs


def run_evaluation() -> int:
    """Validate RAG golden set against thresholds.

    WED Milestone: This is a placeholder. Full eval happens when:
    1. Corpus is indexed (ingest.py complete)
    2. Multi-query expander is tested
    3. Hybrid retriever returns results
    4. RAGAS metrics are computed

    For now, this validates golden set structure and threshold config.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    print("\n🧪 RAG Eval Runner (WED Milestone)")
    print(f"   Thresholds: {THRESHOLDS_PATH}")
    print(f"   Golden set: {GOLDEN_SET_PATH}")

    # Load data
    thresholds = load_thresholds()
    golden_set = load_golden_set()

    print(f"\n📊 Golden set size: {len(golden_set)} Q/A pairs")
    print(f"   Thresholds: {thresholds.get('rag', {})}")

    # Validate golden set structure
    print("\n✅ Validating golden set structure:")
    required_fields = {"id", "question", "ideal_answer", "ground_truth_chunks"}
    for qa in golden_set:
        if not all(f in qa for f in required_fields):
            print(f"   ✗ Missing fields in {qa.get('id')}: {required_fields - set(qa.keys())}")
            return 1

    print(f"   ✓ All {len(golden_set)} Q/A pairs have required fields")

    # Validate threshold config
    rag_config = thresholds.get("rag", {})
    required_thresholds = {
        "faithfulness",
        "answer_relevancy",
        "hit_at_5",
    }
    if not all(k in rag_config for k in required_thresholds):
        print(f"   ✗ Missing thresholds: {required_thresholds - set(rag_config.keys())}")
        return 1

    print(f"   ✓ Threshold config valid:")
    for key, val in rag_config.items():
        if key != "rationale":
            print(f"     - {key}: {val}")

    print("\n✨ RAG infrastructure ready for WED steps.")
    print("   Next: Corpus ingestion → Retrieval testing → Eval computation")
    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation())
