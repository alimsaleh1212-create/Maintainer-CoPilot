#!/usr/bin/env python
"""Classification eval runner: validate golden set against thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score, f1_score

# ── Paths ────────────────────────────────────────────────────────────────────
_EVAL_DIR = Path(__file__).parent
_THRESHOLDS_PATH = _EVAL_DIR / "thresholds.yaml"
_GOLDEN_SET_PATH = _EVAL_DIR / "golden_set.jsonl"
_EVAL_REPORT_PATH = _EVAL_DIR.parent.parent.parent / "ml" / "artifacts" / "eval_report.json"


def load_thresholds() -> dict:
    """Load classification thresholds from YAML.

    Returns:
        Dict of threshold name → value.
    """
    with open(_THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def load_golden_set() -> list[dict]:
    """Load golden classification set from JSONL.

    Returns:
        List of issue dicts with ``label`` field.
    """
    issues = []
    with open(_GOLDEN_SET_PATH) as f:
        for line in f:
            if line.strip():
                issues.append(json.loads(line))
    return issues


def load_eval_report() -> dict:
    """Load the eval report with trained model metrics.

    Returns:
        Dict parsed from eval_report.json.

    Raises:
        FileNotFoundError: If eval_report.json does not exist.
    """
    with open(_EVAL_REPORT_PATH) as f:
        return json.load(f)  # type: ignore[no-any-return]


def run_evaluation() -> int:
    """Validate golden set against thresholds.

    Returns:
        0 if all thresholds pass, 1 otherwise.
    """
    print("\n🧪 Classification Eval Runner")
    print(f"   Thresholds: {_THRESHOLDS_PATH}")
    print(f"   Golden set: {_GOLDEN_SET_PATH}")
    print(f"   Eval report: {_EVAL_REPORT_PATH}")

    thresholds = load_thresholds()
    golden_set = load_golden_set()
    eval_report = load_eval_report()

    print(f"\n📊 Golden set size: {len(golden_set)} issues")
    print(f"📈 Eval report winner: {eval_report['winner']['model']}")

    required_macro_f1 = thresholds.get("macro_f1", 0.0)
    required_per_class_f1_min = thresholds.get("per_class_f1_min", 0.0)

    winner_metrics = eval_report["winner"]["metrics"]
    actual_macro_f1 = winner_metrics["macro_f1"]
    actual_per_class_f1 = winner_metrics["per_class_f1"]

    print("\n✅ Classification Thresholds:")
    print(f"   macro_f1: {actual_macro_f1:.4f} >= {required_macro_f1:.4f}? ", end="")
    macro_pass = actual_macro_f1 >= required_macro_f1
    print("✓" if macro_pass else "✗")

    for cls, f1 in actual_per_class_f1.items():
        print(f"   {cls}: {f1:.4f} >= {required_per_class_f1_min:.4f}? ", end="")
        print("✓" if f1 >= required_per_class_f1_min else "✗")

    all_pass = macro_pass and all(
        f1 >= required_per_class_f1_min for f1 in actual_per_class_f1.values()
    )

    if all_pass:
        print("\n✨ All thresholds PASS. Eval gate OK.")
        return 0
    print("\n❌ Thresholds FAILED. Eval gate BLOCKED.")
    return 1


if __name__ == "__main__":
    sys.exit(run_evaluation())
