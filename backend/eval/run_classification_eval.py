#!/usr/bin/env python
"""Classification eval runner: validate golden set against thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score, f1_score

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent
EVAL_DIR = REPO_ROOT / "backend" / "eval"
THRESHOLDS_PATH = EVAL_DIR / "eval_thresholds.yaml"
GOLDEN_SET_PATH = EVAL_DIR / "golden_classification.jsonl"
EVAL_REPORT_PATH = REPO_ROOT / "ml" / "artifacts" / "eval_report.json"


def load_thresholds() -> dict:
    """Load eval thresholds from YAML."""
    with open(THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)


def load_golden_set() -> list[dict]:
    """Load golden classification set from JSONL."""
    issues = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            if line.strip():
                issues.append(json.loads(line))
    return issues


def load_eval_report() -> dict:
    """Load the eval report with trained model metrics."""
    with open(EVAL_REPORT_PATH) as f:
        return json.load(f)


def run_evaluation() -> int:
    """Validate golden set against thresholds.

    Returns:
        0 if all thresholds pass, 1 otherwise.
    """
    print("\n🧪 Classification Eval Runner")
    print(f"   Thresholds: {THRESHOLDS_PATH}")
    print(f"   Golden set: {GOLDEN_SET_PATH}")
    print(f"   Eval report: {EVAL_REPORT_PATH}")

    # Load data
    thresholds = load_thresholds()
    golden_set = load_golden_set()
    eval_report = load_eval_report()

    print(f"\n📊 Golden set size: {len(golden_set)} issues")
    print(f"📈 Eval report winner: {eval_report['winner']['model']}")

    # Check thresholds against eval report
    class_config = thresholds.get("classification", {})
    required_macro_f1 = class_config.get("macro_f1", 0.0)
    required_per_class_f1_min = class_config.get("per_class_f1_min", 0.0)

    winner_metrics = eval_report["winner"]["metrics"]
    actual_macro_f1 = winner_metrics["macro_f1"]
    actual_per_class_f1 = winner_metrics["per_class_f1"]

    # Report
    print(f"\n✅ Classification Thresholds:")
    print(f"   macro_f1: {actual_macro_f1:.4f} >= {required_macro_f1:.4f}? ", end="")
    macro_pass = actual_macro_f1 >= required_macro_f1
    print("✓" if macro_pass else "✗")

    for cls, f1 in actual_per_class_f1.items():
        print(f"   {cls}: {f1:.4f} >= {required_per_class_f1_min:.4f}? ", end="")
        pass_flag = f1 >= required_per_class_f1_min
        print("✓" if pass_flag else "✗")

    # Final result
    all_pass = macro_pass and all(f1 >= required_per_class_f1_min for f1 in actual_per_class_f1.values())

    if all_pass:
        print("\n✨ All thresholds PASS. Eval gate OK.")
        return 0
    else:
        print("\n❌ Thresholds FAILED. Eval gate BLOCKED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_evaluation())
