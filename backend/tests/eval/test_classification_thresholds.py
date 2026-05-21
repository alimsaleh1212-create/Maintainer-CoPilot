"""Classification threshold smoke tests.

CLAUDE.md rule: 'eval_thresholds.yaml values gate merge — zero is refuse-to-boot.'

Loads eval_report.json and checks the winning model's macro-F1 and per-class F1
against the committed thresholds.  Tagged @pytest.mark.eval so local runs skip
by default; CI always runs them.

Run manually:
    uv run pytest tests/eval/test_classification_thresholds.py -v -m eval
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_ROOT = Path(__file__).parent.parent.parent
_EVAL_DIR = _BACKEND_ROOT / "eval"
_THRESHOLDS_PATH = _EVAL_DIR / "eval_thresholds.yaml"
_EVAL_REPORT_PATH = _BACKEND_ROOT.parent / "ml" / "artifacts" / "eval_report.json"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def thresholds() -> dict:
    """Load committed classification thresholds."""
    with open(_THRESHOLDS_PATH) as f:
        data: dict = yaml.safe_load(f)
    return data["classification"]


@pytest.fixture(scope="module")
def winner_metrics() -> dict:
    """Load the winning model's metrics from eval_report.json."""
    if not _EVAL_REPORT_PATH.exists():
        pytest.skip(f"eval_report.json not found at {_EVAL_REPORT_PATH}")
    with open(_EVAL_REPORT_PATH) as f:
        report: dict = json.load(f)
    return report["winner"]["metrics"]


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.eval
class TestClassificationThresholds:
    def test_macro_f1_meets_threshold(
        self,
        thresholds: dict,
        winner_metrics: dict,
    ) -> None:
        """CRITICAL PATH: macro-F1 on test split must clear the committed floor."""
        required = thresholds["macro_f1"]
        actual = winner_metrics["macro_f1"]

        assert actual >= required, (
            f"macro_f1 {actual:.4f} is below committed threshold {required:.4f}. "
            "Either improve the model or update the threshold in a dedicated PR "
            "with the eval_report diff."
        )

    def test_per_class_f1_meets_minimum(
        self,
        thresholds: dict,
        winner_metrics: dict,
    ) -> None:
        """Each class F1 must clear per_class_f1_min — guards against class collapse."""
        required_min = thresholds["per_class_f1_min"]
        per_class: dict = winner_metrics["per_class_f1"]

        failing = {cls: f1 for cls, f1 in per_class.items() if f1 < required_min}
        assert not failing, (
            f"Classes below per_class_f1_min={required_min:.4f}: {failing}. "
            "Check training data balance for under-performing classes."
        )

    def test_all_three_classes_present_in_report(
        self,
        winner_metrics: dict,
    ) -> None:
        """eval_report must cover all three canonical classes."""
        per_class = winner_metrics["per_class_f1"]
        assert set(per_class.keys()) == {"bug", "feature", "support"}, (
            f"Unexpected class set in eval_report: {set(per_class.keys())}. "
            "Class mapping must be: bug / feature / support."
        )

    def test_threshold_config_has_no_zero_values(
        self,
        thresholds: dict,
    ) -> None:
        """Refuse-to-boot rule: no threshold may be zero or disabled."""
        for key, val in thresholds.items():
            if key == "rationale":
                continue
            assert val > 0, (
                f"Threshold '{key}' is {val!r}. "
                "Zero thresholds are forbidden — they disable the gate."
            )

    def test_eval_report_sample_size_is_non_trivial(
        self,
        winner_metrics: dict,
    ) -> None:
        """Test split must have enough samples to yield meaningful per-class F1."""
        sample_size: int = winner_metrics.get("eval_sample_size", 0)
        assert sample_size >= 50, (
            f"eval_sample_size={sample_size} is too small for reliable F1. "
            "Need at least 50 test examples to report per-class F1 with confidence."
        )
