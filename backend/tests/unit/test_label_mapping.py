"""Unit tests for canonical label contract in app.domain.issue.

The mapping from GitHub's 4-label schema to our 3-class problem is documented
in docs/DECISIONS.md and implemented in the Colab training notebook. This test
verifies that the backend's inference-time constants (CLASS_NAMES, CLASS_TO_IDX)
match the notebook's training-time schema exactly.

Per CLAUDE.md: load_classifier() refuses to boot if model_card["classes"] !=
CLASS_NAMES, so these constants MUST be immutable and correct.
"""

from __future__ import annotations

from app.domain.issue import CLASS_NAMES, CLASS_TO_IDX, ClassLabel


class TestCanonicalLabels:
    """Verify the 3-class label contract is correctly defined."""

    def test_class_names_tuple_structure(self) -> None:
        """CLASS_NAMES is a tuple with exactly the three expected classes."""
        # Arrange & Act: Already defined as module constant
        # Assert
        assert isinstance(CLASS_NAMES, tuple)
        assert len(CLASS_NAMES) == 3
        assert set(CLASS_NAMES) == {"bug", "feature", "support"}

    def test_class_names_order_matters(self) -> None:
        """CLASS_NAMES order is: bug (0), feature (1), support (2)."""
        # Assert: Order is critical for model output indexing
        assert CLASS_NAMES[0] == "bug"
        assert CLASS_NAMES[1] == "feature"
        assert CLASS_NAMES[2] == "support"

    def test_class_to_idx_complete_mapping(self) -> None:
        """CLASS_TO_IDX maps all three classes to unique indices 0, 1, 2."""
        # Arrange & Act: Already defined as module constant
        # Assert
        assert CLASS_TO_IDX["bug"] == 0
        assert CLASS_TO_IDX["feature"] == 1
        assert CLASS_TO_IDX["support"] == 2

    def test_class_to_idx_keys_match_class_names(self) -> None:
        """CLASS_TO_IDX keys exactly match CLASS_NAMES."""
        # Assert: Consistency check
        assert set(CLASS_TO_IDX.keys()) == set(CLASS_NAMES)

    def test_class_to_idx_values_are_sequential(self) -> None:
        """CLASS_TO_IDX values are 0, 1, 2 (no gaps, no duplicates)."""
        # Assert
        values = sorted(CLASS_TO_IDX.values())
        assert values == [0, 1, 2]

    def test_class_to_idx_inverse_mapping(self) -> None:
        """Can recover CLASS_NAMES from CLASS_TO_IDX."""
        # Arrange
        idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

        # Assert
        assert idx_to_class[0] == "bug"
        assert idx_to_class[1] == "feature"
        assert idx_to_class[2] == "support"


class TestClassLabelType:
    """Verify ClassLabel type enforces the three-class contract at type-check time."""

    def test_class_label_accepts_valid_values(self) -> None:
        """ClassLabel type hints accept the three canonical values."""
        # Note: This test is primarily for mypy verification.
        # At runtime, Literal doesn't enforce the constraint without Pydantic.
        # We use it to document the contract.

        def classify(label: ClassLabel) -> str:
            return f"Classified as: {label}"

        # These should all be type-valid (would pass mypy)
        assert classify("bug") == "Classified as: bug"
        assert classify("feature") == "Classified as: feature"
        assert classify("support") == "Classified as: support"


class TestLabelMapping3From4:
    """Document the 3-class merge from the 4-label GitHub schema.

    Per DECISIONS.md (documented in Colab notebook Section 2):
    - bug (337 issues) → bug
    - feature_request (535 issues) → feature
    - documentation (28 issues) + questions (250 issues) → support

    The merge is justified: 28-example class yields ~5 test samples
    (statistical noise). Routing decision is identical for both.
    """

    def test_canonical_class_count_three_not_four(self) -> None:
        """Backend enforces 3-class problem, not the original 4."""
        # Assert: The brief listed 4 classes, but data-driven merge to 3.
        assert len(CLASS_NAMES) == 3

    def test_documentation_and_questions_both_map_to_support(self) -> None:
        """The merge rationale: both GitHub labels → support class.

        This test documents the decision: questions and documentation
        have identical maintainer routing (point to docs/FAQ), so merging
        them is semantically valid and yields a balanced 3-class problem.
        """
        # Assert: "support" class exists and is the merge target
        assert "support" in CLASS_NAMES

    def test_bug_and_feature_unchanged(self) -> None:
        """bug and feature_request map directly (no merge)."""
        # Assert
        assert "bug" in CLASS_NAMES
        assert "feature" in CLASS_NAMES
        # (feature_request GitHub label → feature class)
