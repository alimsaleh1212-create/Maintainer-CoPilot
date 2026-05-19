"""Unit tests for the 4-label → 3-class mapping logic in app.domain.issue."""

from datetime import UTC, datetime

from app.domain.issue import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    LABEL_MAP,
    RawIssue,
    build_labeled_issue,
    resolve_label,
)


# ---------------------------------------------------------------------------
# resolve_label — basic mapping
# ---------------------------------------------------------------------------
class TestResolveLabel:
    def test_bug_maps_to_bug(self) -> None:
        assert resolve_label(["bug"]) == "bug"

    def test_feature_request_maps_to_feature(self) -> None:
        assert resolve_label(["feature request"]) == "feature"

    def test_documentation_maps_to_support(self) -> None:
        assert resolve_label(["documentation"]) == "support"

    def test_questions_maps_to_support(self) -> None:
        assert resolve_label(["questions"]) == "support"

    def test_documentation_and_questions_both_merge_to_support(self) -> None:
        # Two labels but both map to the same canonical class → unambiguous.
        assert resolve_label(["documentation", "questions"]) == "support"

    def test_unknown_label_returns_none(self) -> None:
        assert resolve_label(["needs-triage"]) is None

    def test_empty_labels_returns_none(self) -> None:
        assert resolve_label([]) is None

    def test_mixed_known_unknown_preserves_known(self) -> None:
        # Extra non-target labels should be ignored, not block resolution.
        assert resolve_label(["bug", "needs-triage", "confirmed"]) == "bug"

    def test_ambiguous_bug_and_feature_returns_none(self) -> None:
        assert resolve_label(["bug", "feature request"]) is None

    def test_ambiguous_bug_and_support_returns_none(self) -> None:
        assert resolve_label(["bug", "questions"]) is None


# ---------------------------------------------------------------------------
# resolve_label — class coverage (all 3 canonical classes reachable)
# ---------------------------------------------------------------------------
class TestClassCoverage:
    def test_all_class_names_are_reachable(self) -> None:
        reachable = {resolve_label([lbl]) for lbl in LABEL_MAP}
        assert reachable == set(CLASS_NAMES)

    def test_class_to_idx_covers_all_classes(self) -> None:
        assert set(CLASS_TO_IDX.keys()) == set(CLASS_NAMES)
        assert sorted(CLASS_TO_IDX.values()) == list(range(len(CLASS_NAMES)))


# ---------------------------------------------------------------------------
# build_labeled_issue — integration of resolve_label + text assembly
# ---------------------------------------------------------------------------
_NOW = datetime(2024, 6, 1, tzinfo=UTC)

_RAW_BUG = RawIssue(
    id=1,
    number=101,
    title="NullPointerException in predict()",
    body="When I call predict() with an empty tensor I get NPE.",
    labels=["bug", "confirmed"],
    created_at=_NOW,
    closed_at=_NOW,
)


class TestBuildLabeledIssue:
    def test_bug_issue_builds_correctly(self) -> None:
        result = build_labeled_issue(_RAW_BUG)
        assert result is not None
        assert result.label == "bug"
        assert result.label_idx == CLASS_TO_IDX["bug"]
        assert "NullPointerException" in result.text
        assert "empty tensor" in result.text

    def test_none_body_handled_gracefully(self) -> None:
        raw = RawIssue(
            id=2,
            number=102,
            title="Add support for DICOM export",
            body=None,
            labels=["feature request"],
            created_at=_NOW,
            closed_at=_NOW,
        )
        result = build_labeled_issue(raw)
        assert result is not None
        assert result.label == "feature"
        assert result.text == "Add support for DICOM export"

    def test_missing_closed_at_returns_none(self) -> None:
        raw = RawIssue(
            id=3,
            number=103,
            title="Open issue with no close date",
            labels=["bug"],
            created_at=_NOW,
            closed_at=None,
        )
        assert build_labeled_issue(raw) is None

    def test_no_target_label_returns_none(self) -> None:
        raw = RawIssue(
            id=4,
            number=104,
            title="Improve CI pipeline",
            labels=["ci", "infra"],
            created_at=_NOW,
            closed_at=_NOW,
        )
        assert build_labeled_issue(raw) is None

    def test_ambiguous_label_returns_none(self) -> None:
        raw = RawIssue(
            id=5,
            number=105,
            title="Bug or feature?",
            labels=["bug", "feature request"],
            created_at=_NOW,
            closed_at=_NOW,
        )
        assert build_labeled_issue(raw) is None
