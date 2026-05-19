"""Domain models and label mapping for GitHub issues.

Label mapping rationale (documented in docs/DECISIONS.md):
- 'documentation' has only 28 closed issues in MONAI → ~5 examples in test after
  a 70/15/15 stratified split → per-class F1 is statistical noise (1 error = 20-pt swing).
- Routing for 'documentation' and 'questions' is identical for a maintainer
  (point user to the right doc/FAQ). Merging to 'support' is semantically valid.
- Final 3-class problem: bug=337, feature=535, support=278. Balanced and defensible.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Label mapping: GitHub maintainer label → canonical 3-class label
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, str] = {
    "bug": "bug",
    "feature request": "feature",
    "documentation": "support",
    "questions": "support",
}

# Canonical output classes (order is the class index for the classifier head).
CLASS_NAMES: tuple[str, ...] = ("bug", "feature", "support")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}

_TARGET_LABELS: frozenset[str] = frozenset(LABEL_MAP.keys())


def resolve_label(labels: list[str]) -> str | None:
    """Map a list of GitHub label names to a single canonical class.

    Args:
        labels: Raw label names from the GitHub API response.

    Returns:
        One of "bug", "feature", "support", or None if the issue should be
        dropped (no matching label, or conflicting labels like bug + feature).
    """
    mapped: set[str] = {LABEL_MAP[lbl] for lbl in labels if lbl in _TARGET_LABELS}
    if len(mapped) == 1:
        return mapped.pop()
    # 0 → no target label present; >1 → ambiguous multi-class (drop both).
    return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class RawIssue(BaseModel):
    """GitHub issue as returned by the fetch script (minimal fields)."""

    id: int
    number: int
    title: str
    body: str | None = None
    labels: list[str] = Field(default_factory=list)
    created_at: datetime
    closed_at: datetime | None = None


class LabeledIssue(BaseModel):
    """Issue after label resolution and text normalisation — ready for splitting."""

    id: int
    number: int
    # Combined classifier input: title + body, normalised.
    text: str
    label: str  # "bug" | "feature" | "support"
    label_idx: int  # 0 | 1 | 2 — index into CLASS_NAMES
    closed_at: datetime


def build_labeled_issue(raw: RawIssue) -> LabeledIssue | None:
    """Convert a RawIssue into a LabeledIssue, or return None if it should be dropped.

    Args:
        raw: Issue as fetched from GitHub.

    Returns:
        LabeledIssue or None if no unambiguous label or missing closed_at.
    """
    label = resolve_label(raw.labels)
    if label is None or raw.closed_at is None:
        return None

    body = raw.body or ""
    text = f"{raw.title}\n\n{body}".strip()

    return LabeledIssue(
        id=raw.id,
        number=raw.number,
        text=text,
        label=label,
        label_idx=CLASS_TO_IDX[label],
        closed_at=raw.closed_at,
    )
