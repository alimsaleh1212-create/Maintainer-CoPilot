"""Canonical 3-class label contract for the issue classifier.

This module is the inference-time source of truth for the classifier's
output space. The training-side label mapping (4 GitHub labels → 3 classes,
including the documentation+questions → support merge) lives in the Colab
training notebook (backend/notebooks/train_classifier_colab.ipynb) and is
documented in docs/DECISIONS.md.

`load_classifier()` asserts that `model_card["classes"] == list(CLASS_NAMES)`
at boot — a deployed model with a different class set refuses to load.
"""

from __future__ import annotations

from typing import Literal

ClassLabel = Literal["bug", "feature", "support"]

CLASS_NAMES: tuple[ClassLabel, ...] = ("bug", "feature", "support")
CLASS_TO_IDX: dict[ClassLabel, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
