"""Unit tests for classifier loader — refuse-to-boot behaviour.

These tests do NOT load a real model. They verify that ClassifierLoadError
is raised on all the failure conditions that trigger refuse-to-boot, including
the class-set drift check that asserts the deployed model's class space
matches the backend's compile-time contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ml.classifier import ClassifierLoadError, load_classifier

VALID_CLASSES = ["bug", "feature", "support"]


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    """Return an empty temp directory representing a model dir."""
    return tmp_path / "classifier"


class TestLoadClassifierRefusal:
    def test_missing_directory_raises(self, model_dir: Path) -> None:
        with pytest.raises(ClassifierLoadError, match="not found"):
            load_classifier(model_dir)

    def test_missing_model_card_raises(self, model_dir: Path) -> None:
        model_dir.mkdir()
        # No model_card.json — weights exist but card is absent.
        (model_dir / "pytorch_model.bin").write_bytes(b"fake")
        with pytest.raises(ClassifierLoadError, match="model_card.json missing"):
            load_classifier(model_dir)

    def test_malformed_model_card_raises(self, model_dir: Path) -> None:
        model_dir.mkdir()
        (model_dir / "model_card.json").write_text("{not valid json")
        with pytest.raises(ClassifierLoadError, match="malformed"):
            load_classifier(model_dir)

    def test_model_card_without_classes_raises(self, model_dir: Path) -> None:
        model_dir.mkdir()
        (model_dir / "model_card.json").write_text(json.dumps({"version": "1.0.0"}))
        (model_dir / "pytorch_model.bin").write_bytes(b"fake")
        with pytest.raises(ClassifierLoadError, match="no classes field"):
            load_classifier(model_dir)

    def test_class_set_drift_raises(self, model_dir: Path) -> None:
        """Trained for 4 classes but backend contract is 3 → refuse to boot."""
        model_dir.mkdir()
        card = {
            "version": "1.0.0",
            "classes": ["bug", "feature", "support", "documentation"],
            "model_sha256": "sha256:anything",
            "hyperparameters": {"max_length": 512},
        }
        (model_dir / "model_card.json").write_text(json.dumps(card))
        (model_dir / "pytorch_model.bin").write_bytes(b"fake")
        with pytest.raises(ClassifierLoadError, match="Class-set drift"):
            load_classifier(model_dir)

    def test_class_order_drift_raises(self, model_dir: Path) -> None:
        """Same classes but reordered → refuse to boot (index mapping changes)."""
        model_dir.mkdir()
        card = {
            "version": "1.0.0",
            "classes": ["feature", "bug", "support"],  # bug↔feature swapped
            "model_sha256": "sha256:anything",
            "hyperparameters": {"max_length": 512},
        }
        (model_dir / "model_card.json").write_text(json.dumps(card))
        (model_dir / "pytorch_model.bin").write_bytes(b"fake")
        with pytest.raises(ClassifierLoadError, match="Class-set drift"):
            load_classifier(model_dir)

    def test_model_card_without_sha256_raises(self, model_dir: Path) -> None:
        model_dir.mkdir()
        card = {"version": "1.0.0", "classes": VALID_CLASSES}
        (model_dir / "model_card.json").write_text(json.dumps(card))
        (model_dir / "pytorch_model.bin").write_bytes(b"fake")
        with pytest.raises(ClassifierLoadError, match="model_sha256"):
            load_classifier(model_dir)

    def test_sha256_mismatch_raises(self, model_dir: Path) -> None:
        model_dir.mkdir()
        weight_file = model_dir / "pytorch_model.bin"
        weight_file.write_bytes(b"real weights content")
        card = {
            "version": "1.0.0",
            "classes": VALID_CLASSES,
            "model_sha256": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "hyperparameters": {"max_length": 512},
        }
        (model_dir / "model_card.json").write_text(json.dumps(card))
        with pytest.raises(ClassifierLoadError, match="SHA-256 mismatch"):
            load_classifier(model_dir)

    def test_no_weight_files_raises(self, model_dir: Path) -> None:
        model_dir.mkdir()
        card = {
            "version": "1.0.0",
            "classes": VALID_CLASSES,
            "model_sha256": "sha256:anything",
            "hyperparameters": {"max_length": 512},
        }
        (model_dir / "model_card.json").write_text(json.dumps(card))
        # No .safetensors or .bin files at all.
        with pytest.raises(ClassifierLoadError, match="No weight files"):
            load_classifier(model_dir)
