"""DistilBERT classifier loader — used by the model-server lifespan.

Loads the fine-tuned model from disk, verifies SHA-256 against the model card,
and exposes a synchronous predict() method (wrapped in asyncio.to_thread by callers).

Refuse-to-boot conditions (wired in model_server/main.py lifespan):
  - model_dir does not exist
  - model_card.json is missing or malformed
  - SHA-256 of the model weights does not match model_card.json
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import structlog

from app.domain.classification import ClassificationResult
from app.domain.issue import CLASS_NAMES, ClassLabel

logger = structlog.get_logger(__name__)


class ClassifierLoadError(RuntimeError):
    """Raised when the classifier cannot be loaded (missing weights, bad hash)."""


def _sha256_dir_weights(model_dir: Path) -> str:
    """Hash all .safetensors or .bin weight files, sorted by name."""
    h = hashlib.sha256()
    patterns = ["*.safetensors", "*.bin"]
    matched: list[Path] = []
    for pattern in patterns:
        matched.extend(sorted(model_dir.glob(pattern)))
    if not matched:
        raise ClassifierLoadError(f"No weight files found in {model_dir}")
    for p in matched:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


class DistilBertClassifier:
    """Loaded DistilBERT classifier.  Thread-safe for read-only inference."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_version: str,
        classes: tuple[ClassLabel, ...] = CLASS_NAMES,
        max_length: int = 512,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._version = model_version
        self._classes = classes
        self._max_length = max_length

    @property
    def version(self) -> str:
        return self._version

    def predict(self, text: str) -> ClassificationResult:
        """Run inference on a single text string.

        Args:
            text: Raw issue text (title + body).

        Returns:
            ClassificationResult with label, confidence, version, latency_ms.
        """
        import torch

        t0 = time.perf_counter()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax().item())
        confidence = float(probs[idx].item())
        label = self._classes[idx]

        latency_ms = (time.perf_counter() - t0) * 1000

        return ClassificationResult(
            label=label,
            confidence=confidence,
            model_version=self._version,
            latency_ms=latency_ms,
        )


def load_classifier(model_dir: Path) -> DistilBertClassifier:
    """Load and verify the fine-tuned classifier.

    Args:
        model_dir: Directory containing model weights, tokenizer, model_card.json.

    Returns:
        Loaded DistilBertClassifier, ready for inference.

    Raises:
        ClassifierLoadError: if weights are missing, model card is absent,
            or the SHA-256 hash does not match.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    if not model_dir.exists():
        raise ClassifierLoadError(f"Classifier model directory not found: {model_dir}")

    card_path = model_dir / "model_card.json"
    if not card_path.exists():
        raise ClassifierLoadError(f"model_card.json missing from {model_dir}")

    try:
        card: dict[str, Any] = json.loads(card_path.read_text())
    except json.JSONDecodeError as exc:
        raise ClassifierLoadError(f"model_card.json is malformed: {exc}") from exc

    card_classes = card.get("classes")
    if card_classes is None:
        raise ClassifierLoadError("model_card.json has no classes field")
    if tuple(card_classes) != CLASS_NAMES:
        raise ClassifierLoadError(
            f"Class-set drift: model_card.json classes {card_classes!r} "
            f"do not match backend contract {list(CLASS_NAMES)!r}. "
            f"This artifact was trained for a different class set and will not load."
        )

    expected_hash: str = card.get("model_sha256", "")
    if not expected_hash:
        raise ClassifierLoadError("model_card.json has no model_sha256 field")

    actual_hash = _sha256_dir_weights(model_dir)
    if actual_hash != expected_hash:
        raise ClassifierLoadError(
            f"SHA-256 mismatch for {model_dir}: expected {expected_hash}, got {actual_hash}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    version: str = card.get("version", "unknown")
    max_length: int = card.get("hyperparameters", {}).get("max_length", 512)

    logger.info(
        "classifier_loaded",
        version=version,
        model_sha256=actual_hash[:16] + "…",
        val_f1=card.get("metrics", {}).get("val_f1_macro", "?"),
    )

    return DistilBertClassifier(
        model=model,
        tokenizer=tokenizer,
        model_version=version,
        classes=CLASS_NAMES,
        max_length=max_length,
    )
