"""Inference helpers for DistilBERT classifier (model-server)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ClassificationResult(BaseModel):
    label: str
    confidence: float
    model_version: str
    latency_ms: float


def sha256_model_dir(model_dir: Path) -> str:
    """Hash all .safetensors and .bin weight files in model_dir, sorted by name."""
    h = hashlib.sha256()
    matched: list[Path] = []
    for pattern in ("*.safetensors", "*.bin"):
        matched.extend(sorted(model_dir.glob(pattern)))
    for p in matched:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def predict_sync(
    text: str,
    tokenizer: Any,
    model: Any,
    device: str,
    classes: tuple[str, ...],
    model_version: str,
    max_length: int = 512,
) -> ClassificationResult:
    """Synchronous single-text classification (call via asyncio.to_thread)."""
    import torch

    t0 = time.perf_counter()
    inputs = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax().item())
    confidence = float(probs[idx].item())
    label = classes[idx]
    latency_ms = (time.perf_counter() - t0) * 1000

    return ClassificationResult(
        label=label,
        confidence=confidence,
        model_version=model_version,
        latency_ms=latency_ms,
    )
