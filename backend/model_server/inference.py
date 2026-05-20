"""Inference helpers for DistilBERT classifier."""

from __future__ import annotations

from pydantic import BaseModel


class ClassificationResult(BaseModel):
    label: str
    confidence: float
    model_version: str
    latency_ms: float


async def predict(text: str, tokenizer: Any, model: Any, device: str, classes: tuple[str, ...]) -> dict[str, Any]:
    """Predict label for issue text."""
    import torch
    import time

    start = time.perf_counter()

    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0].cpu().numpy()

    label_idx = logits.argmax()
    label = classes[int(label_idx)]
    confidence = float(logits[label_idx])

    return {
        "label": label,
        "confidence": confidence,
        "latency_ms": (time.perf_counter() - start) * 1000,
    }
