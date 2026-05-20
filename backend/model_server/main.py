"""Model-server: FastAPI inference service for DistilBERT classifier.

Loaded models via lifespan singleton. Routes:
- GET /healthz — health check
- POST /predict — classify issue text
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from inference import ClassificationResult, predict

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent
BEST_MODEL_DIR = REPO_ROOT / "ml" / "artifacts" / "classifier" / "best"
MODEL_CARD_PATH = BEST_MODEL_DIR / "model_card.json"


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    max_length: int = Field(default=512, ge=128, le=1024)


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Load model on startup, dispose on shutdown."""
    # Load model card to verify SHA-256
    with open(MODEL_CARD_PATH) as f:
        card = json.load(f)

    logger.info(f"Loading classifier from {BEST_MODEL_DIR}")

    # Verify model SHA-256
    model_path = BEST_MODEL_DIR / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()

    if f"sha256:{model_sha}" != card["model_sha256"]:
        raise ValueError(
            f"Model SHA-256 mismatch. Expected {card['model_sha256']}, got sha256:{model_sha}"
        )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Store in app state
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = device
    app.state.model_card = card
    app.state.classes = tuple(card["classes"])

    logger.info(
        f"Classifier loaded. Version: {card['version']}, device: {device}, classes: {app.state.classes}"
    )

    yield

    # Cleanup
    del app.state.tokenizer
    del app.state.model
    logger.info("Model unloaded")


app = FastAPI(title="model-server", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
async def classify(req: PredictionRequest) -> PredictionResponse:
    """Classify issue text.

    Args:
        req: PredictionRequest with text and optional max_length

    Returns:
        PredictionResponse with label, confidence, model_version, latency_ms
    """
    import time

    start = time.perf_counter()

    try:
        tokenizer = app.state.tokenizer
        model = app.state.model
        device = app.state.device
        model_card = app.state.model_card
        classes = app.state.classes

        # Tokenize
        inputs = tokenizer(
            req.text,
            max_length=req.max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0].cpu().numpy()

        # Argmax + softmax
        label_idx = logits.argmax()
        confidence = float(1.0 / (1.0 + sum(1 for l in logits if l > logits[label_idx])))
        label = classes[int(label_idx)]

        latency_ms = (time.perf_counter() - start) * 1000

        return PredictionResponse(
            label=label,
            confidence=confidence,
            model_version=model_card["version"],
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
