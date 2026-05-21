"""Model-server: FastAPI inference service for the DistilBERT classifier.

Loaded as a singleton via lifespan. Refuses to start if weights are missing
or SHA-256 mismatches model_card.json.

Routes:
    GET  /healthz  — liveness probe
    POST /predict  — classify issue text → {label, confidence, model_version, latency_ms}
    POST /ner      — extract code entities → [{text, label, start, end}, ...]
    POST /summarize — summarize text → {summary}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_server.inference import ClassificationResult, predict_sync, sha256_model_dir

logger = structlog.get_logger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Model path ─────────────────────────────────────────────────────────────────
# Priority:
#   1. CLASSIFIER_MODEL_DIR env var (explicit local path — for dev bind-mount)
#   2. Download from MinIO if MINIO_ENDPOINT + MINIO_MODEL_PREFIX are set
#   3. Fallback: sibling ml/ directory (local dev without Docker)
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts" / "classifier" / "best"
MODEL_DIR = Path(os.getenv("CLASSIFIER_MODEL_DIR", str(_DEFAULT_MODEL_DIR)))

# MinIO pull config (download weights on container start)
_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
_MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
_MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
_MINIO_BUCKET = os.getenv("MINIO_BUCKET", "copilot")
_MINIO_MODEL_PREFIX = os.getenv("MINIO_MODEL_PREFIX", "models/classifier/best")


def _pull_model_from_minio(dest: Path) -> bool:
    """Download model artifacts from MinIO into dest/.

    Returns True on success, False if MinIO unavailable or not configured.
    """
    if not _MINIO_ENDPOINT:
        return False
    try:
        from minio import Minio  # type: ignore[import-untyped]

        client = Minio(
            _MINIO_ENDPOINT,
            access_key=_MINIO_ACCESS_KEY,
            secret_key=_MINIO_SECRET_KEY,
            secure=False,
        )
        if not client.bucket_exists(_MINIO_BUCKET):
            logger.warning("minio.bucket_missing", bucket=_MINIO_BUCKET)
            return False

        objects = list(client.list_objects(_MINIO_BUCKET, prefix=_MINIO_MODEL_PREFIX, recursive=True))
        if not objects:
            logger.warning("minio.no_objects", prefix=_MINIO_MODEL_PREFIX)
            return False

        dest.mkdir(parents=True, exist_ok=True)
        for obj in objects:
            rel = obj.object_name[len(_MINIO_MODEL_PREFIX):].lstrip("/")
            if not rel:
                continue
            local = dest / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            client.fget_object(_MINIO_BUCKET, obj.object_name, str(local))
            logger.info("minio.downloaded", object=obj.object_name)

        logger.info("minio.pull_complete", dest=str(dest), count=len(objects))
        return True
    except Exception as exc:
        logger.warning("minio.pull_failed", error=str(exc))
        return False


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192)
    max_length: int = Field(default=512, ge=64, le=1024)


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192)


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=16384)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Load models on startup; dispose on shutdown."""
    # ── Classifier — try MinIO first, then local path ──────────────────────
    if not MODEL_DIR.exists() and _MINIO_ENDPOINT:
        logger.info("minio.pulling_weights", dest=str(MODEL_DIR))
        _pull_model_from_minio(MODEL_DIR)

    if not MODEL_DIR.exists():
        logger.critical("refuse_to_boot", reason="classifier_dir_missing", path=str(MODEL_DIR))
        sys.exit(1)

    card_path = MODEL_DIR / "model_card.json"
    if not card_path.exists():
        logger.critical("refuse_to_boot", reason="model_card_missing", path=str(card_path))
        sys.exit(1)

    try:
        card: dict[str, Any] = json.loads(card_path.read_text())
    except json.JSONDecodeError as exc:
        logger.critical("refuse_to_boot", reason="model_card_malformed", detail=str(exc))
        sys.exit(1)

    expected_hash = card.get("model_sha256", "")
    actual_hash = sha256_model_dir(MODEL_DIR)
    if expected_hash and actual_hash != expected_hash:
        logger.critical(
            "refuse_to_boot",
            reason="sha256_mismatch",
            expected=expected_hash[:20],
            actual=actual_hash[:20],
        )
        sys.exit(1)

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    except Exception as exc:
        logger.critical("refuse_to_boot", reason="model_load_failed", detail=str(exc))
        sys.exit(1)

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = device
    app.state.classes = tuple(card["classes"])
    app.state.model_version = card.get("version", "unknown")
    app.state.max_length = card.get("hyperparameters", {}).get("max_length", 512)

    logger.info(
        "classifier_loaded",
        version=app.state.model_version,
        device=device,
        classes=app.state.classes,
        sha256=actual_hash[:16],
    )

    yield

    del app.state.tokenizer
    del app.state.model
    logger.info("model_server_shutdown")


app = FastAPI(title="model-server", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=ClassificationResult)
async def classify(req: PredictionRequest) -> ClassificationResult:
    """Classify issue text into bug / feature / support."""
    try:
        result = await asyncio.to_thread(
            predict_sync,
            req.text,
            app.state.tokenizer,
            app.state.model,
            app.state.device,
            app.state.classes,
            app.state.model_version,
            req.max_length,
        )
        return result
    except Exception as exc:
        logger.error("predict_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/ner")
async def ner(req: NERRequest) -> dict[str, list[dict[str, str]]]:
    """Extract code-shaped named entities from issue text."""
    import re

    # Regex-based NER (no spaCy required in model-server)
    entities: list[dict[str, str]] = []
    patterns = [
        (r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", "ClassName"),
        (r"\b([a-z_]+(?:_[a-z]+)+)\s*\(", "FunctionName"),
        (r"(?:[\w./]+/)+[\w.]+\.py\b", "FilePath"),
        (r"\b[A-Z][a-zA-Z]*(?:Error|Exception|Warning)\b", "ErrorType"),
        (r"v?(\d+\.\d+(?:\.\d+)?(?:[a-zA-Z0-9]+)?)", "VersionNumber"),
    ]
    for pattern, label in patterns:
        for m in re.finditer(pattern, req.text):
            entities.append({
                "text": m.group(0),
                "label": label,
                "start": str(m.start()),
                "end": str(m.end()),
            })

    return {"entities": entities}


@app.post("/summarize")
async def summarize(req: SummarizeRequest) -> dict[str, str]:
    """Return a truncated summary placeholder (actual summarization uses LLM in API)."""
    # Model-server provides a simple extractive summary; full LLM summarization is in API.
    text = req.text[:500].strip()
    summary = text.split("\n")[0][:200]
    return {"summary": summary}
