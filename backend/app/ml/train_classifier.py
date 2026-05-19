"""DistilBERT 3-class issue classifier — training pipeline.

Entry point: scripts/train_classifier.py (CLI wrapper).
Outputs: artifacts/classifier/best/ with weights + tokenizer + model_card.json.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from app.domain.issue import CLASS_NAMES, CLASS_TO_IDX, LabeledIssue

logger = structlog.get_logger(__name__)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512


@dataclass
class TrainConfig:
    """Hyperparameters for training — document in DECISIONS.md after first run."""

    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    logging_steps: int = 50
    output_dir: Path = field(default_factory=lambda: Path("artifacts/classifier"))


def load_split(path: Path) -> list[LabeledIssue]:
    """Load a JSONL split produced by scripts/build_splits.py."""
    issues: list[LabeledIssue] = []
    with path.open() as fh:
        for line in fh:
            issues.append(LabeledIssue.model_validate_json(line.strip()))
    return issues


def _sha256_dir_weights(directory: Path) -> str:
    """Hash all weight files in a model directory."""
    h = hashlib.sha256()
    matched: list[Path] = []
    for pat in ("*.safetensors", "*.bin"):
        matched.extend(sorted(directory.glob(pat)))
    for p in matched:
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _make_compute_metrics(metric: Any) -> Any:
    def compute(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        result: dict[str, float] = metric.compute(
            predictions=predictions,
            references=labels,
            average="macro",
        )
        return result

    return compute


def _issues_to_hf_dict(issues: list[LabeledIssue]) -> dict[str, list[Any]]:
    return {
        "text": [i.text for i in issues],
        "label": [i.label_idx for i in issues],
    }


def train(
    train_path: Path,
    val_path: Path,
    config: TrainConfig | None = None,
) -> Path:
    """Fine-tune DistilBERT for 3-class issue classification.

    Args:
        train_path: Path to train.jsonl produced by build_splits.py.
        val_path:   Path to val.jsonl.
        config:     Hyperparameter config (uses defaults if None).

    Returns:
        Path to the best model directory (contains weights + model_card.json).
    """
    import evaluate  # type: ignore[import-untyped]
    from datasets import Dataset  # type: ignore[import-untyped]
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    if config is None:
        config = TrainConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = config.output_dir / "best"

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    train_issues = load_split(train_path)
    val_issues = load_split(val_path)
    logger.info("data_loaded", train_size=len(train_issues), val_size=len(val_issues))

    # ------------------------------------------------------------------ #
    # 2. Tokeniser
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(  # type: ignore[no-any-return]
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    train_ds = Dataset.from_dict(_issues_to_hf_dict(train_issues)).map(
        tokenize, batched=True
    )
    val_ds = Dataset.from_dict(_issues_to_hf_dict(val_issues)).map(
        tokenize, batched=True
    )

    # ------------------------------------------------------------------ #
    # 3. Model
    # ------------------------------------------------------------------ #
    id2label = {i: c for i, c in enumerate(CLASS_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=dict(CLASS_TO_IDX),
    )

    # ------------------------------------------------------------------ #
    # 4. Trainer
    # ------------------------------------------------------------------ #
    f1_metric = evaluate.load("f1")

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=config.logging_steps,
        save_total_limit=2,
        report_to="none",  # local JSONL log only; swap to "wandb" if W&B is available
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_make_compute_metrics(f1_metric),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
    )

    trainer.train()

    # ------------------------------------------------------------------ #
    # 5. Save best checkpoint
    # ------------------------------------------------------------------ #
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # ------------------------------------------------------------------ #
    # 6. Evaluate and write model card
    # ------------------------------------------------------------------ #
    eval_results: dict[str, float] = trainer.evaluate()
    val_f1: float = eval_results.get("eval_f1", 0.0)

    model_sha256 = _sha256_dir_weights(best_dir)
    train_sha256 = _sha256_file(train_path)

    model_card: dict[str, Any] = {
        "architecture": MODEL_NAME,
        "num_labels": len(CLASS_NAMES),
        "classes": list(CLASS_NAMES),
        "class_to_idx": dict(CLASS_TO_IDX),
        "hyperparameters": {
            "learning_rate": config.learning_rate,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "num_train_epochs": config.num_train_epochs,
            "weight_decay": config.weight_decay,
            "warmup_ratio": config.warmup_ratio,
            "max_length": MAX_LENGTH,
        },
        "freeze_policy": "all layers unfrozen — full fine-tune of DistilBERT",
        "training_data_sha256": train_sha256,
        "training_data_size": {"train": len(train_issues), "val": len(val_issues)},
        "metrics": {
            "val_f1_macro": val_f1,
            "raw_eval": eval_results,
        },
        "model_sha256": model_sha256,
        "trained_at": datetime.now(UTC).isoformat(),
        "version": "1.0.0",
    }

    (best_dir / "model_card.json").write_text(json.dumps(model_card, indent=2))

    logger.info(
        "training_complete",
        best_dir=str(best_dir),
        val_f1_macro=val_f1,
        model_sha256=model_sha256[:20] + "…",
    )
    return best_dir
