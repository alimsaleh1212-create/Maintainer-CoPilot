"""CLI entry point for training the DistilBERT 3-class classifier.

Usage:
    cd backend
    uv run python scripts/train_classifier.py
    uv run python scripts/train_classifier.py --train data/train.jsonl --val data/val.jsonl
    uv run python scripts/train_classifier.py --epochs 3 --batch-size 8  # small machine
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/classifier"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    for p in (args.train, args.val):
        if not p.exists():
            print(f"Error: {p} not found. Run build_splits.py first.")
            raise SystemExit(1)

    from app.ml.train_classifier import TrainConfig, train

    config = TrainConfig(
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        output_dir=args.output,
    )

    best_dir = train(args.train, args.val, config)
    print(f"\nTraining complete. Best model: {best_dir}")
    print("Next: run eval/run_classification_eval.py to compute three-way comparison.")


if __name__ == "__main__":
    main()
