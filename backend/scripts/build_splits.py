"""Build stratified train/val/test splits from raw MONAI issues.

Split strategy (time-aware — test is strictly newer than train):
  1. Resolve labels and drop ambiguous/unlabeled issues.
  2. Sort ALL remaining issues by closed_at (oldest first).
  3. Test  = most recent 15% of the sorted list (no temporal leakage into train).
  4. Of the remaining 85%, stratified-shuffle split:
       Val   = 15/85 ≈ 17.6% of the remaining (≈15% of total)
       Train = the rest                         (≈70% of total)

Stratification preserves per-class proportions in both train and val.
Random seed is fixed (42) for reproducibility; document in DECISIONS.md.

Usage:
    cd backend
    uv run python scripts/build_splits.py
    uv run python scripts/build_splits.py --input data/raw_issues.jsonl --outdir data/
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from app.domain.issue import LabeledIssue, RawIssue, build_labeled_issue

RANDOM_SEED = 42
TEST_FRACTION = 0.15
VAL_OF_REMAINING_FRACTION = 0.15 / 0.85  # ≈ 0.176 → gives ~15% of total


def _stratified_split(
    items: list[LabeledIssue],
    fraction: float,
    rng: random.Random,
) -> tuple[list[LabeledIssue], list[LabeledIssue]]:
    """Split items into two groups, preserving class distribution.

    Args:
        items: Items to split.
        fraction: Fraction to put into the second group (val/test).
        rng: Seeded random instance for reproducibility.

    Returns:
        (majority, minority) where minority has ~fraction of each class.
    """
    by_class: dict[str, list[LabeledIssue]] = defaultdict(list)
    for item in items:
        by_class[item.label].append(item)

    majority: list[LabeledIssue] = []
    minority: list[LabeledIssue] = []

    for class_items in by_class.values():
        shuffled = class_items[:]
        rng.shuffle(shuffled)
        n_minority = max(1, round(len(shuffled) * fraction))
        majority.extend(shuffled[n_minority:])
        minority.extend(shuffled[:n_minority])

    return majority, minority


def build_splits(input_path: Path, outdir: Path) -> dict[str, int]:
    """Build and write train/val/test JSONL files.

    Args:
        input_path: Raw issues JSONL from fetch_issues.py.
        outdir: Directory to write train.jsonl, val.jsonl, test.jsonl.

    Returns:
        Dict with keys "train", "val", "test" mapping to issue counts.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)  # noqa: S311 — not crypto, reproducible ML shuffle

    # ------------------------------------------------------------------ #
    # 1. Load and resolve labels
    # ------------------------------------------------------------------ #
    labeled: list[LabeledIssue] = []
    dropped = 0

    with input_path.open() as fh:
        for line in fh:
            record = json.loads(line)
            raw = RawIssue.model_validate(record)
            issue = build_labeled_issue(raw)
            if issue is not None:
                labeled.append(issue)
            else:
                dropped += 1

    print(f"Loaded {len(labeled)} labeled issues ({dropped} dropped).")

    # ------------------------------------------------------------------ #
    # 2. Sort by closed_at (oldest first) for time-aware test split
    # ------------------------------------------------------------------ #
    labeled.sort(key=lambda x: x.closed_at)

    # ------------------------------------------------------------------ #
    # 3. Time-aware test split — most recent 15% strictly
    # ------------------------------------------------------------------ #
    n_test = max(1, round(len(labeled) * TEST_FRACTION))
    remaining = labeled[:-n_test]
    test = labeled[-n_test:]

    cutoff = test[0].closed_at
    print(f"Test cutoff: issues closed on/after {cutoff.date()} (n={len(test)})")
    assert all(  # noqa: S101 — invariant guard in a CLI script, not a test
        issue.closed_at >= cutoff for issue in test
    ), "Time-aware invariant violated: train issue is newer than a test issue"

    # ------------------------------------------------------------------ #
    # 4. Stratified train/val split from the remaining
    # ------------------------------------------------------------------ #
    train, val = _stratified_split(remaining, VAL_OF_REMAINING_FRACTION, rng)

    # ------------------------------------------------------------------ #
    # 5. Write JSONL splits
    # ------------------------------------------------------------------ #
    counts: dict[str, int] = {}
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = outdir / f"{split_name}.jsonl"
        with path.open("w") as fh:
            for issue in split_data:
                fh.write(issue.model_dump_json() + "\n")
        counts[split_name] = len(split_data)

    # ------------------------------------------------------------------ #
    # 6. Summary
    # ------------------------------------------------------------------ #
    from collections import Counter

    summary: dict[str, object] = {
        "random_seed": RANDOM_SEED,
        "total": sum(counts.values()),
        "splits": counts,
        "class_distribution": {},
    }

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        dist: Counter[str] = Counter(issue.label for issue in split_data)
        summary["class_distribution"] = {  # type: ignore[assignment]
            **summary.get("class_distribution", {}),  # type: ignore[arg-type]
            split_name: dict(dist),
        }

    summary_path = outdir / "splits_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSplit summary:")
    print(f"  train : {counts['train']:4d}")
    print(f"  val   : {counts['val']:4d}")
    print(f"  test  : {counts['test']:4d}")
    print(f"  total : {sum(counts.values()):4d}")
    print(f"\nSummary written to {summary_path}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw_issues.jsonl"),
        help="Raw issues JSONL (default: data/raw_issues.jsonl)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/"),
        help="Output directory for splits (default: data/)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found. Run fetch_issues.py first.")
        raise SystemExit(1)

    build_splits(args.input, args.outdir)


if __name__ == "__main__":
    main()
