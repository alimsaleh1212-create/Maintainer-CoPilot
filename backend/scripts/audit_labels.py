"""Audit label distribution in the raw issues file.

Prints per-label counts and confirms the class mapping choices.

Usage:
    cd backend
    uv run python scripts/audit_labels.py
    uv run python scripts/audit_labels.py --input data/raw_issues.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.domain.issue import CLASS_NAMES, LABEL_MAP, resolve_label


def audit(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run fetch_issues.py first."
        )

    raw_label_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    dropped = 0
    total = 0

    with input_path.open() as fh:
        for line in fh:
            record = json.loads(line)
            labels: list[str] = record.get("labels", [])
            total += 1
            for lbl in labels:
                if lbl in LABEL_MAP:
                    raw_label_counts[lbl] += 1
            resolved = resolve_label(labels)
            if resolved:
                mapped_counts[resolved] += 1
            else:
                dropped += 1

    print(f"Total issues: {total}")
    print(f"Dropped (no/ambiguous label): {dropped}")
    print()
    print("Raw GitHub label counts (target labels only):")
    for lbl, count in sorted(raw_label_counts.items(), key=lambda x: -x[1]):
        canonical = LABEL_MAP[lbl]
        print(f"  {lbl:<20} → {canonical:<10} : {count}")
    print()
    print("Canonical 3-class distribution:")
    kept = sum(mapped_counts.values())
    for cls in CLASS_NAMES:
        count = mapped_counts[cls]
        pct = 100 * count / kept if kept else 0
        print(f"  {cls:<10} : {count:4d}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<10} : {kept:4d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw_issues.jsonl"),
        help="Raw issues JSONL (default: data/raw_issues.jsonl)",
    )
    args = parser.parse_args()
    audit(args.input)


if __name__ == "__main__":
    main()
