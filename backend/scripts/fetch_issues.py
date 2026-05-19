"""Fetch closed issues from Project-MONAI/MONAI via the GitHub REST API.

Paginates through all closed issues, keeps only those carrying at least one
target label (bug, feature request, documentation, questions), and writes
them to a JSONL file.

Usage:
    cd backend
    GITHUB_TOKEN=<pat> uv run python scripts/fetch_issues.py
    GITHUB_TOKEN=<pat> uv run python scripts/fetch_issues.py --output data/raw_issues.jsonl
    GITHUB_TOKEN=<pat> uv run python scripts/fetch_issues.py --force  # re-download even if cached

The GITHUB_TOKEN env var is optional but strongly recommended:
  unauthenticated → 60 req/h, authenticated → 5000 req/h.

Note: scripts are standalone CLIs; os.getenv is permitted here (not app runtime code).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO = "Project-MONAI/MONAI"
API_BASE = "https://api.github.com"
PER_PAGE = 100

TARGET_LABELS: frozenset[str] = frozenset(
    ["bug", "feature request", "documentation", "questions"]
)


def _headers() -> dict[str, str]:
    token = os.getenv("GITHUB_TOKEN", "")
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _has_target_label(labels: list[dict[str, str]]) -> bool:
    return any(lbl["name"] in TARGET_LABELS for lbl in labels)


def fetch_issues(output_path: Path, force: bool = False) -> int:
    """Fetch all closed MONAI issues with target labels and write to JSONL.

    Args:
        output_path: Destination JSONL file.
        force: Re-download even if output_path already exists.

    Returns:
        Number of issues written.
    """
    if output_path.exists() and not force:
        print(f"Cache hit: {output_path} already exists. Use --force to re-download.")
        count = sum(1 for _ in output_path.open())
        print(f"Cached file contains {count} issues.")
        return count

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    page = 1

    with httpx.Client(timeout=30.0) as client, output_path.open("w") as fh:
        while True:
            url = (
                f"{API_BASE}/repos/{REPO}/issues"
                f"?state=closed&per_page={PER_PAGE}&page={page}&sort=created&direction=asc"
            )
            resp = client.get(url, headers=_headers())

            if resp.status_code == 403:
                print("Rate limited. Waiting 60s…", file=sys.stderr)
                time.sleep(60)
                continue

            resp.raise_for_status()
            items: list[dict] = resp.json()  # type: ignore[type-arg]

            if not items:
                break

            for item in items:
                # GitHub /issues endpoint returns PRs too — skip them.
                if "pull_request" in item:
                    continue
                if not _has_target_label(item.get("labels", [])):
                    continue

                record = {
                    "id": item["id"],
                    "number": item["number"],
                    "title": item["title"],
                    "body": item.get("body") or "",
                    "labels": [lbl["name"] for lbl in item.get("labels", [])],
                    "created_at": item["created_at"],
                    "closed_at": item.get("closed_at"),
                }
                fh.write(json.dumps(record) + "\n")
                written += 1

            print(f"  page {page}: fetched {len(items)} items, kept {written} total so far")
            page += 1

            # Respect secondary rate limit — stay well under 5000 req/h.
            time.sleep(0.5)

    print(f"\nDone. Wrote {written} issues to {output_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw_issues.jsonl"),
        help="Output JSONL file (default: data/raw_issues.jsonl)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the output file already exists",
    )
    args = parser.parse_args()

    if not os.getenv("GITHUB_TOKEN"):
        print(
            "Warning: GITHUB_TOKEN not set. Using unauthenticated API (60 req/h limit).",
            file=sys.stderr,
        )

    fetch_issues(args.output, force=args.force)


if __name__ == "__main__":
    main()
