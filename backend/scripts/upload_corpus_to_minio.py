"""Upload corpus files to MinIO.

Uploads everything under <repo-root>/corpus/ except:
  - corpus/excluded/          (meeting notes and noise filtered at ingest time)
  - corpus/monai_wiki/.git/   (git internals)
  - corpus/monai_wiki/images/ (binary assets not used by the RAG pipeline)

Destination layout inside the bucket:
  corpus/raw_issues.jsonl
  corpus/monai_wiki/<filename>.md
  ...

Run from the backend/ directory:
    uv run python scripts/upload_corpus_to_minio.py

Or with explicit env vars:
    MINIO_ENDPOINT=localhost:9000 uv run python scripts/upload_corpus_to_minio.py
"""

from __future__ import annotations

import mimetypes
import os
import sys
from pathlib import Path

# ── MinIO config (env vars or local dev defaults) ─────────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "copilot")
CORPUS_PREFIX = "corpus"

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = _REPO_ROOT / "corpus"

# Subdirectories to skip entirely (relative to CORPUS_DIR)
_SKIP_DIRS = {
    "excluded",
    "monai_wiki/.git",
    "monai_wiki/images",
}


def _is_excluded(path: Path) -> bool:
    """Return True if *path* is inside any of the skip directories."""
    rel = path.relative_to(CORPUS_DIR)
    rel_str = rel.as_posix()
    return any(rel_str == skip or rel_str.startswith(skip + "/") for skip in _SKIP_DIRS)


def _content_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def main() -> None:  # noqa: C901
    try:
        from minio import Minio
        from minio.error import S3Error
    except ImportError:
        print("ERROR: minio package not found.  Run: uv add minio")
        sys.exit(1)

    if not CORPUS_DIR.exists():
        print(f"ERROR: corpus directory not found: {CORPUS_DIR}")
        print("Run scripts/fetch_issues.py and clone the MONAI wiki first.")
        sys.exit(1)

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    # Ensure bucket exists
    try:
        if not client.bucket_exists(BUCKET):
            client.make_bucket(BUCKET)
            print(f"Created bucket: {BUCKET}")
        else:
            print(f"Bucket exists: {BUCKET}")
    except S3Error as exc:
        print(f"ERROR creating/checking bucket: {exc}")
        sys.exit(1)

    # Collect files
    all_files = sorted(f for f in CORPUS_DIR.rglob("*") if f.is_file())
    to_upload = [f for f in all_files if not _is_excluded(f)]
    skipped = len(all_files) - len(to_upload)

    print(f"\nCorpus root : {CORPUS_DIR}")
    print(f"Files found : {len(all_files)}  (skipping {skipped} excluded/git/image files)")
    print(f"To upload   : {len(to_upload)}")
    print()

    uploaded = 0
    errors = 0
    for f in to_upload:
        rel = f.relative_to(CORPUS_DIR)
        object_key = f"{CORPUS_PREFIX}/{rel.as_posix()}"
        ct = _content_type(f)
        size = f.stat().st_size
        try:
            client.fput_object(BUCKET, object_key, str(f), content_type=ct)
            print(f"  ✓  {object_key}  ({size:,} bytes)")
            uploaded += 1
        except S3Error as exc:
            print(f"  ✗  {object_key}: {exc}")
            errors += 1

    print(f"\n{'─' * 60}")
    print(f"Uploaded : {uploaded} files  →  minio://{BUCKET}/{CORPUS_PREFIX}/")
    if errors:
        print(f"Errors   : {errors} (see above)")
        sys.exit(1)
    else:
        print("All files uploaded successfully.")


if __name__ == "__main__":
    main()
