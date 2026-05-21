"""Upload classifier artifacts to MinIO.

Run once after training:
    uv run python scripts/upload_model_to_minio.py

Or inside the api container after copying the script:
    python scripts/upload_model_to_minio.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── MinIO config (use env vars or defaults for dev) ───────────────────────────
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "copilot")
PREFIX = "models/classifier/best"

# ── Artifact path ─────────────────────────────────────────────────────────────
_DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts" / "classifier" / "best"
)
ARTIFACT_DIR = Path(os.getenv("CLASSIFIER_ARTIFACT_DIR", str(_DEFAULT_ARTIFACT_DIR)))


def main() -> None:
    try:
        from minio import Minio
        from minio.error import S3Error
    except ImportError:
        print("ERROR: minio package not found. Install with: uv add minio")
        sys.exit(1)

    if not ARTIFACT_DIR.exists():
        print(f"ERROR: Artifact directory not found: {ARTIFACT_DIR}")
        sys.exit(1)

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    # Create bucket if needed
    try:
        if not client.bucket_exists(BUCKET):
            client.make_bucket(BUCKET)
            print(f"Created bucket: {BUCKET}")
        else:
            print(f"Bucket exists: {BUCKET}")
    except S3Error as e:
        print(f"ERROR creating bucket: {e}")
        sys.exit(1)

    # Upload all files
    files = sorted(ARTIFACT_DIR.rglob("*"))
    uploaded = 0
    for f in files:
        if f.is_file():
            object_name = f"{PREFIX}/{f.relative_to(ARTIFACT_DIR)}"
            try:
                client.fput_object(BUCKET, object_name, str(f))
                print(f"  Uploaded: {object_name}  ({f.stat().st_size:,} bytes)")
                uploaded += 1
            except S3Error as e:
                print(f"  ERROR uploading {object_name}: {e}")

    print(f"\nDone: {uploaded} files uploaded to minio://{BUCKET}/{PREFIX}")
    print("\nSet in model-server environment:")
    print(f"  MINIO_ENDPOINT={MINIO_ENDPOINT}")
    print(f"  MINIO_BUCKET={BUCKET}")
    print(f"  MINIO_MODEL_PREFIX={PREFIX}")


if __name__ == "__main__":
    main()
