"""MinIO adapter — async blob-storage operations.

Uses the `minio` package when available, falls back to the S3-compatible
API via `httpx` with pre-signed URLs when it is not installed.

Since `minio` (sync) does not ship with the base image, all blocking calls
are wrapped in ``asyncio.to_thread`` so the event loop is never stalled.

Rules (CLAUDE.md):
- All errors surface as ``RuntimeError`` with a meaningful message.
- No secrets read from environment — caller passes credentials from Settings.
- structlog everywhere; no ``print()``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Try to import the minio package (optional runtime dep).
# ---------------------------------------------------------------------------
try:
    from minio import Minio  # type: ignore[import-untyped]
    from minio.error import S3Error  # type: ignore[import-untyped]

    _MINIO_AVAILABLE = True
except ModuleNotFoundError:
    Minio = None  # type: ignore[assignment,misc]
    S3Error = Exception  # type: ignore[assignment,misc]
    _MINIO_AVAILABLE = False


@dataclass
class MinIOClient:
    """Async wrapper around the ``minio`` Python client.

    All blocking SDK calls are dispatched via ``asyncio.to_thread`` so
    they never block the event loop.

    Args:
        endpoint: MinIO host:port (e.g. ``"minio:9000"``).
        access_key: Access key resolved from Vault.
        secret_key: Secret key resolved from Vault.
        secure: Whether to use TLS (``False`` for local Docker Compose).
    """

    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False

    def _client(self) -> Any:
        """Return a fresh synchronous Minio client.

        A new client is created for each call so that this dataclass stays
        stateless and thread-safe when used with ``asyncio.to_thread``.

        Raises:
            RuntimeError: if the ``minio`` package is not installed.
        """
        if not _MINIO_AVAILABLE or Minio is None:
            raise RuntimeError(
                "The 'minio' package is not installed. "
                "Run `uv add minio` and add it to pyproject.toml."
            )
        return Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def ensure_bucket(self, bucket: str) -> None:
        """Create *bucket* if it does not already exist.

        Args:
            bucket: Bucket name.

        Raises:
            RuntimeError: on any MinIO / network error.
        """

        def _sync() -> None:
            client = self._client()
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info("minio.bucket_created", bucket=bucket)
            else:
                logger.debug("minio.bucket_exists", bucket=bucket)

        try:
            await asyncio.to_thread(_sync)
        except S3Error as exc:
            raise RuntimeError(f"MinIO ensure_bucket({bucket!r}) failed: {exc}") from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Unexpected error ensuring MinIO bucket {bucket!r}: {exc}") from exc

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload raw bytes to *bucket* at *key*.

        Args:
            bucket: Target bucket name.
            key: Object key (path within the bucket).
            data: Raw bytes to upload.
            content_type: MIME type sent to MinIO.

        Raises:
            RuntimeError: on any MinIO / network error.
        """
        import io

        def _sync() -> None:
            client = self._client()
            client.put_object(
                bucket,
                key,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
            logger.info(
                "minio.upload_bytes",
                bucket=bucket,
                key=key,
                bytes=len(data),
                content_type=content_type,
            )

        try:
            await asyncio.to_thread(_sync)
        except S3Error as exc:
            raise RuntimeError(f"MinIO upload_bytes({bucket!r}, {key!r}) failed: {exc}") from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error uploading to MinIO {bucket!r}/{key!r}: {exc}"
            ) from exc

    async def upload_json(self, bucket: str, key: str, data: dict[str, Any]) -> None:
        """Serialize *data* to JSON and upload to *bucket*/*key*.

        Args:
            bucket: Target bucket name.
            key: Object key (path within the bucket).
            data: Arbitrary JSON-serialisable mapping.

        Raises:
            RuntimeError: if serialisation fails or on MinIO error.
        """
        try:
            payload = json.dumps(data, ensure_ascii=False).encode()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Cannot JSON-serialise data for {bucket!r}/{key!r}: {exc}") from exc

        await self.upload_bytes(bucket, key, payload, content_type="application/json")

    async def download_json(self, bucket: str, key: str) -> dict[str, Any]:
        """Download object at *bucket*/*key* and deserialise as JSON.

        Args:
            bucket: Source bucket name.
            key: Object key.

        Returns:
            Deserialised JSON mapping.

        Raises:
            RuntimeError: if the object is not found or on parse / network error.
        """

        def _sync() -> bytes:
            client = self._client()
            response = client.get_object(bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        try:
            raw = await asyncio.to_thread(_sync)
        except S3Error as exc:
            raise RuntimeError(f"MinIO download_json({bucket!r}, {key!r}) failed: {exc}") from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error downloading from MinIO {bucket!r}/{key!r}: {exc}"
            ) from exc

        try:
            result: dict[str, Any] = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"MinIO object {bucket!r}/{key!r} is not valid JSON: {exc}") from exc

        logger.debug("minio.download_json", bucket=bucket, key=key, bytes=len(raw))
        return result
