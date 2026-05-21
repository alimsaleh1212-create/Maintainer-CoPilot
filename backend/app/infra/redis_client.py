"""Redis client wrapper — async key-value store operations.

Thin async wrapper around ``redis.asyncio.Redis``.  The ``RedisClient``
instance is constructed in ``lifespan()`` (``app.state.redis``) and injected
via ``Depends()``.  Callers never interact with the raw ``Redis`` object.

Rules (CLAUDE.md):
- No secrets / connection strings read here; caller passes the Redis instance.
- structlog everywhere; no ``print()``.
- Every method is async and never blocks the event loop.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from redis.asyncio import Redis

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class RedisClient:
    """Async key-value store client built on ``redis.asyncio.Redis``.

    The underlying ``Redis`` connection pool is created outside this class
    (in ``lifespan()``) and passed in.  This keeps the client stateless and
    easy to replace with a fake in tests via ``app.dependency_overrides``.

    Args:
        redis: An already-connected ``redis.asyncio.Redis`` instance.
    """

    def __init__(self, redis: Redis) -> None:  # type: ignore[type-arg]
        self._redis: Redis = redis  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Low-level primitives
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Return ``True`` if the Redis server is reachable.

        Returns:
            ``True`` on a successful PING, ``False`` on any error.
        """
        try:
            result: Any = await self._redis.ping()
            return bool(result)
        except Exception as exc:
            logger.warning("redis.ping_failed", error=str(exc))
            return False

    async def get(self, key: str) -> str | None:
        """Retrieve the string value stored at *key*.

        Args:
            key: Redis key.

        Returns:
            Decoded UTF-8 string if the key exists, else ``None``.
        """
        raw: bytes | None = await self._redis.get(key)
        if raw is None:
            logger.debug("redis.get_miss", key=key)
            return None
        value = raw.decode("utf-8")
        logger.debug("redis.get_hit", key=key)
        return value

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        """Store a string *value* at *key* with an optional TTL.

        Args:
            key: Redis key.
            value: UTF-8 string value.
            ex: Optional expiry in seconds.  ``None`` means no expiry.
        """
        await self._redis.set(key, value, ex=ex)
        logger.debug("redis.set", key=key, ex=ex)

    async def delete(self, key: str) -> int:
        """Delete *key* and return the number of keys removed (0 or 1).

        Args:
            key: Redis key to remove.

        Returns:
            1 if the key existed and was deleted, 0 otherwise.
        """
        count: int = await self._redis.delete(key)
        logger.debug("redis.delete", key=key, removed=count)
        return count

    async def keys(self, pattern: str = "*") -> list[str]:
        """Return all keys matching *pattern*.

        Args:
            pattern: Redis glob-style pattern (e.g. ``"session:*"``).

        Returns:
            List of matching key strings.

        Note:
            Avoid ``KEYS *`` on large Redis instances in production; prefer
            ``SCAN``-based iteration for large keyspaces.
        """
        raw_keys: list[bytes] = await self._redis.keys(pattern)
        result = [k.decode("utf-8") for k in raw_keys]
        logger.debug("redis.keys", pattern=pattern, count=len(result))
        return result

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    async def get_json(self, key: str) -> Any | None:
        """Retrieve and deserialise a JSON value stored at *key*.

        Args:
            key: Redis key.

        Returns:
            Deserialised Python object, or ``None`` if the key does not exist.

        Raises:
            RuntimeError: if the stored value cannot be parsed as JSON.
        """
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Redis key {key!r} contains invalid JSON: {exc}") from exc

    async def set_json(
        self,
        key: str,
        data: Any,
        ex: int | None = None,
    ) -> None:
        """Serialise *data* to JSON and store it at *key*.

        Args:
            key: Redis key.
            data: Any JSON-serialisable Python object.
            ex: Optional expiry in seconds.  ``None`` means no expiry.

        Raises:
            RuntimeError: if *data* cannot be serialised to JSON.
        """
        try:
            value = json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Cannot JSON-serialise data for Redis key {key!r}: {exc}") from exc
        await self.set(key, value, ex=ex)
