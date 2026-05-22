"""Memory service: short-term (Redis) + long-term (pgvector episodic) memory.

Short-term: Redis key ``conv:{conversation_id}:history``, TTL 24 h, JSON list
of ``{role, content}`` dicts.

Long-term: per-user Memory rows in Postgres with a pgvector embedding column.
Every write produces an AuditLog row — required by CLAUDE.md.
"""

from __future__ import annotations

import json
import uuid

import structlog
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.errors import NotFoundError
from app.infra.redaction import redact_text
from app.rag.embeddings import EmbeddingModel
from app.repositories.models import AuditLog, Memory

logger = structlog.get_logger(__name__)

_HISTORY_TTL_SECONDS = 86_400  # 24 h (matches Settings.redis_conversation_ttl_seconds)


def _history_key(conversation_id: str) -> str:
    return f"conv:{conversation_id}:history"


def _conv_meta_key(conversation_id: str) -> str:
    return f"conv:{conversation_id}:meta"


def _user_conv_index_key(user_id: uuid.UUID) -> str:
    """Sorted set of a user's conversation IDs (score = updated_at unix ts)."""
    return f"user:{user_id}:conversations"


class MemoryService:
    """Short-term (Redis) and long-term (pgvector) memory for conversations.

    Args:
        redis: Async Redis client from app.state.
        session_factory: SQLAlchemy async session factory from app.state.
        embedder: Embedding model used to embed memory summaries.
    """

    def __init__(
        self,
        redis: Redis,  # type: ignore[type-arg]
        session_factory: async_sessionmaker[AsyncSession],
        embedder: EmbeddingModel,
    ) -> None:
        self._redis = redis
        self._session_factory = session_factory
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Short-term history (Redis)
    # ------------------------------------------------------------------

    async def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Return the conversation history from Redis.

        Args:
            conversation_id: Unique conversation identifier.

        Returns:
            List of ``{role, content}`` dicts, oldest first.
        """
        raw: str | None = await self._redis.get(_history_key(conversation_id))
        if raw is None:
            return []
        try:
            data: list[dict[str, str]] = json.loads(raw)
            return data
        except json.JSONDecodeError:
            logger.warning("memory.history_parse_error", conversation_id=conversation_id)
            return []

    async def append_message(self, conversation_id: str, role: str, content: str) -> None:
        """Append a single message to the Redis conversation history.

        Resets the TTL on every write so active conversations never expire
        mid-flow.

        Args:
            conversation_id: Unique conversation identifier.
            role: Message author — ``"user"``, ``"assistant"``, or ``"tool"``.
            content: Message text (will be stored as-is; redaction is the
                caller's responsibility before memory writes land on disk).
        """
        key = _history_key(conversation_id)
        history = await self.get_history(conversation_id)
        history.append({"role": role, "content": content})
        await self._redis.set(key, json.dumps(history), ex=_HISTORY_TTL_SECONDS)
        logger.debug(
            "memory.history_appended",
            conversation_id=conversation_id,
            role=role,
            history_len=len(history),
        )

    async def clear_history(self, conversation_id: str) -> None:
        """Delete the Redis history for a conversation.

        Args:
            conversation_id: Unique conversation identifier.
        """
        await self._redis.delete(_history_key(conversation_id))
        logger.info("memory.history_cleared", conversation_id=conversation_id)

    # ------------------------------------------------------------------
    # Conversation index per user (multi-tenant isolation)
    # ------------------------------------------------------------------

    async def verify_conversation_owner(
        self, conversation_id: str, user_id: uuid.UUID
    ) -> bool:
        """Return True if the conversation has no recorded owner OR is owned by user_id.

        Conversations created before this index existed have no meta record; we
        treat those as "claim on first touch" so legacy sessions don't break.
        """
        meta = await self._redis.hgetall(_conv_meta_key(conversation_id))  # type: ignore[misc]
        if not meta:
            return True
        owner = meta.get("user_id", "")
        return str(owner) == str(user_id)

    async def touch_conversation(
        self,
        conversation_id: str,
        user_id: uuid.UUID,
        first_message: str | None = None,
    ) -> None:
        """Mark this conversation as owned by user_id and bump its updated_at.

        Idempotent — safe to call on every chat turn. Sets the title from the
        first user message when one isn't already stored.
        """
        import time

        meta_key = _conv_meta_key(conversation_id)
        index_key = _user_conv_index_key(user_id)
        now = time.time()

        existing = await self._redis.hgetall(meta_key)  # type: ignore[misc]
        updates: dict[str, str] = {
            "user_id": str(user_id),
            "updated_at": str(now),
        }
        if "created_at" not in existing:
            updates["created_at"] = str(now)
        if first_message and not existing.get("title"):
            # Use the first user message (truncated) as the conversation title.
            title = first_message.strip().replace("\n", " ")
            updates["title"] = title[:80] if len(title) <= 80 else title[:77] + "..."

        await self._redis.hset(meta_key, mapping=updates)  # type: ignore[misc]
        await self._redis.expire(meta_key, _HISTORY_TTL_SECONDS * 30)  # 30 days
        await self._redis.zadd(index_key, {conversation_id: now})  # type: ignore[misc]

    async def list_user_conversations(
        self, user_id: uuid.UUID, limit: int = 50
    ) -> list[dict[str, str]]:
        """Return the user's recent conversations, newest first.

        Each entry has ``{conversation_id, title, created_at, updated_at}``.
        Conversations whose meta has expired are silently dropped from the
        returned list (and from the index on demand).
        """
        # zrevrange returns newest-first
        ids: list[str] = await self._redis.zrevrange(  # type: ignore[misc]
            _user_conv_index_key(user_id), 0, limit - 1
        )
        out: list[dict[str, str]] = []
        for cid in ids:
            meta = await self._redis.hgetall(_conv_meta_key(cid))  # type: ignore[misc]
            if not meta:
                # Expired meta — drop from index too so the list stays clean.
                await self._redis.zrem(_user_conv_index_key(user_id), cid)  # type: ignore[misc]
                continue
            out.append(
                {
                    "conversation_id": cid,
                    "title": meta.get("title", "New conversation"),
                    "created_at": meta.get("created_at", ""),
                    "updated_at": meta.get("updated_at", ""),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Long-term episodic memory (pgvector)
    # ------------------------------------------------------------------

    async def save_memory(
        self,
        user_id: uuid.UUID,
        summary: str,
        actor_id: uuid.UUID | None = None,
    ) -> Memory:
        """Embed a conversation summary and persist it to Postgres.

        Every call writes an AuditLog row so the write is traceable.
        The summary is redacted before storage to prevent secrets leaking
        into the memory store.

        Args:
            user_id: Owner of this memory.
            summary: Conversation summary text.
            actor_id: Who triggered the write (defaults to user_id).

        Returns:
            The persisted Memory ORM instance.
        """
        redacted_summary = redact_text(summary)
        embedding: list[float] = await self._embedder.embed(redacted_summary)

        effective_actor = actor_id if actor_id is not None else user_id

        async with self._session_factory() as session:
            memory = Memory(
                id=uuid.uuid4(),
                user_id=user_id,
                summary=redacted_summary,
                embedding=embedding,
            )
            session.add(memory)

            audit = AuditLog(
                id=uuid.uuid4(),
                actor_id=effective_actor,
                action="memory_write",
                target_type="memory",
                target_id=memory.id,
                extra_metadata={"summary_length": len(redacted_summary)},
            )
            session.add(audit)

            await session.commit()
            await session.refresh(memory)

        logger.info(
            "memory.saved",
            user_id=str(user_id),
            memory_id=str(memory.id),
            summary_length=len(redacted_summary),
        )
        return memory

    async def search_memories(
        self,
        user_id: uuid.UUID,
        query: str,
        top_k: int = 3,
    ) -> list[Memory]:
        """Find the most relevant long-term memories for a query.

        Uses pgvector cosine distance to rank results.

        Args:
            user_id: Restrict search to this user's memories.
            query: Natural-language query (e.g., the current user message).
            top_k: Maximum number of memories to return.

        Returns:
            List of Memory ORM instances, most similar first.
        """
        query_embedding: list[float] = await self._embedder.embed(query)

        async with self._session_factory() as session:
            # pgvector cosine distance operator: <=>
            stmt = (
                select(Memory)
                .where(Memory.user_id == user_id)
                .order_by(Memory.embedding.cosine_distance(query_embedding))  # type: ignore[attr-defined]
                .limit(top_k)
            )
            result = await session.execute(stmt)
            memories: list[Memory] = list(result.scalars().all())

        logger.info(
            "memory.searched",
            user_id=str(user_id),
            query_length=len(query),
            results_found=len(memories),
        )
        return memories

    async def delete_memory(
        self,
        memory_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Delete a memory, verifying ownership.

        Args:
            memory_id: Memory to delete.
            user_id: Caller's user ID — must own the memory.

        Raises:
            NotFoundError: If the memory does not exist or is owned by
                another user.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Memory).where(Memory.id == memory_id, Memory.user_id == user_id)
            )
            memory: Memory | None = result.scalar_one_or_none()
            if memory is None:
                raise NotFoundError(f"Memory {memory_id} not found")

            audit = AuditLog(
                id=uuid.uuid4(),
                actor_id=user_id,
                action="memory_delete",
                target_type="memory",
                target_id=memory_id,
                extra_metadata=None,
            )
            session.add(audit)
            await session.delete(memory)
            await session.commit()

        logger.info("memory.deleted", memory_id=str(memory_id), user_id=str(user_id))

    async def list_memories(
        self,
        user_id: uuid.UUID,
    ) -> list[Memory]:
        """Return all long-term memories for a user, newest first.

        Args:
            user_id: Owner filter.

        Returns:
            List of Memory ORM instances.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at.desc())
            )
            memories: list[Memory] = list(result.scalars().all())
        return memories
