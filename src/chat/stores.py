from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from src.core.errors import UpstreamError
from src.core.registry import Registries
from src.core.types import ChatMessage

if TYPE_CHECKING:
    pass

_MAX_STORED = 200  # hard cap per conversation, independent of how many are sent to the model


class MemoryConversationStore:
    """Per-process; conversations are lost on restart and invisible to other replicas."""

    def __init__(self, max_conversations: int, ttl: int) -> None:
        self._max = max_conversations
        self._ttl = ttl
        self._data: OrderedDict[str, tuple[float, list[ChatMessage]]] = OrderedDict()

    async def load(self, conversation_id: str, limit: int) -> list[ChatMessage]:
        entry = self._data.get(conversation_id)
        if entry is None:
            return []
        expires, messages = entry
        if expires <= time.monotonic():
            del self._data[conversation_id]
            return []
        self._data.move_to_end(conversation_id)
        return list(messages[-limit:]) if limit else []

    async def append(self, conversation_id: str, messages: Sequence[ChatMessage]) -> None:
        _, existing = self._data.get(conversation_id, (0.0, []))
        existing = [*existing, *messages][-_MAX_STORED:]
        self._data[conversation_id] = (time.monotonic() + self._ttl, existing)
        self._data.move_to_end(conversation_id)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    async def delete(self, conversation_id: str) -> bool:
        return self._data.pop(conversation_id, None) is not None

    async def ping(self) -> None:
        return None

    async def close(self) -> None:
        self._data.clear()


class RedisConversationStore:
    def __init__(self, url: str, ttl: int, *, namespace: str = "rag") -> None:
        self._client: aioredis.Redis = aioredis.from_url(url, decode_responses=True, health_check_interval=30)
        self._ttl = ttl
        self._ns = namespace

    def _key(self, conversation_id: str) -> str:
        return f"{self._ns}:chat:{conversation_id}"

    async def load(self, conversation_id: str, limit: int) -> list[ChatMessage]:
        if not limit:
            return []
        try:
            raw = await self._client.lrange(self._key(conversation_id), -limit, -1)
        except RedisError as exc:
            raise UpstreamError(f"redis conversation store failed: {exc}") from exc
        return [ChatMessage(**json.loads(item)) for item in raw]

    async def append(self, conversation_id: str, messages: Sequence[ChatMessage]) -> None:
        key = self._key(conversation_id)
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.rpush(
                    key,
                    *(
                        json.dumps({"role": m.role, "content": m.content}, ensure_ascii=False)
                        for m in messages
                    ),
                )
                pipe.ltrim(key, -_MAX_STORED, -1)
                pipe.expire(key, self._ttl)
                await pipe.execute()
        except RedisError as exc:
            raise UpstreamError(f"redis conversation store failed: {exc}") from exc

    async def delete(self, conversation_id: str) -> bool:
        try:
            return bool(await self._client.delete(self._key(conversation_id)))
        except RedisError as exc:
            raise UpstreamError(f"redis conversation store failed: {exc}") from exc

    async def ping(self) -> None:
        try:
            await self._client.ping()
        except RedisError as exc:
            raise UpstreamError(f"redis ping failed: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()


def register_builtin_stores(registries: Registries) -> None:
    registries.conversation_stores.register(
        "memory", lambda s: MemoryConversationStore(s.chat_memory_conversations, s.chat_history_ttl_seconds)
    )
    registries.conversation_stores.register(
        "redis",
        lambda s: RedisConversationStore(
            s.redis_url, s.chat_history_ttl_seconds, namespace=s.redis_namespace
        ),
    )
