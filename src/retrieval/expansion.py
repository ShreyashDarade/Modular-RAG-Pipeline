from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING

from src.core.errors import RequestTimeoutError
from src.core.registry import Registries
from src.core.types import ChatMessage
from src.ports.models import ChatModel
from src.ports.runtime import Cache

if TYPE_CHECKING:
    from src.core.config import Settings

_PROMPT = (
    "You are a search query rewriter. Given a user query, generate 3 diverse "
    "alternative phrasings to improve search recall. Keep variations concise. "
    "Return only the variations as a bullet list, one per line."
)


class IdentityQueryExpander:
    """Selected explicitly (``query_expander = "identity"``) when expansion is not wanted."""

    async def expand(self, query: str) -> list[str]:
        return [query]


class LlmQueryExpander:
    """Asks a chat model for alternative phrasings. Results are cached (the same question is
    asked often, and the model is the slowest step of retrieval). A model failure or timeout is
    an error: callers that do not want an LLM in the retrieval path configure ``identity``."""

    def __init__(self, chat: ChatModel, cache: Cache, *, timeout: float, ttl: int) -> None:
        self._chat = chat
        self._cache = cache
        self._timeout = timeout
        self._ttl = ttl

    async def expand(self, query: str) -> list[str]:
        key = f"qe:{self._chat.model_id}:{hashlib.sha256(query.encode()).hexdigest()}"
        cached = await self._cache.get(key)
        if cached is not None:
            return json.loads(cached)
        try:
            async with asyncio.timeout(self._timeout):
                text = await self._chat.complete(
                    [ChatMessage("system", _PROMPT), ChatMessage("user", f"Query: {query}")]
                )
        except TimeoutError as exc:
            raise RequestTimeoutError(f"query expansion did not finish within {self._timeout}s") from exc
        variants = [query, *self._parse(text, query)]
        await self._cache.set(key, json.dumps(variants).encode(), self._ttl)
        return variants

    @staticmethod
    def _parse(text: str, query: str) -> list[str]:
        seen = {query.lower().strip()}
        variants: list[str] = []
        for line in text.splitlines():
            cleaned = line.strip(" -•*")
            normalized = cleaned.lower().strip()
            if len(cleaned) >= 3 and normalized not in seen:
                variants.append(cleaned)
                seen.add(normalized)
        return variants[:3]


def build_llm_expander(settings: Settings, chat: ChatModel, cache: Cache) -> LlmQueryExpander:
    return LlmQueryExpander(
        chat,
        cache,
        timeout=settings.query_expansion_timeout_seconds,
        ttl=settings.query_expansion_cache_ttl_seconds,
    )


def build_identity_expander(settings: Settings, chat: ChatModel, cache: Cache) -> IdentityQueryExpander:
    return IdentityQueryExpander()


def register_builtin_expanders(registries: Registries) -> None:
    registries.query_expanders.register("llm", build_llm_expander)
    registries.query_expanders.register("identity", build_identity_expander)
