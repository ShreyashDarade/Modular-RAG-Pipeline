from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from src.core.types import ChatMessage


class Embedder(Protocol):
    """Turns text into vectors. ``dimensions`` is known up front so indices can be created
    without a probing call."""

    model_id: str
    dimensions: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    model_id: str

    async def complete(self, messages: Sequence[ChatMessage]) -> str: ...

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...
