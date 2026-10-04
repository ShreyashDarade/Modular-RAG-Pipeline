from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING

from src.chat.prompts import answer_messages
from src.core.types import AnswerResult, ChatMessage, RetrievalResult, RetrievalScope, RetrievedDocument
from src.models.registry import ModelRegistry
from src.retrieval.pipeline import RetrievalPipeline

if TYPE_CHECKING:
    from src.core.config import Settings


class AnswerService:
    """Retrieve, then generate with the chosen chat model. Retrieval and model failures
    propagate as typed errors - there is no canned "sorry, something went wrong" answer."""

    def __init__(self, *, retrieval: RetrievalPipeline, models: ModelRegistry, settings: Settings) -> None:
        self._retrieval = retrieval
        self._models = models
        self._top_k = settings.rerank_top_k

    async def prepare(
        self, query: str, scope: RetrievalScope, history: Sequence[ChatMessage] = ()
    ) -> tuple[RetrievalResult, list[RetrievedDocument], list[ChatMessage]]:
        retrieval = await self._retrieval.retrieve(query, scope)
        documents = retrieval.documents[: self._top_k]
        return retrieval, documents, answer_messages(query, retrieval.expanded_queries, documents, history)

    async def ask(self, query: str, scope: RetrievalScope, model: str | None = None) -> AnswerResult:
        chat = self._models.chat(model)
        retrieval, documents, messages = await self.prepare(query, scope)
        answer = await chat.complete(messages)
        return AnswerResult(
            query=retrieval.query,
            expanded_queries=retrieval.expanded_queries,
            answer=answer,
            model=chat.model_id,
            documents=documents,
        )

    def stream(self, messages: Sequence[ChatMessage], model: str | None = None) -> AsyncIterator[str]:
        return self._models.chat(model).stream(messages)
