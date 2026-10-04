from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.chat.answer import AnswerService
from src.chat.prompts import condense_messages
from src.core.errors import InvalidRequestError, NotFoundError
from src.core.types import ChatMessage, ChatResult, RetrievalScope, RetrievedDocument
from src.models.registry import ModelRegistry
from src.ports.runtime import ConversationStore

if TYPE_CHECKING:
    from src.core.config import Settings


@dataclass(slots=True)
class ChatStarted:
    conversation_id: str
    standalone_query: str
    expanded_queries: list[str]
    documents: list[RetrievedDocument]
    model: str


@dataclass(slots=True)
class ChatDelta:
    text: str


@dataclass(slots=True)
class ChatFinished:
    answer: str


ChatEvent = ChatStarted | ChatDelta | ChatFinished


class ChatService:
    """Multi-turn chat over retrieval.

    Each turn: load the conversation, rewrite a follow-up into a standalone query (so retrieval
    does not need the history), retrieve from the selected scope, answer with the history in
    the prompt using the chosen model, and record the turn. A conversation id is issued by the
    server on the first turn; presenting an unknown or expired id is an error, not a new chat.
    """

    def __init__(
        self,
        *,
        answers: AnswerService,
        models: ModelRegistry,
        store: ConversationStore,
        utility_model: str,
        settings: Settings,
    ) -> None:
        self._answers = answers
        self._models = models
        self._store = store
        self._utility_model = utility_model
        self._history_limit = settings.chat_history_messages
        self._condense = settings.chat_condense_questions
        self._max_chars = settings.max_query_chars

    async def _begin(self, conversation_id: str | None, message: str) -> tuple[str, list[ChatMessage], str]:
        message = message.strip()
        if not message:
            raise InvalidRequestError("message must not be empty")
        if len(message) > self._max_chars:
            raise InvalidRequestError(f"message is longer than {self._max_chars} characters")
        if conversation_id is None:
            return uuid.uuid4().hex, [], message
        history = await self._store.load(conversation_id, self._history_limit)
        if not history:
            raise NotFoundError(f"conversation '{conversation_id}' not found (unknown or expired)")
        standalone = message
        if self._condense:
            standalone = (
                await self._models.chat(self._utility_model).complete(condense_messages(history, message))
            ).strip() or message
        return conversation_id, history, standalone

    async def chat(
        self,
        message: str,
        scope: RetrievalScope,
        *,
        conversation_id: str | None = None,
        model: str | None = None,
    ) -> ChatResult:
        conversation_id, history, standalone = await self._begin(conversation_id, message)
        chat = self._models.chat(model)
        retrieval, documents, messages = await self._answers.prepare(standalone, scope, history)
        answer = await chat.complete(messages)
        await self._store.append(
            conversation_id, [ChatMessage("user", message.strip()), ChatMessage("assistant", answer)]
        )
        return ChatResult(
            conversation_id=conversation_id,
            answer=answer,
            standalone_query=retrieval.query,
            model=chat.model_id,
            documents=documents,
        )

    async def stream(
        self,
        message: str,
        scope: RetrievalScope,
        *,
        conversation_id: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[ChatEvent]:
        conversation_id, history, standalone = await self._begin(conversation_id, message)
        chat = self._models.chat(model)
        retrieval, documents, messages = await self._answers.prepare(standalone, scope, history)
        yield ChatStarted(
            conversation_id, retrieval.query, retrieval.expanded_queries, documents, chat.model_id
        )
        parts: list[str] = []
        async for text in chat.stream(messages):
            parts.append(text)
            yield ChatDelta(text)
        answer = "".join(parts)
        await self._store.append(
            conversation_id, [ChatMessage("user", message.strip()), ChatMessage("assistant", answer)]
        )
        yield ChatFinished(answer)

    async def history(self, conversation_id: str) -> list[ChatMessage]:
        messages = await self._store.load(conversation_id, 200)
        if not messages:
            raise NotFoundError(f"conversation '{conversation_id}' not found (unknown or expired)")
        return messages

    async def delete(self, conversation_id: str) -> None:
        if not await self._store.delete(conversation_id):
            raise NotFoundError(f"conversation '{conversation_id}' not found (unknown or expired)")
