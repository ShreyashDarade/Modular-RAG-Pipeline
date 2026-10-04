from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.deps import ContainerDep, RateLimited, deadline
from src.api.schemas import AskContextItem, ChatMessageSchema, ChatRequest, ChatResponse, ConversationResponse
from src.chat.service import ChatDelta, ChatEvent, ChatFinished, ChatStarted
from src.core.errors import RagError

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@router.post("", response_model=ChatResponse, dependencies=[RateLimited])
async def chat(payload: ChatRequest, container: ContainerDep):
    """One turn of a conversation. Omit ``conversation_id`` to start a new one."""
    scope = container.retrieval.scope(payload.collections, payload.kinds, payload.sources)
    async with deadline(container):
        result = await container.chat.chat(
            payload.message, scope, conversation_id=payload.conversation_id, model=payload.model
        )
    return ChatResponse(
        conversation_id=result.conversation_id,
        answer=result.answer,
        standalone_query=result.standalone_query,
        model=result.model,
        context=[AskContextItem.ranked(i, d) for i, d in enumerate(result.documents, start=1)],
    )


@router.post("/stream", dependencies=[RateLimited])
async def chat_stream(payload: ChatRequest, container: ContainerDep):
    """Server-sent events: ``start`` (sources, conversation id), ``delta`` (text), ``end``."""
    scope = container.retrieval.scope(payload.collections, payload.kinds, payload.sources)
    events = container.chat.stream(
        payload.message, scope, conversation_id=payload.conversation_id, model=payload.model
    )
    # Pull the first event before answering: validation / retrieval errors become proper HTTP
    # errors instead of an error event inside a 200 stream.
    async with deadline(container):
        first = await anext(events)

    async def stream() -> AsyncIterator[bytes]:
        try:
            async with deadline(container):  # the whole answer, not only the first event
                yield _encode(first)
                async for event in events:
                    yield _encode(event)
        except RagError as exc:
            yield _sse("error", {"detail": exc.public_message, "code": exc.code})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def _encode(event: ChatEvent) -> bytes:
    match event:
        case ChatStarted():
            return _sse(
                "start",
                {
                    "conversation_id": event.conversation_id,
                    "standalone_query": event.standalone_query,
                    "expanded_queries": event.expanded_queries,
                    "model": event.model,
                    "context": [
                        AskContextItem.ranked(i, d).model_dump()
                        for i, d in enumerate(event.documents, start=1)
                    ],
                },
            )
        case ChatDelta():
            return _sse("delta", {"text": event.text})
        case ChatFinished():
            return _sse("end", {"answer": event.answer})
    raise AssertionError(f"unhandled chat event {event!r}")


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, container: ContainerDep):
    messages = await container.chat.history(conversation_id)
    return ConversationResponse(
        conversation_id=conversation_id,
        messages=[ChatMessageSchema(role=m.role, content=m.content) for m in messages],
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, container: ContainerDep) -> None:
    await container.chat.delete(conversation_id)
