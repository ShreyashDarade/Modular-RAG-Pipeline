from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.deps import RateLimited, ServiceDep
from src.contracts.models import (
    STREAM_EVENT_NAMES,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    ConversationResponse,
    ErrorBody,
)
from src.core.errors import RagError

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
_EVENT_NAME = {cls: name for name, cls in STREAM_EVENT_NAMES.items()}


#: Legal raw inside a JSON string, but `str.splitlines()` (and so many SSE clients) treats them as line breaks.
_LINE_BREAKS = {"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}


def _sse(event: str, data: dict[str, Any]) -> bytes:
    body = json.dumps(data, ensure_ascii=False)
    for (
        char,
        escaped,
    ) in _LINE_BREAKS.items():  # the same JSON, spelled so no client can mistake it for a line end
        body = body.replace(char, escaped)
    return f"event: {event}\ndata: {body}\n\n".encode()


def _encode(event: ChatStreamEvent) -> bytes:
    return _sse(_EVENT_NAME[type(event)], event.model_dump())


@router.post("", response_model=ChatResponse, dependencies=[RateLimited])
async def chat(payload: ChatRequest, service: ServiceDep):
    """One turn of a conversation. Omit ``conversation_id`` to start a new one."""
    return await service.chat(payload)


@router.post("/stream", dependencies=[RateLimited])
async def chat_stream(payload: ChatRequest, service: ServiceDep):
    """Server-sent events: ``start`` (sources, conversation id), ``delta`` (text), ``end``."""
    events = service.chat_stream(payload)
    # Pull the first event before answering: validation / retrieval errors become proper HTTP
    # errors instead of an error event inside a 200 stream.
    first = await anext(events)

    async def stream() -> AsyncIterator[bytes]:
        try:
            yield _encode(first)
            async for event in events:
                yield _encode(event)
        except RagError as exc:
            yield _sse("error", ErrorBody(detail=exc.public_message, code=exc.code).model_dump())

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, service: ServiceDep):
    return await service.conversation(conversation_id)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, service: ServiceDep) -> None:
    await service.delete_conversation(conversation_id)
