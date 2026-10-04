"""The narrow contract between the public facade and a transport (internal).

Two implementations exist: the HTTP backend (:mod:`turinton_rag._http`) and the embedded backend
(:mod:`turinton_rag.embedded`). Everything the user sees - method names, argument names, defaults, validation,
error types - lives in the facade, once; a backend only moves already-validated request models to wherever the
engine is and brings response models back.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import BinaryIO, Protocol

from src.contracts.models import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    CollectionInfo,
    ConversationResponse,
    DeleteResponse,
    DocumentList,
    IngestResponse,
    JobResponse,
    ModelsResponse,
    RetrieveRequest,
    RetrieveResponse,
)


@dataclass(frozen=True, slots=True)
class Upload:
    """A file to ingest: its name and a binary stream positioned at the start."""

    filename: str
    stream: BinaryIO


@dataclass(frozen=True, slots=True)
class IngestOptions:
    collection: str | None = None
    image_language: str | None = None
    kinds: tuple[str, ...] | None = None
    force: bool = False
    wait: bool | None = None
    timeout: float | None = None


class Backend(Protocol):
    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse: ...

    async def ask(self, request: AskRequest) -> AskResponse: ...

    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]: ...

    async def get_conversation(self, conversation_id: str) -> ConversationResponse: ...

    async def delete_conversation(self, conversation_id: str) -> None: ...

    async def ingest(self, upload: Upload, options: IngestOptions) -> IngestResponse: ...

    async def get_job(self, job_id: str) -> JobResponse: ...

    async def list_documents(self, collection: str | None, limit: int, offset: int) -> DocumentList: ...

    async def delete_document(self, source: str, collection: str | None) -> DeleteResponse: ...

    async def list_collections(self) -> list[CollectionInfo]: ...

    async def list_models(self) -> ModelsResponse: ...

    async def aclose(self) -> None: ...
