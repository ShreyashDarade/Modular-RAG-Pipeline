"""The request and response models shared by the HTTP API and both SDK transports.

They are the wire contract: ``/api/v1`` changes to them are additive only. Clients ignore unknown response
fields (so a newer server never breaks an older SDK) but validate required ones strictly.
"""

from src.contracts.models import (
    STREAM_EVENT_NAMES,
    AskRequest,
    AskResponse,
    ChatDeltaEvent,
    ChatEndEvent,
    ChatRequest,
    ChatResponse,
    ChatStartEvent,
    ChatStreamEvent,
    CollectionInfo,
    ContextChunk,
    ConversationMessage,
    ConversationResponse,
    DeleteResponse,
    DocumentInfo,
    DocumentList,
    ErrorBody,
    IngestResponse,
    JobResponse,
    ModelInfo,
    ModelsResponse,
    RetrievedChunk,
    RetrieveRequest,
    RetrieveResponse,
)
from src.core.types import ContentKind

__all__ = [
    "STREAM_EVENT_NAMES",
    "AskRequest",
    "AskResponse",
    "ChatDeltaEvent",
    "ChatEndEvent",
    "ChatRequest",
    "ChatResponse",
    "ChatStartEvent",
    "ChatStreamEvent",
    "CollectionInfo",
    "ContentKind",
    "ContextChunk",
    "ConversationMessage",
    "ConversationResponse",
    "DeleteResponse",
    "DocumentInfo",
    "DocumentList",
    "ErrorBody",
    "IngestResponse",
    "JobResponse",
    "ModelInfo",
    "ModelsResponse",
    "RetrieveRequest",
    "RetrieveResponse",
    "RetrievedChunk",
]
