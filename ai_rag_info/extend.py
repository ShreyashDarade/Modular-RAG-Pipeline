"""The extension contract: what a plug-in implements and how it registers.

A plug-in is a module with ``register(registries)``, enabled with ``PLUGINS=["pkg.module"]``. It adds
components *by name* to the registries below; the engine never discovers anything implicitly. Each port
is a small ``Protocol`` - implement only what it lists. ``ai_rag_info.testing`` has a conformance check
for each, so a plug-in can prove it honours the contract.
"""

from collections.abc import Callable

from src.core.registry import Registries, Registry
from src.core.types import ChatMessage, RetrievedDocument
from src.ports.models import ChatModel, Embedder
from src.ports.parsing import (
    ChunkDraft,
    Chunker,
    ImageRef,
    OcrEngine,
    OcrResult,
    ParsedUnit,
    Parser,
    TableBlock,
    TextBlock,
)
from src.ports.retrieval import QueryExpander, Reranker
from src.ports.runtime import Cache, ConversationStore, JobBackend, RateLimiter

#: The signature of a plug-in's ``register`` function.
PluginRegister = Callable[[Registries], None]

__all__ = [
    "Cache",
    "ChatMessage",
    "ChatModel",
    "ChunkDraft",
    "Chunker",
    "ConversationStore",
    "Embedder",
    "ImageRef",
    "JobBackend",
    "OcrEngine",
    "OcrResult",
    "ParsedUnit",
    "Parser",
    "PluginRegister",
    "QueryExpander",
    "RateLimiter",
    "Registries",
    "Registry",
    "Reranker",
    "RetrievedDocument",
    "TableBlock",
    "TextBlock",
]
