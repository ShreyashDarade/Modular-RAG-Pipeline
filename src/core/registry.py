"""Name -> factory registries: the extension mechanism (open/closed principle).

Adding a parser, model provider, chunker, reranker, ... means registering a factory under a
name. Nothing that *uses* the component is edited. Asking for a name nobody registered is a
:class:`UnknownComponentError` listing the valid names - there is no implicit default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.errors import ConfigError, UnknownComponentError

if TYPE_CHECKING:
    from src.ports.models import ChatModel, Embedder
    from src.ports.parsing import Chunker, OcrEngine, Parser
    from src.ports.retrieval import QueryExpander, Reranker
    from src.ports.runtime import Cache, ConversationStore, JobBackend, RateLimiter


class Registry[T]:
    """A named set of factories producing ``T``."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T], *, replace: bool = False) -> None:
        if name in self._factories and not replace:
            raise ConfigError(f"{self.kind} '{name}' is already registered")
        self._factories[name] = factory

    def create(self, name: str, /, *args: Any, **kwargs: Any) -> T:
        try:
            factory = self._factories[name]
        except KeyError:
            raise UnknownComponentError(
                f"Unknown {self.kind} '{name}'. Registered: {', '.join(sorted(self._factories)) or '(none)'}"
            ) from None
        return factory(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories


@dataclass
class Registries:
    """Every extension point in one place. Built by the composition root, handed to plugins."""

    parsers: Registry[Parser] = field(default_factory=lambda: Registry("parser"))
    ocr_engines: Registry[OcrEngine] = field(default_factory=lambda: Registry("OCR engine"))
    chunkers: Registry[Chunker] = field(default_factory=lambda: Registry("chunker"))
    chat_providers: Registry[ChatModel] = field(default_factory=lambda: Registry("chat provider"))
    embedding_providers: Registry[Embedder] = field(default_factory=lambda: Registry("embedding provider"))
    query_expanders: Registry[QueryExpander] = field(default_factory=lambda: Registry("query expander"))
    rerankers: Registry[Reranker] = field(default_factory=lambda: Registry("reranker"))
    caches: Registry[Cache] = field(default_factory=lambda: Registry("cache backend"))
    rate_limiters: Registry[RateLimiter] = field(default_factory=lambda: Registry("rate limiter"))
    conversation_stores: Registry[ConversationStore] = field(
        default_factory=lambda: Registry("conversation store")
    )
    job_backends: Registry[JobBackend] = field(default_factory=lambda: Registry("job backend"))
