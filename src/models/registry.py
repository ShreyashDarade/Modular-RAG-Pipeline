"""Resolve named model profiles to ready-to-use ports.

Profiles are validated and built *once, at start-up* (``build_all``): a bad provider name, a
missing API key or a missing optional package stops the process immediately instead of
failing on the first request that happens to use that model.
"""

from __future__ import annotations

import hashlib
import json

from src.core.config import Settings
from src.core.errors import NotFoundError
from src.core.registry import Registries
from src.core.specs import EmbeddingModelSpec, RagConfig
from src.models.adapters import CachingEmbedder
from src.ports.models import ChatModel, Embedder
from src.ports.runtime import Cache


def _identity(spec: EmbeddingModelSpec) -> str:
    """Everything a vector depends on. Provider options matter (prefixes, pooling, revision...)."""
    options = hashlib.sha256(json.dumps(spec.options, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return f"{spec.provider}/{spec.model}/{spec.base_url or ''}/{options}"


class ModelRegistry:
    def __init__(
        self,
        config: RagConfig,
        settings: Settings,
        registries: Registries,
        *,
        query_cache: Cache | None = None,
    ) -> None:
        self._chat: dict[str, ChatModel] = {}
        self._embedders: dict[str, Embedder] = {}
        self.default_chat = config.default_chat_model
        for name, chat_spec in config.chat_models.items():
            self._chat[name] = registries.chat_providers.create(chat_spec.provider, name, chat_spec, settings)
        for name, embedding_spec in config.embedding_models.items():
            inner = registries.embedding_providers.create(
                embedding_spec.provider, name, embedding_spec, settings
            )
            self._embedders[name] = CachingEmbedder(
                inner,
                max_entries=settings.embedding_cache_entries,
                shared=query_cache,
                shared_ttl=settings.cache_ttl_seconds,
                identity=_identity(embedding_spec),
            )

    def chat(self, name: str | None = None) -> ChatModel:
        key = name or self.default_chat
        try:
            return self._chat[key]
        except KeyError:
            raise NotFoundError(
                f"unknown chat model '{key}'. Available: {', '.join(sorted(self._chat))}"
            ) from None

    def embedder(self, name: str) -> Embedder:
        try:
            return self._embedders[name]
        except KeyError:
            raise NotFoundError(
                f"unknown embedding model '{name}'. Available: {', '.join(sorted(self._embedders))}"
            ) from None

    def chat_names(self) -> list[str]:
        return sorted(self._chat)

    def embedding_names(self) -> list[str]:
        return sorted(self._embedders)
