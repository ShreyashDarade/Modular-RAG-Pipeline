"""Declarative description of *what the system is made of*: named chat models, embedding
models and collections. Loaded from a TOML file (``RAG_CONFIG``) or synthesised from the
classic environment variables. Pure data + cross-reference validation - no I/O beyond reading
the file, no knowledge of any provider or backend.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.errors import ConfigError, NotFoundError
from src.core.types import CONTENT_KINDS, ContentKind

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_INDEX_SUFFIX: dict[ContentKind, str] = {"text": "text", "table": "tables", "image": "images"}


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatModelSpec(_Spec):
    provider: str
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
    base_url: str | None = None
    #: Provider-specific constructor kwargs, passed through untouched.
    options: dict[str, Any] = Field(default_factory=dict)


class EmbeddingModelSpec(_Spec):
    provider: str
    model: str
    #: Output size. Required unless the model's size is built in (OpenAI text-embedding-3-*).
    dimensions: int | None = Field(default=None, gt=0)
    base_url: str | None = None
    batch_size: int | None = Field(default=None, gt=0)
    options: dict[str, Any] = Field(default_factory=dict)


class RerankerSpec(_Spec):
    """A reranker. ``heuristic`` and ``identity`` need nothing else; model-based ones need ``model``."""

    provider: str
    model: str = ""
    base_url: str | None = None
    #: Candidates scored per forward pass (``cross-encoder``). Hosted APIs take all candidates in one request.
    batch_size: int = Field(default=16, gt=0)
    #: Passages are cut to this many characters before scoring (a model truncates by tokens anyway).
    max_chars: int = Field(default=4000, gt=0)
    timeout_seconds: float | None = None
    max_retries: int | None = Field(default=None, ge=0)
    #: Provider-specific settings (``device``, ``max_length``, ``activation`` for ``cross-encoder``).
    options: dict[str, Any] = Field(default_factory=dict)


class ChunkerSpec(_Spec):
    name: str = "recursive"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    min_chunk_size: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class CollectionSpec(_Spec):
    """An independently searchable corpus: its own embedding model, chunker and indices."""

    name: str = ""
    description: str = ""
    embedding_model: str
    chunker: ChunkerSpec = Field(default_factory=ChunkerSpec)
    #: Which content kinds this collection indexes. Others are skipped *by configuration*.
    kinds: tuple[ContentKind, ...] = CONTENT_KINDS
    #: Parsers allowed to feed this collection; ``None`` means every registered parser.
    parsers: tuple[str, ...] | None = None
    index_prefix: str | None = None
    #: Directory under DATA_DIR holding this collection's files. Defaults to the collection name;
    #: "" means DATA_DIR itself (the legacy single-collection layout).
    data_subdir: str | None = None
    #: Explicit index names; overrides ``index_prefix``.
    indices: dict[ContentKind, str] | None = None
    shards: int | None = Field(default=None, gt=0)
    replicas: int | None = Field(default=None, ge=0)
    vector_index_type: str | None = None

    @field_validator("kinds")
    @classmethod
    def _kinds_not_empty(cls, value: tuple[ContentKind, ...]) -> tuple[ContentKind, ...]:
        if not value:
            raise ValueError("a collection must index at least one content kind")
        return value

    @property
    def subdir(self) -> str:
        return self.name if self.data_subdir is None else self.data_subdir

    def index_name(self, kind: ContentKind) -> str:
        if self.indices is not None:
            try:
                return self.indices[kind]
            except KeyError:
                raise ConfigError(
                    f"collection '{self.name}' has no index configured for kind '{kind}'"
                ) from None
        return f"{self.index_prefix or self.name}-{_INDEX_SUFFIX[kind]}"

    def index_names(self, kinds: tuple[ContentKind, ...] | None = None) -> dict[ContentKind, str]:
        return {kind: self.index_name(kind) for kind in (kinds or self.kinds) if kind in self.kinds}


class RagConfig(_Spec):
    default_chat_model: str
    default_collection: str
    query_expander: str = "llm"
    #: Chat model used by the ``llm`` query expander and for chat question condensing.
    utility_model: str
    #: A key of ``reranker_models``, or the name of a built-in that needs no model: ``identity`` (keep the
    #: fused order) or ``heuristic`` (hand-weighted signals; measured to hurt on a public benchmark).
    reranker: str = "identity"
    chat_models: dict[str, ChatModelSpec]
    embedding_models: dict[str, EmbeddingModelSpec]
    reranker_models: dict[str, RerankerSpec] = Field(default_factory=dict)
    collections: dict[str, CollectionSpec]

    @model_validator(mode="before")
    @classmethod
    def _fill_names(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "utility_model" not in data and "default_chat_model" in data:
                data["utility_model"] = data["default_chat_model"]
            raw = data.get("collections") or {}
            data["collections"] = {
                name: ({**spec, "name": name} if isinstance(spec, dict) else spec)
                for name, spec in raw.items()
            }
        return data

    @model_validator(mode="after")
    def _cross_references(self) -> Self:
        for name in (*self.chat_models, *self.embedding_models, *self.reranker_models, *self.collections):
            if not _NAME_RE.match(name):
                raise ValueError(f"invalid name '{name}': use lowercase letters, digits, '-' and '_'")
        for ref, pool, label in (
            (self.default_chat_model, self.chat_models, "default_chat_model"),
            (self.utility_model, self.chat_models, "utility_model"),
            (self.default_collection, self.collections, "default_collection"),
        ):
            if ref not in pool:
                raise ValueError(
                    f"{label} '{ref}' is not defined (have: {', '.join(sorted(pool)) or 'none'})"
                )
        for collection in self.collections.values():
            if collection.embedding_model not in self.embedding_models:
                raise ValueError(
                    f"collection '{collection.name}' uses unknown embedding model '{collection.embedding_model}'"
                )
        seen: dict[str, str] = {}
        for collection in self.collections.values():
            for kind in collection.kinds:
                index = collection.index_name(kind)
                if index in seen:
                    raise ValueError(
                        f"index '{index}' is used by both '{seen[index]}' and '{collection.name}'"
                    )
                seen[index] = collection.name
        return self

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read RAG config {path}: {exc}") from exc
        try:
            return cls.model_validate(raw)
        except ValueError as exc:
            raise ConfigError(f"invalid RAG config {path}: {exc}") from exc

    def reranker_spec(self) -> RerankerSpec:
        return self.reranker_models.get(self.reranker) or RerankerSpec(provider=self.reranker)

    def collection(self, name: str) -> CollectionSpec:
        try:
            return self.collections[name]
        except KeyError:
            raise NotFoundError(
                f"unknown collection '{name}'. Available: {', '.join(sorted(self.collections))}"
            ) from None
