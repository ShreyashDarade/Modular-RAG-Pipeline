"""Provider factories, registered by name. Heavy SDKs are imported inside the factory, so a
provider only costs anything when a profile actually uses it - and a missing optional package is
a :class:`ProviderUnavailableError` naming the extra to install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

from src.core.errors import ConfigError, ProviderUnavailableError
from src.core.registry import Registries
from src.models.adapters import LangChainChatModel, LangChainEmbedder
from src.runtime.concurrency import Bulkhead

if TYPE_CHECKING:
    from src.core.config import Settings
    from src.core.specs import ChatModelSpec, EmbeddingModelSpec
    from src.ports.models import ChatModel, Embedder

#: Output sizes of embedding models whose dimensionality is fixed and well known.
KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _import(module: str, extra: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ProviderUnavailableError(
            f"{module} is not installed; install the '{extra}' extra: pip install 'ai-rag-info[{extra}]'"
        ) from exc


def _secret(value: SecretStr | None, env_name: str, provider: str) -> SecretStr:
    if value is None or not value.get_secret_value():
        raise ConfigError(f"{provider}: {env_name} is required")
    return value


def _chat_kwargs(spec: ChatModelSpec, settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "timeout": spec.timeout_seconds or settings.model_timeout_seconds,
        "max_retries": settings.model_max_retries if spec.max_retries is None else spec.max_retries,
    }
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    return kwargs


def _bulkhead(provider: str, settings: Settings) -> Bulkhead:
    return Bulkhead(settings.model_max_concurrency, name=provider)


# --- chat ----------------------------------------------------------------------------------
def openai_chat(model_id: str, spec: ChatModelSpec, settings: Settings) -> ChatModel:
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(
        model=spec.model,
        api_key=_secret(settings.openai_api_key, "OPENAI_API_KEY", "openai"),
        base_url=spec.base_url,
        max_completion_tokens=spec.max_tokens,
        **_chat_kwargs(spec, settings),
        **spec.options,
    )
    return LangChainChatModel(model_id, chat, service="openai", bulkhead=_bulkhead("openai", settings))


def azure_openai_chat(model_id: str, spec: ChatModelSpec, settings: Settings) -> ChatModel:
    from langchain_openai import AzureChatOpenAI

    chat = AzureChatOpenAI(
        azure_deployment=spec.model,
        azure_endpoint=spec.base_url or settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=_secret(settings.azure_openai_api_key, "AZURE_OPENAI_API_KEY", "azure_openai"),
        max_completion_tokens=spec.max_tokens,
        **_chat_kwargs(spec, settings),
        **spec.options,
    )
    return LangChainChatModel(
        model_id, chat, service="azure_openai", bulkhead=_bulkhead("azure_openai", settings)
    )


def anthropic_chat(model_id: str, spec: ChatModelSpec, settings: Settings) -> ChatModel:
    module = _import("langchain_anthropic", "anthropic")
    chat = module.ChatAnthropic(
        model=spec.model,
        api_key=_secret(settings.anthropic_api_key, "ANTHROPIC_API_KEY", "anthropic"),
        base_url=spec.base_url,
        max_tokens=spec.max_tokens or 2048,
        **_chat_kwargs(spec, settings),
        **spec.options,
    )
    return LangChainChatModel(model_id, chat, service="anthropic", bulkhead=_bulkhead("anthropic", settings))


def google_chat(model_id: str, spec: ChatModelSpec, settings: Settings) -> ChatModel:
    module = _import("langchain_google_genai", "google")
    chat = module.ChatGoogleGenerativeAI(
        model=spec.model,
        api_key=_secret(settings.google_api_key, "GOOGLE_API_KEY", "google"),
        max_tokens=spec.max_tokens,
        **_chat_kwargs(spec, settings),
        **spec.options,
    )
    return LangChainChatModel(model_id, chat, service="google", bulkhead=_bulkhead("google", settings))


def ollama_chat(model_id: str, spec: ChatModelSpec, settings: Settings) -> ChatModel:
    module = _import("langchain_ollama", "ollama")
    kwargs: dict[str, Any] = {}
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    if spec.max_tokens is not None:
        kwargs["num_predict"] = spec.max_tokens
    chat = module.ChatOllama(
        model=spec.model, base_url=spec.base_url or settings.ollama_base_url, **kwargs, **spec.options
    )
    return LangChainChatModel(model_id, chat, service="ollama", bulkhead=_bulkhead("ollama", settings))


# --- embeddings ----------------------------------------------------------------------------
def _dimensions(model_id: str, spec: EmbeddingModelSpec) -> int:
    if spec.dimensions is not None:
        return spec.dimensions
    if spec.model in KNOWN_DIMENSIONS and spec.provider in ("openai", "azure_openai"):
        return KNOWN_DIMENSIONS[spec.model]
    raise ConfigError(
        f"embedding model '{model_id}' ({spec.provider}/{spec.model}): set `dimensions` explicitly - "
        "the pipeline never probes the provider to guess it"
    )


def _embedder(
    model_id: str,
    spec: EmbeddingModelSpec,
    settings: Settings,
    embeddings: Any,
    service: str,
    *,
    batch_queries: bool,
) -> Embedder:
    return LangChainEmbedder(
        model_id,
        embeddings,
        dimensions=_dimensions(model_id, spec),
        batch_size=spec.batch_size or settings.embedding_batch_size,
        service=service,
        bulkhead=_bulkhead(service, settings),
        batch_queries=batch_queries,
    )


def openai_embeddings(model_id: str, spec: EmbeddingModelSpec, settings: Settings) -> Embedder:
    from langchain_openai import OpenAIEmbeddings

    dims = _dimensions(model_id, spec)
    shortened = spec.dimensions is not None and spec.model.startswith("text-embedding-3")
    embeddings = OpenAIEmbeddings(
        model=spec.model,
        api_key=_secret(settings.openai_api_key, "OPENAI_API_KEY", "openai"),
        base_url=spec.base_url,
        dimensions=dims if shortened else None,
        timeout=settings.model_timeout_seconds,
        max_retries=settings.model_max_retries,
        # chunks and queries are far below the 8k-token input limit; skipping the client-side
        # tiktoken pass saves CPU and works with OpenAI-compatible servers
        check_embedding_ctx_length=False,
        chunk_size=spec.batch_size or settings.embedding_batch_size,
        **spec.options,
    )
    return _embedder(model_id, spec, settings, embeddings, "openai", batch_queries=True)


def azure_openai_embeddings(model_id: str, spec: EmbeddingModelSpec, settings: Settings) -> Embedder:
    from langchain_openai import AzureOpenAIEmbeddings

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=spec.model,
        azure_endpoint=spec.base_url or settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=_secret(settings.azure_openai_api_key, "AZURE_OPENAI_API_KEY", "azure_openai"),
        timeout=settings.model_timeout_seconds,
        max_retries=settings.model_max_retries,
        check_embedding_ctx_length=False,
        **spec.options,
    )
    return _embedder(model_id, spec, settings, embeddings, "azure_openai", batch_queries=True)


def google_embeddings(model_id: str, spec: EmbeddingModelSpec, settings: Settings) -> Embedder:
    module = _import("langchain_google_genai", "google")
    embeddings = module.GoogleGenerativeAIEmbeddings(
        model=spec.model,
        api_key=_secret(settings.google_api_key, "GOOGLE_API_KEY", "google"),
        output_dimensionality=spec.dimensions,
        **spec.options,
    )
    return _embedder(model_id, spec, settings, embeddings, "google", batch_queries=False)


def ollama_embeddings(model_id: str, spec: EmbeddingModelSpec, settings: Settings) -> Embedder:
    module = _import("langchain_ollama", "ollama")
    embeddings = module.OllamaEmbeddings(
        model=spec.model, base_url=spec.base_url or settings.ollama_base_url, **spec.options
    )
    return _embedder(model_id, spec, settings, embeddings, "ollama", batch_queries=False)


def huggingface_embeddings(model_id: str, spec: EmbeddingModelSpec, settings: Settings) -> Embedder:
    from src.models.local import HuggingFaceEmbedder

    return HuggingFaceEmbedder(model_id, spec, settings)


def register_builtin_providers(registries: Registries) -> None:
    for name, chat_factory in {
        "openai": openai_chat,
        "azure_openai": azure_openai_chat,
        "anthropic": anthropic_chat,
        "google": google_chat,
        "ollama": ollama_chat,
    }.items():
        registries.chat_providers.register(name, chat_factory)
    for name, embedding_factory in {
        "openai": openai_embeddings,
        "azure_openai": azure_openai_embeddings,
        "google": google_embeddings,
        "ollama": ollama_embeddings,
        "huggingface": huggingface_embeddings,
    }.items():
        registries.embedding_providers.register(name, embedding_factory)


__all__ = ["KNOWN_DIMENSIONS", "register_builtin_providers"]
