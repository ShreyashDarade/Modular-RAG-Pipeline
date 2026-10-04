"""Self-hosted models on Hugging Face ``transformers`` (extra ``local``): a text embedder here, a
cross-encoder reranker in :mod:`src.models.rerankers`. No API key, no per-call cost; runs on CPU
or a GPU. Weights come from the Hugging Face hub (or its local cache - set ``HF_HUB_OFFLINE=1`` to
forbid network access) or from a local directory.

Everything heavy is imported lazily, so a deployment that does not use these models never imports
torch.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from src.core.errors import ConfigError, ModelError, ProviderUnavailableError
from src.core.logger import logger

if TYPE_CHECKING:
    from src.core.config import Settings
    from src.core.specs import EmbeddingModelSpec


def require_stack() -> tuple[Any, Any]:
    """``(torch, transformers)``, or a :class:`ProviderUnavailableError` naming the extra."""
    try:
        return importlib.import_module("torch"), importlib.import_module("transformers")
    except ImportError as exc:
        raise ProviderUnavailableError(
            f"{exc.name} is not installed; install the 'local' extra: pip install 'ai-rag-info[local]'"
        ) from exc


def take_options(owner: str, given: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    """``given`` merged over ``defaults``; an option nobody understands is an error, not ignored."""
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise ConfigError(f"{owner}: unknown option(s) {unknown}; supported: {sorted(defaults)}")
    return {**defaults, **given}


def positive_int(owner: str, name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{owner}: option `{name}` must be an integer >= 1, got {value!r}")
    return value


def resolve_device(torch: Any, requested: str, owner: str) -> str:
    """``auto`` picks CUDA, then Apple MPS, then CPU. An explicit device that is not there is an error."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ConfigError(f"{owner}: device '{requested}' requested but CUDA is not available")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise ConfigError(f"{owner}: device 'mps' requested but it is not available")
    return requested


def resolve_max_length(owner: str, requested: int | None, tokenizer: Any, config: Any) -> int:
    """512 or the model's own limit, whichever is smaller; an explicit value beyond the limit is an
    error (it would crash on the first long input)."""
    declared = int(tokenizer.model_max_length)
    limit = declared if declared < 1_000_000 else int(config.max_position_embeddings)
    if requested is None:
        return min(512, limit)
    if requested > limit:
        raise ConfigError(f"{owner}: max_length {requested} exceeds the model's limit of {limit} tokens")
    return requested


def length_sorted_batches(texts: Sequence[str], size: int) -> list[list[int]]:
    """Indices grouped so each batch holds similarly long texts (less padding = less compute)."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    return [order[i : i + size] for i in range(0, len(order), size)]


class HuggingFaceEmbedder:
    """Encoder-only embedding model (bge, e5, MiniLM, ...). Loads its weights at construction, so a
    bad model name or a wrong ``dimensions`` stops start-up rather than the first request."""

    def __init__(self, model_id: str, spec: EmbeddingModelSpec, settings: Settings) -> None:
        owner = f"huggingface embedder '{model_id}'"
        options = take_options(
            owner,
            spec.options,
            {
                "device": "auto",
                "max_length": None,  # default: 512, or the model's limit if smaller
                "pooling": "mean",  # mean: MiniLM, e5, gte | cls: bge
                "normalize": True,
                "query_prefix": "",  # e.g. bge: "Represent this sentence for searching relevant passages: "
                "document_prefix": "",  # e.g. e5: "passage: " (queries: "query: ")
                "revision": None,
                "concurrency": 1,
            },
        )
        if options["pooling"] not in ("mean", "cls"):
            raise ConfigError(f"{owner}: pooling must be 'mean' or 'cls'")
        if spec.dimensions is None:
            raise ConfigError(f"{owner}: set `dimensions` explicitly - the pipeline never probes a model")
        torch, transformers = require_stack()
        self._torch = torch
        self.model_id = model_id
        self.dimensions = spec.dimensions
        self._opts = options
        self._batch = spec.batch_size or 32
        self._device = resolve_device(torch, options["device"], owner)
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                spec.model, revision=options["revision"]
            )
            self._model = (
                transformers.AutoModel.from_pretrained(spec.model, revision=options["revision"])
                .eval()
                .to(self._device)
            )
        except OSError as exc:
            raise ConfigError(f"{owner}: cannot load '{spec.model}': {exc}") from exc
        self._max_length = resolve_max_length(
            owner, options["max_length"], self._tokenizer, self._model.config
        )
        hidden = int(self._model.config.hidden_size)
        if hidden != spec.dimensions:
            raise ConfigError(
                f"{owner}: the model outputs {hidden} dimensions, `dimensions` says {spec.dimensions}"
            )
        self._slots = asyncio.Semaphore(positive_int(owner, "concurrency", options["concurrency"]))
        logger.info("loaded embedding model %s on %s", spec.model, self._device)

    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        torch = self._torch
        prepared = [prefix + t for t in texts]
        out: list[list[float]] = [[] for _ in prepared]
        with torch.inference_mode():
            for indices in length_sorted_batches(prepared, self._batch):
                batch = self._tokenizer(
                    [prepared[i] for i in indices],
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**batch).last_hidden_state
                if self._opts["pooling"] == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                if self._opts["normalize"]:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                for position, vector in zip(indices, pooled.cpu().tolist(), strict=True):
                    out[position] = vector
        return out

    async def _run(self, texts: Sequence[str], prefix: str, operation: str) -> list[list[float]]:
        if not texts:
            return []
        try:
            async with self._slots:
                return await asyncio.to_thread(self._encode, texts, prefix)
        except Exception as exc:
            logger.error("local embedding failed", extra={"model": self.model_id}, exc_info=exc)
            raise ModelError(f"{self.model_id}: {operation} failed: {type(exc).__name__}: {exc}") from exc

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._run(texts, self._opts["document_prefix"], "embed_documents")

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._run(texts, self._opts["query_prefix"], "embed_queries")
