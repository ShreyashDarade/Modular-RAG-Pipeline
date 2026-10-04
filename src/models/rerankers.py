"""Model-based rerankers, registered by provider name:

* ``cross-encoder`` - a local sequence-classification model (ms-marco MiniLM, bge-reranker, ...)
  on Hugging Face ``transformers`` (extra ``local``). It reads query and passage *together*, which
  is what makes it far more accurate than the bi-encoder similarity used for first-stage retrieval.
* ``cohere`` / ``jina`` - hosted rerank APIs (Cohere v2 ``/rerank`` and the Jina-compatible form of
  it; point ``base_url`` at any server that speaks the same request/response).

A failure - a timeout, a 5xx after the retries, a malformed response, a model that cannot be
loaded - raises a typed error. Nothing here ever returns the candidates un-reranked.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import httpx

from src.core.errors import ConfigError, ModelError
from src.core.logger import logger
from src.core.registry import Registries
from src.core.types import RetrievedDocument
from src.models.local import (
    length_sorted_batches,
    positive_int,
    require_stack,
    resolve_device,
    resolve_max_length,
    take_options,
)
from src.runtime.concurrency import Bulkhead
from src.runtime.metrics import UPSTREAM_ERRORS, UPSTREAM_LATENCY

if TYPE_CHECKING:
    from pydantic import SecretStr

    from src.core.config import Settings
    from src.core.specs import RerankerSpec


class CrossEncoderReranker:
    def __init__(self, name: str, spec: RerankerSpec, settings: Settings) -> None:
        owner = f"cross-encoder reranker '{name}'"
        if not spec.model:
            raise ConfigError(f"{owner}: `model` is required (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)")
        self._owner = owner
        self._spec = spec
        self._opts = take_options(
            owner,
            spec.options,
            {
                "device": "auto",
                "max_length": None,  # default: 512, or the model's limit if smaller
                "activation": "sigmoid",  # sigmoid: scores in (0, 1) | none: raw logits
                "revision": None,
                "concurrency": 1,
            },
        )
        if self._opts["activation"] not in ("sigmoid", "none"):
            raise ConfigError(f"{owner}: activation must be 'sigmoid' or 'none'")
        require_stack()  # fail at construction if torch/transformers are missing
        self._model: Any = None
        self._max_length = 512
        self._tokenizer: Any = None
        self._device = "cpu"
        self._slots = asyncio.Semaphore(positive_int(owner, "concurrency", self._opts["concurrency"]))
        self._load_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._load_lock:
            if self._model is None:
                await asyncio.to_thread(self._load)

    def _load(self) -> None:
        torch, transformers = require_stack()
        self._device = resolve_device(torch, self._opts["device"], self._owner)
        revision = self._opts["revision"]
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self._spec.model, revision=revision)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self._spec.model, revision=revision
            )
        except OSError as exc:
            raise ConfigError(f"{self._owner}: cannot load '{self._spec.model}': {exc}") from exc
        if int(model.config.num_labels) != 1:
            raise ConfigError(
                f"{self._owner}: '{self._spec.model}' has {model.config.num_labels} output labels; "
                "only single-score relevance models are supported"
            )
        self._max_length = resolve_max_length(self._owner, self._opts["max_length"], tokenizer, model.config)
        self._tokenizer, self._model = tokenizer, model.eval().to(self._device)
        logger.info("loaded reranker %s on %s", self._spec.model, self._device)

    def _score(self, query: str, passages: Sequence[str]) -> list[float]:
        torch = require_stack()[0]
        scores = [0.0] * len(passages)
        with torch.inference_mode():
            for indices in length_sorted_batches(passages, self._spec.batch_size):
                batch = self._tokenizer(
                    [query] * len(indices),
                    [passages[i] for i in indices],
                    padding=True,
                    truncation="longest_first",  # never raises, even for a query that alone fills the window
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**batch).logits[:, 0]
                if self._opts["activation"] == "sigmoid":
                    logits = torch.sigmoid(logits)
                for position, value in zip(indices, logits.float().cpu().tolist(), strict=True):
                    scores[position] = value
        return scores

    async def rerank(self, documents: Sequence[RetrievedDocument], query: str) -> list[RetrievedDocument]:
        if not documents:
            return []
        await self.start()  # a no-op once loaded; makes direct use (tests, scripts) safe
        passages = [d.content[: self._spec.max_chars] for d in documents]
        started = time.perf_counter()
        try:
            async with self._slots:
                scores = await asyncio.to_thread(self._score, query, passages)
        except Exception as exc:
            UPSTREAM_ERRORS.labels("cross-encoder", "rerank").inc()
            logger.error("reranking failed", extra={"model": self._spec.model}, exc_info=exc)
            raise ModelError(f"{self._owner}: rerank failed: {type(exc).__name__}: {exc}") from exc
        UPSTREAM_LATENCY.labels("cross-encoder", "rerank").observe(time.perf_counter() - started)
        for doc, score in zip(documents, scores, strict=True):
            doc.rerank_score = score
        return list(documents)

    async def close(self) -> None:
        self._model = self._tokenizer = None


_API_DEFAULTS = {
    "cohere": ("https://api.cohere.com", "/v2/rerank", "COHERE_API_KEY"),
    "jina": ("https://api.jina.ai", "/v1/rerank", "JINA_API_KEY"),
}
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class ApiReranker:
    """Cohere-style ``POST /rerank``: ``{model, query, documents[], top_n}`` ->
    ``{results: [{index, relevance_score}]}``."""

    def __init__(
        self,
        provider: str,
        name: str,
        spec: RerankerSpec,
        settings: Settings,
        key: SecretStr | None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        base, self._path, env_name = _API_DEFAULTS[provider]
        owner = f"{provider} reranker '{name}'"
        if not spec.model:
            raise ConfigError(f"{owner}: `model` is required")
        if key is None or not key.get_secret_value():
            raise ConfigError(f"{provider}: {env_name} is required")
        take_options(owner, spec.options, {})
        self._provider, self._owner, self._spec = provider, owner, spec
        self._retries = settings.model_max_retries if spec.max_retries is None else spec.max_retries
        self._bulkhead = Bulkhead(settings.model_max_concurrency, name=provider)
        self._client = httpx.AsyncClient(
            base_url=(spec.base_url or base).rstrip("/"),
            headers={"Authorization": f"Bearer {key.get_secret_value()}", "Content-Type": "application/json"},
            timeout=spec.timeout_seconds or settings.model_timeout_seconds,
            transport=transport,
        )

    async def start(self) -> None:
        return None

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        delay = 0.5
        for attempt in range(self._retries + 1):
            try:
                async with self._bulkhead:
                    response = await self._client.post(self._path, json=body)
            except httpx.TransportError as exc:  # timeouts, resets, DNS
                if attempt == self._retries:
                    raise ModelError(f"{self._owner}: request failed: {type(exc).__name__}: {exc}") from exc
            else:
                if response.status_code < 400:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise ModelError(f"{self._owner}: response is not JSON") from exc
                    if not isinstance(payload, dict):
                        raise ModelError(f"{self._owner}: unexpected response shape")
                    return payload
                if response.status_code not in _RETRY_STATUS or attempt == self._retries:
                    raise ModelError(f"{self._owner}: HTTP {response.status_code}: {response.text[:300]}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8.0)
        raise AssertionError("unreachable")  # the loop returns or raises on its last attempt

    async def rerank(self, documents: Sequence[RetrievedDocument], query: str) -> list[RetrievedDocument]:
        if not documents:
            return []
        started = time.perf_counter()
        try:
            payload = await self._post(
                {
                    "model": self._spec.model,
                    "query": query,
                    "documents": [d.content[: self._spec.max_chars] for d in documents],
                    "top_n": len(documents),
                }
            )
        except ModelError:
            UPSTREAM_ERRORS.labels(self._provider, "rerank").inc()
            raise
        UPSTREAM_LATENCY.labels(self._provider, "rerank").observe(time.perf_counter() - started)
        scores: dict[int, float] = {}
        for item in payload.get("results") or []:
            index, score = (
                item.get("index") if isinstance(item, dict) else None,
                (item.get("relevance_score") if isinstance(item, dict) else None),
            )
            if (
                not isinstance(index, int)
                or not 0 <= index < len(documents)
                or not isinstance(score, int | float)
            ):
                raise ModelError(f"{self._owner}: malformed result entry: {item!r}")
            scores[index] = float(score)
        if len(scores) != len(documents):
            raise ModelError(f"{self._owner}: scored {len(scores)} of {len(documents)} documents")
        for index, doc in enumerate(documents):
            doc.rerank_score = scores[index]
        return list(documents)

    async def close(self) -> None:
        await self._client.aclose()


def register_reranker_providers(registries: Registries) -> None:
    def cross_encoder(name: str, spec: RerankerSpec, settings: Settings) -> CrossEncoderReranker:
        return CrossEncoderReranker(name, spec, settings)

    def cohere(name: str, spec: RerankerSpec, settings: Settings) -> ApiReranker:
        return ApiReranker("cohere", name, spec, settings, settings.cohere_api_key)

    def jina(name: str, spec: RerankerSpec, settings: Settings) -> ApiReranker:
        return ApiReranker("jina", name, spec, settings, settings.jina_api_key)

    registries.rerankers.register("cross-encoder", cross_encoder)
    registries.rerankers.register("cohere", cohere)
    registries.rerankers.register("jina", jina)
