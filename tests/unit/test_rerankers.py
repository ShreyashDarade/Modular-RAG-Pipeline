"""Rerankers: the API adapter against a mock server, the local cross-encoder on a tiny model."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr
from src.core.config import Settings
from src.core.errors import ConfigError, ModelError, UnknownComponentError
from src.core.registry import Registries
from src.core.specs import RagConfig, RerankerSpec
from src.core.types import RetrievedDocument
from src.models.rerankers import ApiReranker, CrossEncoderReranker, register_reranker_providers
from src.retrieval.rerank import IdentityReranker, register_builtin_rerankers

from tests.unit.tiny_models import save_tiny_bert

TEXTS = ["revenue growth in the cloud", "paris is the capital of france", "bread", "tower in paris " * 40]


def docs() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            content=t, metadata={}, score=0.01 * (i + 1), collection="c", kind="text", index="i"
        )
        for i, t in enumerate(TEXTS)
    ]


def settings(**kw) -> Settings:
    return Settings(_env_file=None, plugins=[], model_max_retries=2, **kw)


# --- the hosted API adapter ----------------------------------------------------------------
def api(handler, *, retries: int = 2, provider: str = "cohere", **spec) -> ApiReranker:
    return ApiReranker(
        provider,
        "r",
        RerankerSpec(provider=provider, model="rerank-x", max_retries=retries, **spec),
        settings(),
        SecretStr("key"),
        transport=httpx.MockTransport(handler),
    )


def scores_for(request: httpx.Request, scores: list[float]) -> httpx.Response:
    body = json.loads(request.content)
    n = len(body["documents"])
    return httpx.Response(
        200, json={"results": [{"index": i, "relevance_score": scores[i]} for i in reversed(range(n))]}
    )


async def test_api_reranker_sends_the_documented_request_and_maps_scores_by_index(monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return scores_for(request, [0.1, 0.9, 0.3, 0.2])

    out = await api(handler).rerank(docs(), "capital of france")
    assert [round(d.final_score, 1) for d in out] == [0.1, 0.9, 0.3, 0.2], (
        "results are matched by index, not order"
    )
    (request,) = seen
    assert request.url == "https://api.cohere.com/v2/rerank"
    assert request.headers["authorization"] == "Bearer key"
    body = json.loads(request.content)
    assert body["model"] == "rerank-x" and body["query"] == "capital of france" and body["top_n"] == 4
    assert body["documents"][0] == TEXTS[0]


async def test_api_reranker_truncates_long_passages_and_honours_base_url():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return scores_for(request, [0.5] * 4)

    reranker = api(handler, provider="jina", max_chars=10, base_url="http://rerank.internal:8080/")
    await reranker.rerank(docs(), "q")
    assert str(seen[0].url) == "http://rerank.internal:8080/v1/rerank"
    assert all(len(text) <= 10 for text in json.loads(seen[0].content)["documents"])


async def test_api_reranker_retries_transient_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr("src.models.rerankers.asyncio.sleep", _instant)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429 if calls == 1 else 503, text="slow down")
        return scores_for(request, [0.4, 0.3, 0.2, 0.1])

    out = await api(handler, retries=2).rerank(docs(), "q")
    assert calls == 3 and out[0].final_score == pytest.approx(0.4)


async def _instant(_seconds: float) -> None:
    return None


async def test_api_reranker_gives_up_with_a_typed_error_and_never_passes_candidates_through(monkeypatch):
    monkeypatch.setattr("src.models.rerankers.asyncio.sleep", _instant)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="down")

    candidates = docs()
    with pytest.raises(ModelError, match="HTTP 503"):
        await api(handler, retries=2).rerank(candidates, "q")
    assert calls == 3
    assert all(d.rerank_score is None for d in candidates), "no half-reranked, no silent fall-through"


async def test_api_reranker_does_not_retry_client_errors():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(ModelError, match="HTTP 401"):
        await api(handler, retries=3).rerank(docs(), "q")
    assert calls == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [{"index": 0, "relevance_score": 0.5}]},  # too few
        {"results": [{"index": 9, "relevance_score": 0.5}] * 4},  # index out of range
        {"results": [{"index": 0, "relevance_score": "high"}] * 4},  # not a number
        {"nope": []},
        [],
    ],
)
async def test_api_reranker_rejects_malformed_responses(payload):
    candidates = docs()
    with pytest.raises(ModelError):
        await api(lambda request: httpx.Response(200, json=payload)).rerank(candidates, "q")
    assert all(d.rerank_score is None for d in candidates)


def test_api_reranker_needs_a_key_and_a_model():
    spec = RerankerSpec(provider="cohere", model="m")
    with pytest.raises(ConfigError, match="COHERE_API_KEY"):
        ApiReranker("cohere", "r", spec, settings(), None)
    with pytest.raises(ConfigError, match="`model` is required"):
        ApiReranker("jina", "r", RerankerSpec(provider="jina"), settings(), SecretStr("k"))
    with pytest.raises(ConfigError, match="unknown option"):
        ApiReranker(
            "jina",
            "r",
            RerankerSpec(provider="jina", model="m", options={"typo": 1}),
            settings(),
            SecretStr("k"),
        )


# --- the local cross-encoder, on a tiny random model -------------------------------------------
@pytest.fixture(scope="module")
def tiny_ce(tmp_path_factory):
    pytest.importorskip("transformers")
    return save_tiny_bert(tmp_path_factory.mktemp("ce"), labels=1)


def ce(path, **spec) -> CrossEncoderReranker:
    options = spec.pop("options", {})
    return CrossEncoderReranker(
        "ce",
        RerankerSpec(provider="cross-encoder", model=str(path), options={"device": "cpu", **options}, **spec),
        settings(),
    )


async def test_cross_encoder_scores_are_probabilities_and_do_not_depend_on_batching(tiny_ce):
    candidates = docs()
    big = await ce(tiny_ce, batch_size=16).rerank(candidates, "capital of france")
    one_by_one = await ce(tiny_ce, batch_size=1).rerank(docs(), "capital of france")
    assert all(0.0 < d.final_score < 1.0 for d in big)
    assert len({round(d.final_score, 6) for d in big}) > 1, "different passages must score differently"
    assert [d.final_score for d in big] == pytest.approx([d.final_score for d in one_by_one], abs=1e-5)


async def test_cross_encoder_activation_none_returns_the_raw_logit(tiny_ce):
    import math

    probability = (await ce(tiny_ce).rerank(docs(), "q"))[0].final_score
    logit = (await ce(tiny_ce, options={"activation": "none"}).rerank(docs(), "q"))[0].final_score
    assert 1 / (1 + math.exp(-logit)) == pytest.approx(probability, abs=1e-5)


async def test_cross_encoder_handles_empty_input_and_passages_beyond_the_model_length(tiny_ce):
    reranker = ce(tiny_ce, options={"max_length": 32})
    assert await reranker.rerank([], "q") == []
    long = [
        RetrievedDocument(
            content="paris " * 5000, metadata={}, score=0.1, collection="c", kind="text", index="i"
        )
    ]
    assert len(await reranker.rerank(long, "q")) == 1


async def test_cross_encoder_max_length_defaults_to_the_models_limit_and_rejects_more(tiny_ce):
    ok = ce(tiny_ce)
    await ok.start()
    assert ok._max_length == 64, "the tiny model has 64 positions; the default must not exceed them"  # noqa: SLF001
    with pytest.raises(ConfigError, match="exceeds the model's limit of 64"):
        await ce(tiny_ce, options={"max_length": 512}).start()


async def test_cross_encoder_refuses_models_that_do_not_output_one_score(tmp_path):
    pytest.importorskip("transformers")
    two = save_tiny_bert(tmp_path / "two", labels=2)
    with pytest.raises(ConfigError, match="2 output labels"):
        await ce(two).start()


async def test_cross_encoder_configuration_errors_are_typed_and_early(tmp_path):
    pytest.importorskip("transformers")
    with pytest.raises(ConfigError, match="`model` is required"):
        CrossEncoderReranker("ce", RerankerSpec(provider="cross-encoder"), settings())
    with pytest.raises(ConfigError, match="unknown option"):
        CrossEncoderReranker(
            "ce", RerankerSpec(provider="cross-encoder", model="m", options={"devise": "cpu"}), settings()
        )
    with pytest.raises(ConfigError, match="activation"):
        CrossEncoderReranker(
            "ce",
            RerankerSpec(provider="cross-encoder", model="m", options={"activation": "relu"}),
            settings(),
        )
    with pytest.raises(ConfigError, match="cannot load"):
        await ce(tmp_path / "missing").start()
    with pytest.raises(ConfigError, match="CUDA is not available"):
        await ce(tmp_path, options={"device": "cuda"}).start()


# --- registration ----------------------------------------------------------------------------
def test_providers_are_registered_by_name_and_unknown_names_list_the_valid_ones():
    registries = Registries()
    register_builtin_rerankers(registries)
    register_reranker_providers(registries)
    assert registries.rerankers.names() == ["cohere", "cross-encoder", "heuristic", "identity", "jina"]
    with pytest.raises(UnknownComponentError, match="Registered: cohere, cross-encoder"):
        registries.rerankers.create("bm25-ish", "x", RerankerSpec(provider="bm25-ish"), settings())
    assert isinstance(
        registries.rerankers.create("identity", "identity", RerankerSpec(provider="identity"), settings()),
        IdentityReranker,
    )


def test_config_resolves_a_named_reranker_or_a_builtin():
    base = {
        "default_chat_model": "c",
        "default_collection": "m",
        "chat_models": {"c": {"provider": "fake", "model": "c"}},
        "embedding_models": {"e": {"provider": "fake", "model": "e", "dimensions": 8}},
        "collections": {"m": {"embedding_model": "e"}},
    }
    builtin = RagConfig.model_validate({**base, "reranker": "identity"})
    assert builtin.reranker_spec() == RerankerSpec(provider="identity")
    named = RagConfig.model_validate(
        {
            **base,
            "reranker": "precise",
            "reranker_models": {
                "precise": {"provider": "cross-encoder", "model": "org/model", "batch_size": 8}
            },
        }
    )
    assert named.reranker_spec().model == "org/model" and named.reranker_spec().batch_size == 8


async def test_cross_encoder_survives_a_query_that_alone_fills_the_window(tiny_ce):
    """Regression: truncation="only_second" raised when the query left no room for the passage, so every
    request with a long query failed."""
    out = await ce(tiny_ce).rerank(docs(), "paris " * 500)
    assert len(out) == 4 and all(0.0 < d.final_score < 1.0 for d in out)


def test_pathological_options_are_refused_at_construction_not_hung_on():
    pytest.importorskip("transformers")
    for bad in (0, -1, 1.5, "2", True):
        with pytest.raises(ConfigError, match="`concurrency` must be an integer >= 1"):
            CrossEncoderReranker(
                "ce",
                RerankerSpec(provider="cross-encoder", model="m", options={"concurrency": bad}),
                settings(),
            )
    with pytest.raises(ValueError):
        RerankerSpec(provider="cohere", model="m", max_retries=-1)
