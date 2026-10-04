"""The Hugging Face embedder on a tiny random model: loading, pooling, prefixes, errors."""

from __future__ import annotations

import math

import pytest
from src.core.config import Settings
from src.core.errors import ConfigError
from src.core.specs import EmbeddingModelSpec

pytest.importorskip("transformers")
from src.models.local import HuggingFaceEmbedder  # noqa: E402

from tests.unit.tiny_models import save_tiny_bert  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return save_tiny_bert(tmp_path_factory.mktemp("emb"), labels=None, hidden=16)


def embedder(path, *, dimensions: int | None = 16, **options) -> HuggingFaceEmbedder:
    spec = EmbeddingModelSpec(
        provider="huggingface", model=str(path), dimensions=dimensions, options={"device": "cpu", **options}
    )
    return HuggingFaceEmbedder("e", spec, Settings(_env_file=None, plugins=[]))


TEXTS = ["revenue growth cloud", "paris capital france", "bread", "tower " * 80]


def test_the_tiny_tokenizer_really_distinguishes_words(tiny):
    """Guards the other tests against passing vacuously on a tokenizer that maps everything to [UNK]."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny)
    assert tokenizer("paris revenue")["input_ids"] != tokenizer("bread revenue")["input_ids"]
    assert 1 not in tokenizer("paris revenue")["input_ids"]  # [UNK]


async def test_vectors_have_the_declared_size_and_unit_length_by_default(tiny):
    vectors = await embedder(tiny).embed_documents(TEXTS)
    assert len(vectors) == 4 and all(len(v) == 16 for v in vectors)
    assert all(math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0, abs=1e-5) for v in vectors)


async def test_results_do_not_depend_on_batch_size_or_input_order(tiny):
    small = await embedder(tiny).embed_documents(TEXTS)
    spec = EmbeddingModelSpec(
        provider="huggingface", model=str(tiny), dimensions=16, batch_size=1, options={"device": "cpu"}
    )
    one_by_one = await HuggingFaceEmbedder("e", spec, Settings(_env_file=None, plugins=[])).embed_documents(
        TEXTS
    )
    reversed_order = await embedder(tiny).embed_documents(list(reversed(TEXTS)))
    for a, b in zip(small, one_by_one, strict=True):
        assert a == pytest.approx(b, abs=1e-5)
    for a, b in zip(small, reversed(reversed_order), strict=True):
        assert a == pytest.approx(b, abs=1e-5), "sorting by length must put every vector back in its place"


async def test_pooling_normalisation_and_prefixes_change_the_vectors_as_configured(tiny):
    mean = (await embedder(tiny).embed_documents(["revenue growth"]))[0]
    cls = (await embedder(tiny, pooling="cls").embed_documents(["revenue growth"]))[0]
    raw = (await embedder(tiny, normalize=False).embed_documents(["revenue growth"]))[0]
    assert mean != pytest.approx(cls, abs=1e-4)
    assert math.sqrt(sum(x * x for x in raw)) != pytest.approx(1.0, abs=1e-3)

    prefixed = embedder(tiny, query_prefix="paris ", document_prefix="bread ")
    assert (await prefixed.embed_queries(["revenue"]))[0] != pytest.approx(
        (await prefixed.embed_documents(["revenue"]))[0], abs=1e-4
    ), "queries and documents get their own prefix"
    assert (await embedder(tiny).embed_queries(["revenue"]))[0] == pytest.approx(
        (await embedder(tiny).embed_documents(["revenue"]))[0], abs=1e-6
    )


async def test_empty_input_is_an_empty_result(tiny):
    assert await embedder(tiny).embed_documents([]) == []


def test_configuration_errors_are_typed_and_happen_at_construction(tiny, tmp_path):
    with pytest.raises(ConfigError, match="set `dimensions` explicitly"):
        embedder(tiny, dimensions=None)
    with pytest.raises(ConfigError, match="outputs 16 dimensions, `dimensions` says 32"):
        embedder(tiny, dimensions=32)
    with pytest.raises(ConfigError, match="pooling must be"):
        embedder(tiny, pooling="max")
    with pytest.raises(ConfigError, match="unknown option"):
        embedder(tiny, polling="cls")
    with pytest.raises(ConfigError, match="cannot load"):
        embedder(tmp_path / "nope")
    with pytest.raises(ConfigError, match="CUDA is not available"):
        embedder(tiny, device="cuda")


def test_a_zero_concurrency_would_hang_so_it_is_refused(tiny):
    with pytest.raises(ConfigError, match="`concurrency` must be an integer >= 1"):
        embedder(tiny, concurrency=0)
