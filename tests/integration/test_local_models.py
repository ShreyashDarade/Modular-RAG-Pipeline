"""Real weights: the local cross-encoder and embedder must actually rank sensibly. Skipped unless the
models are already in the Hugging Face cache (CI without network, laptops that never downloaded them)."""

from __future__ import annotations

import pytest
from src.core.config import Settings
from src.core.specs import EmbeddingModelSpec, RerankerSpec
from src.core.types import RetrievedDocument

pytest.importorskip("transformers")
from src.models.local import HuggingFaceEmbedder  # noqa: E402
from src.models.rerankers import CrossEncoderReranker  # noqa: E402

pytestmark = pytest.mark.integration

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _cached(repo: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


PASSAGES = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889.",
    "Bananas are rich in potassium and are usually eaten raw.",
    "Paris is the capital and largest city of France.",
    "Sourdough bread is made with a fermented starter instead of commercial yeast.",
]


@pytest.mark.skipif(not _cached(CE_MODEL), reason=f"{CE_MODEL} is not in the Hugging Face cache")
async def test_the_cross_encoder_ranks_the_answering_passage_first():
    reranker = CrossEncoderReranker(
        "ce",
        RerankerSpec(provider="cross-encoder", model=CE_MODEL, options={"device": "cpu"}),
        Settings(_env_file=None),
    )
    docs = [
        RetrievedDocument(content=t, metadata={}, score=0.01, collection="c", kind="text", index="i")
        for t in PASSAGES
    ]
    ranked = sorted(
        await reranker.rerank(docs, "What is the capital of France?"), key=lambda d: -d.final_score
    )
    assert ranked[0].content == PASSAGES[2]
    assert ranked[0].final_score > 0.9 and ranked[-1].final_score < 0.1


@pytest.mark.skipif(not _cached(EMBED_MODEL), reason=f"{EMBED_MODEL} is not in the Hugging Face cache")
async def test_the_embedder_puts_related_text_closer():
    spec = EmbeddingModelSpec(
        provider="huggingface",
        model=EMBED_MODEL,
        dimensions=384,
        options={
            "device": "cpu",
            "pooling": "cls",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
        },
    )
    embedder = HuggingFaceEmbedder("e", spec, Settings(_env_file=None))
    query = (await embedder.embed_queries(["What is the capital of France?"]))[0]
    sims = [
        sum(a * b for a, b in zip(query, v, strict=True)) for v in await embedder.embed_documents(PASSAGES)
    ]
    assert max(range(4), key=sims.__getitem__) == 2
