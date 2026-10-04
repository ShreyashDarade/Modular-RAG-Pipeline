"""HybridRetriever against an in-memory Searcher: fusion, cross-references, diversity, scope."""

from __future__ import annotations

import pytest
from src.core.config import Settings
from src.core.errors import InvalidRequestError, NotFoundError
from src.core.registry import Registries
from src.core.specs import RagConfig
from src.core.types import RetrievedDocument
from src.models.registry import ModelRegistry
from src.retrieval.hybrid import CROSS_REFERENCE_DISCOUNT, HybridRetriever
from src.retrieval.rerank import HeuristicReranker

from tests.fake_plugin import HashEmbedder, register
from tests.unit.fakes import MemorySearcher

CONFIG = RagConfig.model_validate(
    {
        "default_chat_model": "c",
        "default_collection": "main",
        "chat_models": {"c": {"provider": "fake", "model": "c"}},
        "embedding_models": {"e": {"provider": "fake", "model": "e", "dimensions": 32}},
        "collections": {
            "main": {"embedding_model": "e", "index_prefix": "main"},
            "other": {"embedding_model": "e", "index_prefix": "other", "kinds": ["text"]},
        },
    }
)
EMBED = HashEmbedder("e", 32)


def doc(
    chunk_id: str,
    content: str,
    *,
    page: int = 1,
    source: str = "/a.pdf",
    kind: str = "text",
    siblings=(),
    adjacent=(),
) -> dict:
    return {
        "chunk_id": chunk_id,
        "content": content,
        "content_vector": EMBED._vector(content),
        "source": source,
        "page": page,
        "kind": kind,
        "content_type": f"pdf_{kind}",
        "keywords": [],
        "language": "en",
        "metadata": {"type": f"pdf_{kind}"},
        "sibling_chunk_ids": list(siblings),
        "adjacent_chunk_ids": list(adjacent),
        "has_table_on_page": False,
        "has_image_on_page": False,
    }


def retriever(searcher: MemorySearcher, **settings) -> HybridRetriever:
    s = Settings(_env_file=None, plugins=[], **settings)
    registries = Registries()
    register(registries)
    return HybridRetriever(
        searcher=searcher,
        models=ModelRegistry(CONFIG, s, registries),
        config=CONFIG,
        settings=s,
    )


async def test_hybrid_ranking_combines_lexical_and_vector_signals():
    searcher = MemorySearcher()
    searcher.add("main-text", doc("c1", "quarterly revenue grew twelve percent"))
    searcher.add("main-text", doc("c2", "office plants need watering weekly"))
    searcher.add("main-text", doc("c3", "revenue was discussed at the board meeting"))
    r = retriever(searcher)
    [docs] = await r.retrieve_many(["quarterly revenue growth"], r.resolve_scope(None))
    assert [d.metadata["chunk_id"] for d in docs][0] == "c1"
    best = docs[0].metadata["score_breakdown"]
    assert best["bm25_rrf"] > 0 and best["knn_rrf"] > 0


async def test_alpha_weights_lexical_against_vector():
    searcher = MemorySearcher()
    searcher.add("main-text", doc("c1", "revenue revenue revenue growth"))
    only_lexical = retriever(searcher, hybrid_alpha=1.0)
    [docs] = await only_lexical.retrieve_many(["revenue"], only_lexical.resolve_scope(None))
    assert docs[0].score == pytest.approx(docs[0].metadata["score_breakdown"]["bm25_rrf"])
    only_vector = retriever(searcher, hybrid_alpha=0.0)
    [docs] = await only_vector.retrieve_many(["revenue"], only_vector.resolve_scope(None))
    assert docs[0].score == pytest.approx(docs[0].metadata["score_breakdown"]["knn_rrf"])


async def test_cross_references_support_but_never_outrank_the_direct_match():
    """Regression: neighbours used to get a flat 0.35 against RRF scores of ~0.01, so they always won."""
    searcher = MemorySearcher()
    searcher.add(
        "main-text",
        doc("hit", "quarterly revenue grew twelve percent", siblings=["neighbour"], adjacent=["next-page"]),
    )
    searcher.add("main-text", doc("neighbour", "completely unrelated sentence about plants"))
    searcher.add("main-text", doc("next-page", "another unrelated sentence about weather", page=2))
    for i in range(8):  # closer to the query than the neighbours, so only the links bring those in
        searcher.add(
            "main-text", doc(f"filler{i}", f"quarterly revenue note number {i}", source="/other.pdf")
        )
    r = retriever(searcher, retriever_top_k=1, rerank_top_k=20)  # search size 3, like a real top-k index
    [docs] = await r.retrieve_many(["quarterly revenue"], r.resolve_scope(None))
    by_id = {d.metadata["chunk_id"]: d for d in docs}
    assert docs[0].metadata["chunk_id"] == "hit"
    assert by_id["neighbour"].metadata["is_cross_reference"] and by_id["neighbour"].score < by_id["hit"].score
    assert by_id["neighbour"].score == pytest.approx(by_id["hit"].score * CROSS_REFERENCE_DISCOUNT)


async def test_cross_reference_fetch_is_one_call_per_collection_and_respects_kind_scope():
    searcher = MemorySearcher()
    searcher.add("main-text", doc("hit", "alpha beta gamma", siblings=["tbl"]))
    searcher.add("main-tables", doc("tbl", "table row alpha", kind="table"))
    r = retriever(searcher)
    await r.retrieve_many(["alpha", "alpha beta"], r.resolve_scope(["main"], ["text"]))
    assert len(searcher.fetch_calls) == 1, "all variants share one fetch"
    indices, _ = searcher.fetch_calls[0]
    assert indices == ("main-text",), (
        "a text-only scope must not pull in table chunks through cross-references"
    )


async def test_one_search_round_trip_for_all_variants_and_kinds():
    searcher = MemorySearcher()
    searcher.add("main-text", doc("c1", "alpha"))
    r = retriever(searcher)
    await r.retrieve_many(["alpha", "beta", "gamma", "delta"], r.resolve_scope(None))
    assert searcher.search_calls == [4 * 3], "4 variants x 3 kinds, in a single call"


async def test_source_filter_limits_results():
    searcher = MemorySearcher()
    searcher.add("main-text", doc("a", "revenue growth", source="/a.pdf"))
    searcher.add("main-text", doc("b", "revenue growth", source="/b.pdf"))
    r = retriever(searcher)
    [docs] = await r.retrieve_many(["revenue"], r.resolve_scope(None, sources=["/b.pdf"]))
    assert {d.metadata["source"] for d in docs} == {"/b.pdf"}


def test_scope_resolution():
    r = retriever(MemorySearcher())
    assert r.resolve_scope(None).collections == ("main",)
    assert r.resolve_scope(["main", "other", "main"]).collections == ("main", "other")
    with pytest.raises(NotFoundError, match="unknown collection 'zzz'"):
        r.resolve_scope(["zzz"])
    with pytest.raises(InvalidRequestError, match="unknown content kind"):
        r.resolve_scope(None, ["video"])
    with pytest.raises(InvalidRequestError, match="does not index"):
        r.resolve_scope(["other"], ["table"])


def make_doc(**meta) -> RetrievedDocument:
    base = {"keywords": [], "page": 9}
    base.update(meta)
    return RetrievedDocument(
        content=meta.pop("content", "some text"),
        metadata=base,
        score=1.0,
        collection="main",
        kind=meta.get("kind", "text"),
        index="i",
    )


async def test_heuristic_reranker_signals():
    rr = HeuristicReranker()
    plain = (await rr.rerank([make_doc()], "revenue"))[0].rerank_score
    with_keyword = (await rr.rerank([make_doc(keywords=["revenue"])], "revenue"))[0].rerank_score
    exact = (await rr.rerank([make_doc(content="total revenue grew")], "revenue"))[0].rerank_score
    early = (await rr.rerank([make_doc(page=1)], "revenue"))[0].rerank_score
    assert with_keyword > plain and exact > plain and early > plain
    table = RetrievedDocument(
        content="x", metadata={"keywords": [], "page": 9}, score=1.0, collection="m", kind="table", index="i"
    )
    assert (await rr.rerank([table], "show me the data table"))[0].rerank_score > (
        await rr.rerank([table], "unrelated")
    )[0].rerank_score
