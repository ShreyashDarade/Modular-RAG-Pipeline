"""RetrievalPipeline with a model-style reranker: one call per request, on merged candidates, against
the user's own query; its order wins; its failure is a failure."""

from __future__ import annotations

import pytest
from src.core.config import Settings
from src.core.errors import ModelError
from src.core.types import RetrievalScope, RetrievedDocument
from src.retrieval.pipeline import RetrievalPipeline
from src.runtime.cache import CachedCall, CorpusVersion, MemoryCache

SCOPE = RetrievalScope(("c",))


def doc(name: str, score: float) -> RetrievedDocument:
    return RetrievedDocument(
        content=name,
        metadata={"source": name, "page": 1},
        score=score,
        collection="c",
        kind="text",
        index="i",
    )


class Expander:
    async def expand(self, query: str) -> list[str]:
        return [query, f"{query} explained", f"about {query}", f"{query} details"]


class Retriever:
    """Each variant returns overlapping candidates; the best fused scores are d0 > d1 > ... ."""

    def resolve_scope(self, *a):
        return SCOPE

    async def retrieve_many(self, queries, scope):
        return [[doc(f"d{i}", 1.0 - i / 100 - v / 1000) for i in range(6)] for v in range(len(queries))]


class SpyReranker:
    """Prefers the document whose name sorts last, whatever the fused score says."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.fail = fail

    async def start(self) -> None: ...

    async def rerank(self, documents, query):
        self.calls.append((query, [d.content for d in documents]))
        if self.fail:
            raise ModelError("rerank backend is down")
        for d in documents:
            d.rerank_score = float(d.content[1:])
        return list(documents)

    async def close(self) -> None: ...


def pipeline(reranker, **settings) -> RetrievalPipeline:
    cache = MemoryCache(100)
    return RetrievalPipeline(
        expander=Expander(),
        retriever=Retriever(),  # type: ignore[arg-type]
        reranker=reranker,
        cache=CachedCall(cache, "t"),
        corpus=CorpusVersion(cache),
        settings=Settings(_env_file=None, plugins=[], **settings),
    )


async def test_reranks_once_over_the_merged_candidates_against_the_original_query():
    spy = SpyReranker()
    result = await pipeline(spy, retriever_top_k=3, rerank_candidates=5).retrieve("revenue", SCOPE)
    assert len(spy.calls) == 1, "four query variants, one reranking pass"
    query, candidates = spy.calls[0]
    assert query == "revenue", "scored against what the user asked, not a rewrite"
    assert candidates == ["d0", "d1", "d2", "d3", "d4"], (
        "the best fused candidates, deduplicated across variants"
    )
    assert [d.content for d in result.documents] == ["d4", "d3", "d2"], "the reranker's order wins"


async def test_the_candidate_pool_is_bounded_by_the_setting_not_by_how_much_was_retrieved():
    spy = SpyReranker()
    await pipeline(spy, retriever_top_k=2, rerank_candidates=2).retrieve("q", SCOPE)
    assert spy.calls[0][1] == ["d0", "d1"]


async def test_a_failing_reranker_fails_the_request_and_caches_nothing():
    broken = SpyReranker(fail=True)
    p = pipeline(broken)
    with pytest.raises(ModelError, match="rerank backend is down"):
        await p.retrieve("q", SCOPE)
    broken.fail = False
    assert (await p.retrieve("q", SCOPE)).documents, "the failure was not cached; the next request retries"
    assert len(broken.calls) == 2


async def test_cached_results_do_not_rerank_again():
    spy = SpyReranker()
    p = pipeline(spy)
    await p.retrieve("q", SCOPE)
    await p.retrieve("q", SCOPE)
    assert len(spy.calls) == 1
