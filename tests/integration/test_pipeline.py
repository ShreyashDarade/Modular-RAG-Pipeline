"""End to end against a live Elasticsearch: parse -> chunk -> embed -> index -> search -> answer."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.core.container import Container
from src.core.errors import InvalidRequestError, NotFoundError, UnsupportedTypeError
from src.core.types import JobSpec

from tests.helpers import make_docx, make_pdf

pytestmark = pytest.mark.integration

FINANCE = [
    "Quarterly revenue grew twelve percent driven by cloud subscriptions across all regions.\n"
    "Operating margin improved to eighteen percent after cost reductions in the second half.",
    "Customer retention remained strong with churn below two percent for enterprise accounts.\n"
    "Headcount increased modestly while infrastructure spending was held flat year over year.",
]
TABLE = [["Region", "Q1", "Q2"], ["North", "10", "12"], ["South", "8", "9"]]


async def ingest(container: Container, path: Path, collection: str = "alpha", **kw):
    return await container.ingestion.ingest(JobSpec(collection=collection, path=str(path), **kw))


async def doc_count(container: Container, collection: str, source: Path | None = None) -> int:
    spec = container.config.collection(collection)
    total = 0
    for index in spec.index_names().values():
        await container.elastic.client.indices.refresh(index=index)
        query = {"term": {"source": str(source)}} if source else {"match_all": {}}
        total += (await container.elastic.client.count(index=index, query=query))["count"]
    return total


async def test_ingest_pdf_and_retrieve_with_cross_references(container: Container, tmp_path: Path):
    pdf = make_pdf(tmp_path / "finance.pdf", FINANCE, table=TABLE)
    summary = await ingest(container, pdf)

    assert summary.parser == "pdf" and summary.total_pages == 2
    assert summary.text_chunks >= 2 and summary.table_chunks >= 1
    assert summary.cross_references > 0 and summary.skipped_reason is None

    scope = container.retrieval.scope(["alpha"])
    result = await container.retrieval.retrieve("quarterly revenue growth", scope)
    assert result.expanded_queries[0] == "quarterly revenue growth" and len(result.expanded_queries) == 4
    top = result.documents[0]
    assert "revenue" in top.content and top.metadata["page"] == 1
    assert top.metadata["type"] == "pdf_text" and top.collection == "alpha"
    assert any(d.kind == "table" for d in result.documents)  # one of each kind surfaces

    # a direct hit is never outranked by a cross-reference neighbour
    direct = [d for d in result.documents if not d.metadata.get("is_cross_reference")]
    assert direct and max(d.final_score for d in direct) >= max(d.final_score for d in result.documents)


async def test_reingest_is_idempotent_and_detects_changes(container: Container, tmp_path: Path):
    pdf = make_pdf(tmp_path / "finance.pdf", FINANCE)
    first = await ingest(container, pdf)
    count = await doc_count(container, "alpha", pdf)

    unchanged = await ingest(container, pdf)
    assert unchanged.skipped_reason == "no_changes_detected" and unchanged.reindexed is False
    assert await doc_count(container, "alpha", pdf) == count

    forced = await ingest(container, pdf, force=True)  # same content, new run
    assert forced.reindexed and forced.document_id != first.document_id
    assert await doc_count(container, "alpha", pdf) == count, "forced re-index must not duplicate chunks"

    make_pdf(pdf, ["Completely different content about gardening and soil temperature.\n" * 3])
    changed = await ingest(container, pdf)
    assert changed.reindexed
    scope = container.retrieval.scope(["alpha"])
    result = await container.retrieval.retrieve("gardening soil", scope)
    assert "gardening" in result.documents[0].content
    old = await container.retrieval.retrieve("quarterly revenue growth", scope)
    assert all("revenue" not in d.content for d in old.documents), (
        "stale chunks of the old version must be gone"
    )


async def test_multiple_file_types(container: Container, tmp_path: Path):
    (tmp_path / "notes.txt").write_text(
        "The migration runbook says to drain traffic before failing over the primary database.\n\n" * 3
    )
    (tmp_path / "data.csv").write_text("sku,units\n" + "\n".join(f"widget-{i},{i * 3}" for i in range(60)))
    (tmp_path / "page.html").write_text(
        "<html><body><h1>Handbook</h1><p>Vacation requests need manager approval two weeks ahead.</p></body></html>"
    )
    make_docx(
        tmp_path / "policy.docx", {"Leave": "Employees receive twenty days of paid annual leave per year."}
    )

    parsers = {}
    for name in ("notes.txt", "data.csv", "page.html", "policy.docx"):
        summary = await ingest(container, tmp_path / name)
        parsers[name] = summary.parser
        assert summary.text_chunks + summary.table_chunks > 0, name
    assert parsers == {"notes.txt": "text", "data.csv": "csv", "page.html": "html", "policy.docx": "docx"}

    scope = container.retrieval.scope(["alpha"])
    for query, expected in [
        ("failing over the primary database", "notes.txt"),
        ("annual leave days", "policy.docx"),
        ("vacation manager approval", "page.html"),
    ]:
        result = await container.retrieval.retrieve(query, scope)
        assert result.documents[0].metadata["source"].endswith(expected), (
            query,
            result.documents[0].metadata["source"],
        )


async def test_collections_are_isolated_and_can_use_different_embedding_models(
    container: Container, tmp_path: Path
):
    pdf = make_pdf(tmp_path / "finance.pdf", FINANCE)
    await ingest(container, pdf, collection="alpha")
    (tmp_path / "hr.txt").write_text(
        "Employees accrue vacation days monthly and may carry five days over to the next year.\n" * 3
    )
    await ingest(container, tmp_path / "hr.txt", collection="beta")

    only_alpha = await container.retrieval.retrieve(
        "vacation days carry over", container.retrieval.scope(["alpha"])
    )
    assert all(d.collection == "alpha" for d in only_alpha.documents)
    assert all("vacation" not in d.content for d in only_alpha.documents)

    both = await container.retrieval.retrieve(
        "vacation days carry over", container.retrieval.scope(["alpha", "beta"])
    )
    # first-stage scores are per index and not comparable across collections, so without a reranker the
    # collections are interleaved by rank: both are represented, but the best match is not guaranteed first
    assert {d.collection for d in both.documents} == {"alpha", "beta"}
    assert any(d.collection == "beta" and "vacation" in d.content for d in both.documents)

    # beta indexes text+table only and accepts only pdf/text/csv
    with pytest.raises(InvalidRequestError):
        container.retrieval.scope(["beta"], ["image"])
    (tmp_path / "x.html").write_text("<p>hello there my friend, this is long enough to be a chunk</p>")
    with pytest.raises(UnsupportedTypeError):
        await ingest(container, tmp_path / "x.html", collection="beta")
    with pytest.raises(NotFoundError):
        container.retrieval.scope(["nope"])


async def test_a_reranker_merges_collections_by_relevance(make_settings, rag_config, run_id, tmp_path: Path):
    """The point of reranking across collections: their first-stage scores cannot be compared, a reranker's can."""
    from src.core.specs import RerankerSpec

    config = rag_config.model_copy(
        update={
            "reranker": "overlap",
            "reranker_models": {"overlap": RerankerSpec(provider="overlap", model="words")},
        }
    )
    built = await Container.build(make_settings(), role="api", with_ingestion=True, config=config)
    await built.start()
    try:
        await ingest(built, make_pdf(tmp_path / "finance.pdf", FINANCE), collection="alpha")
        (tmp_path / "hr.txt").write_text(
            "Employees accrue vacation days monthly and may carry five days over to the next year.\n" * 3
        )
        await ingest(built, tmp_path / "hr.txt", collection="beta")
        result = await built.retrieval.retrieve(
            "vacation days carry over", built.retrieval.scope(["alpha", "beta"])
        )
        assert result.documents[0].collection == "beta" and "vacation" in result.documents[0].content
        assert result.documents[0].final_score > result.documents[-1].final_score
    finally:
        names = list(await built.elastic.client.indices.get(index=f"t{run_id}-*", ignore_unavailable=True))
        if names:
            await built.elastic.client.indices.delete(index=names, ignore_unavailable=True)
        await built.close()


async def test_selective_kinds_and_sources(container: Container, tmp_path: Path):
    pdf = make_pdf(tmp_path / "finance.pdf", FINANCE, table=TABLE)
    text_only = await ingest(container, pdf, kinds=("text",))
    assert text_only.table_chunks == 0 and text_only.text_chunks > 0

    unchanged = await ingest(container, pdf, kinds=("text",))
    assert unchanged.reindexed is False
    widened = await ingest(container, pdf, kinds=("text", "table"))  # different kinds => re-run
    assert widened.reindexed and widened.table_chunks > 0

    tables = await container.retrieval.retrieve(
        "region north south", container.retrieval.scope(["alpha"], ["table"])
    )
    assert tables.documents and all(d.kind == "table" for d in tables.documents)

    other = tmp_path / "other.txt"
    other.write_text("Totally separate document about quarterly revenue of a different company.\n" * 3)
    await ingest(container, other)
    scoped = await container.retrieval.retrieve(
        "quarterly revenue", container.retrieval.scope(["alpha"], sources=[str(other)])
    )
    assert scoped.documents and all(d.metadata["source"] == str(other) for d in scoped.documents)


async def test_delete_removes_chunks_ledger_and_cached_results(container: Container, tmp_path: Path):
    pdf = make_pdf(tmp_path / "finance.pdf", FINANCE)
    await ingest(container, pdf)
    scope = container.retrieval.scope(["alpha"])
    assert (await container.retrieval.retrieve("quarterly revenue", scope)).documents  # now cached

    records, total = await container.documents.list("alpha", limit=10, offset=0)
    assert (
        total == 1 and records[0].source == str(pdf) and records[0].kinds == ["table", "text"]
    )  # OCR is off in this fixture, so image is excluded

    deleted = await container.documents.delete("alpha", str(pdf))
    assert deleted > 0
    assert (await container.documents.list("alpha", limit=10, offset=0))[1] == 0
    assert (await container.retrieval.retrieve("quarterly revenue", scope)).documents == [], (
        "cache must be invalidated"
    )
    again = await ingest(container, pdf)  # ledger gone => full re-ingest, not "unchanged"
    assert again.reindexed


async def test_ask_and_multi_model_chat(container: Container, tmp_path: Path):
    await ingest(container, make_pdf(tmp_path / "finance.pdf", FINANCE))
    scope = container.retrieval.scope(["alpha"])

    fast = await container.answers.ask("what happened to revenue", scope)
    smart = await container.answers.ask("what happened to revenue", scope, "smart")
    assert fast.model == "fast" and smart.model == "smart"
    assert "finance.pdf" in fast.answer and fast.documents

    first = await container.chat.chat("what happened to revenue", scope, model="smart")
    assert first.model == "smart" and first.conversation_id
    second = await container.chat.chat("and the margin?", scope, conversation_id=first.conversation_id)
    assert second.standalone_query.startswith("STANDALONE") and second.model == "fast"
    history = await container.chat.history(first.conversation_id)
    assert [m.role for m in history] == ["user", "assistant", "user", "assistant"]
    with pytest.raises(NotFoundError):
        await container.chat.chat("hello", scope, conversation_id="does-not-exist")

    events = [e async for e in container.chat.stream("revenue growth", scope)]
    assert type(events[0]).__name__ == "ChatStarted" and type(events[-1]).__name__ == "ChatFinished"
    assert (
        "".join(e.text for e in events if type(e).__name__ == "ChatDelta").strip()
        == events[-1].answer.strip()
    )


async def test_queries_are_batched_into_one_search_round_trip(
    container: Container, tmp_path: Path, monkeypatch
):
    await ingest(container, make_pdf(tmp_path / "finance.pdf", FINANCE, table=TABLE))
    searches = []
    original = container.elastic.client.msearch

    async def counting(*args, **kwargs):
        searches.append(len(kwargs["searches"]) // 2)
        return await original(*args, **kwargs)

    monkeypatch.setattr(container.elastic.client, "msearch", counting)
    embedder = container.models.embedder("hash64")._inner  # noqa: SLF001
    embedder.calls.clear()
    await container.retrieval.retrieve("never asked before", container.retrieval.scope(["alpha"]))
    # 4 query variants x 3 kinds x (lexical + vector) = 24 searches in exactly one request
    assert searches == [24]
    assert embedder.calls == [("queries", 4)], "all variants embedded in one call"
