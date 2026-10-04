"""IngestionService against recording fakes: the order of effects is what makes it crash-safe."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from src.chunking.keywords import KeywordExtractor
from src.chunking.recursive import RecursiveChunker
from src.core.bootstrap import build_registries
from src.core.config import Settings
from src.core.errors import (
    InvalidRequestError,
    NotFoundError,
    ParseError,
    UnsupportedTypeError,
    UpstreamError,
)
from src.core.registry import Registries
from src.core.specs import RagConfig
from src.core.types import DocumentRecord, JobSpec
from src.ingestion.ocr import LazyOcr
from src.ingestion.service import IngestionService
from src.models.registry import ModelRegistry
from src.parsing.registry import ParserSet
from src.ports.parsing import OcrResult

from tests.fake_plugin import register
from tests.helpers import make_docx, make_pdf, text_png

CONFIG = RagConfig.model_validate(
    {
        "default_chat_model": "c",
        "default_collection": "main",
        "chat_models": {"c": {"provider": "fake", "model": "c"}},
        "embedding_models": {"e": {"provider": "fake", "model": "e", "dimensions": 16}},
        "collections": {
            "main": {
                "embedding_model": "e",
                "index_prefix": "m",
                "chunker": {"chunk_size": 200, "chunk_overlap": 20, "min_chunk_size": 10},
            },
            "small": {"embedding_model": "e", "index_prefix": "s", "kinds": ["text"], "parsers": ["text"]},
        },
    }
)


class Events(list):
    pass


class FakeWriter:
    def __init__(self, events: Events, *, fail_on_write: int | None = None) -> None:
        self.events, self.written, self.fail_on_write, self.writes = events, {}, fail_on_write, 0

    async def ensure_indices(self, specs): ...

    async def write(self, index, docs):
        self.writes += 1
        if self.fail_on_write == self.writes:
            raise UpstreamError("elasticsearch down")
        self.events.append(("write", index, len(docs)))
        self.written.setdefault(index, {}).update({d.id: d.source for d in docs})

    async def delete_other_generations(self, indices, source, keep_document_id):
        self.events.append(("sweep", tuple(indices), source, keep_document_id))
        return 0

    async def delete_source(self, indices, source):
        return 0

    async def refresh(self, indices):
        self.events.append(("refresh",))


class FakeRegistry:
    def __init__(self, events: Events) -> None:
        self.events, self.records = events, {}

    async def get(self, collection, source):
        return self.records.get((collection, source))

    async def put(self, record: DocumentRecord):
        self.events.append(("registry", record.document_id))
        self.records[(record.collection, record.source)] = record

    async def delete(self, collection, source): ...

    async def list(self, collection, *, limit, offset):
        return [], 0


class FakeOcr:
    def __init__(self, text="Recognised words from the picture", fail: bool = False):
        self.reads, self.text, self.fail = 0, text, fail

    def read(self, image: bytes, hint):
        self.reads += 1
        if self.fail:
            raise ParseError("image could not be decoded")
        return OcrResult(text=self.text, language=hint or "en", confidence=0.9)

    def close(self): ...


def build(tmp_path: Path, *, ocr: FakeOcr | None = None, fail_on_write=None, **settings):
    s = Settings(_env_file=None, plugins=[], data_dir=tmp_path / "data", **settings)
    registries = build_registries(s)
    register(registries)
    events = Events()
    writer, registry = FakeWriter(events, fail_on_write=fail_on_write), FakeRegistry(events)

    @asynccontextmanager
    async def lock(key):
        events.append(("lock", key))
        yield

    async def changed():
        events.append(("changed",))

    service = IngestionService(
        settings=s,
        config=CONFIG,
        parsers=ParserSet(registries, s),
        chunkers={n: RecursiveChunker(c.chunker) for n, c in CONFIG.collections.items()},
        models=ModelRegistry(CONFIG, s, registries),
        writer=writer,
        registry=registry,
        ocr=LazyOcr(lambda: ocr, 1) if ocr else None,
        keywords=KeywordExtractor(5),
        lock=lock,
        on_change=changed,
    )
    return service, writer, registry, events


BODY = [
    "Quarterly revenue grew twelve percent driven by cloud subscriptions.\nOperating margin improved after cost reductions in the second half.",
    "Customer retention remained strong with churn below two percent.\nHeadcount increased while infrastructure spending was flat.",
]


def spec(path: Path, **kw) -> JobSpec:
    return JobSpec(collection=kw.pop("collection", "main"), path=str(path), **kw)


async def test_commit_order_is_write_then_sweep_then_ledger_then_cache_invalidation(tmp_path):
    service, writer, registry, events = build(tmp_path)
    pdf = make_pdf(tmp_path / "a.pdf", BODY)
    summary = await service.ingest(spec(pdf))
    kinds = [e[0] for e in events]
    assert kinds[0] == "lock" and kinds[-4:] == ["sweep", "refresh", "registry", "changed"]
    assert "write" in kinds[1:-4]
    sweep = next(e for e in events if e[0] == "sweep")
    assert sweep[3] == summary.document_id == next(e for e in events if e[0] == "registry")[1]
    assert sweep[1] == ("m-text", "m-tables", "m-images"), (
        "stale chunks of every kind are swept, not only the indexed ones"
    )
    assert registry.records[("main", str(pdf))].file_checksum


async def test_failure_before_commit_leaves_the_previous_version_intact(tmp_path):
    # tiny slices so some writes succeed before the third one fails: a genuinely partial run
    service, writer, registry, events = build(
        tmp_path, fail_on_write=3, es_bulk_chunk_size=2, embedding_batch_size=2, ingest_pipeline_depth=1
    )
    pdf = make_pdf(tmp_path / "a.pdf", BODY * 6)
    with pytest.raises(UpstreamError):
        await service.ingest(spec(pdf))
    assert sum(1 for e in events if e[0] == "write") >= 2, (
        "a genuinely partial run: earlier slices were already written"
    )
    kinds = [e[0] for e in events]
    assert "sweep" not in kinds and "registry" not in kinds and "changed" not in kinds, (
        "nothing is swept or recorded on failure"
    )
    assert registry.records == {}, "so the next run redoes the document instead of skipping it"


async def test_retry_after_failure_produces_identical_chunk_ids(tmp_path):
    first, writer1, *_ = build(tmp_path)
    pdf = make_pdf(tmp_path / "a.pdf", BODY)
    await first.ingest(spec(pdf))
    second, writer2, *_ = build(tmp_path)
    await second.ingest(spec(pdf))
    assert {i: set(d) for i, d in writer1.written.items()} == {
        i: set(d) for i, d in writer2.written.items()
    }, "deterministic ids make retries overwrite"


async def test_unchanged_files_are_skipped_unless_forced_or_the_kinds_change(tmp_path):
    service, writer, registry, events = build(tmp_path)
    pdf = make_pdf(tmp_path / "a.pdf", BODY, table=[["h1", "h2"], ["a", "b"]])
    first = await service.ingest(spec(pdf))
    writes = writer.writes
    skipped = await service.ingest(spec(pdf))
    assert (
        skipped.skipped_reason == "no_changes_detected" and not skipped.reindexed and writer.writes == writes
    )
    assert skipped.document_id == first.document_id
    assert (await service.ingest(spec(pdf, force=True))).reindexed
    assert (await service.ingest(spec(pdf, kinds=("text",)))).reindexed, (
        "different kinds => different index content"
    )
    assert (await service.ingest(spec(pdf, kinds=("text",)))).skipped_reason == "no_changes_detected"
    make_pdf(pdf, ["Entirely different words now appear here for the checksum to change."])
    assert (await service.ingest(spec(pdf, kinds=("text",)))).reindexed


async def test_cross_references_link_the_page_and_its_neighbours(tmp_path):
    service, writer, *_ = build(tmp_path, page_context_window=1)
    pdf = make_pdf(
        tmp_path / "a.pdf",
        BODY + ["Third page discussing something else entirely, in enough words."],
        table=[["h1", "h2"], ["a", "b"]],
    )
    summary = await service.ingest(spec(pdf))
    assert summary.cross_references > 0
    text = writer.written["m-text"]
    page1 = [d for d in text.values() if d["page"] == 1]
    page3 = [d for d in text.values() if d["page"] == 3]
    assert page1 and all(d["has_table_on_page"] for d in page1), "page 1 has the table"
    assert all(not d["has_table_on_page"] for d in page3)
    ids_page2 = {d["chunk_id"] for d in text.values() if d["page"] == 2}
    assert ids_page2 & set(page1[0]["adjacent_chunk_ids"]) and not (
        ids_page2 & set(page3[0]["adjacent_chunk_ids"]) - ids_page2
    )
    assert all(d["chunk_id"] not in d["sibling_chunk_ids"] for d in text.values()), (
        "a chunk is not its own sibling"
    )
    assert all(
        len(d["sibling_chunk_ids"]) <= 10 and len(d["adjacent_chunk_ids"]) <= 10 for d in text.values()
    )


async def test_cross_references_can_be_turned_off(tmp_path):
    service, writer, *_ = build(tmp_path, enable_cross_references=False)
    summary = await service.ingest(spec(make_pdf(tmp_path / "a.pdf", BODY)))
    assert summary.cross_references == 0
    assert all(
        d["sibling_chunk_ids"] == [] and d["adjacent_chunk_ids"] == []
        for d in writer.written["m-text"].values()
    )


async def test_slices_bound_memory_and_vectors_arrive_in_every_document(tmp_path):
    service, writer, *_ = build(tmp_path, es_bulk_chunk_size=5, embedding_batch_size=5)
    sections = {
        f"Section {i}": f"Paragraph number {i} describing the topic in a sentence of reasonable length."
        for i in range(24)
    }
    summary = await service.ingest(spec(make_docx(tmp_path / "big.docx", sections)))
    writes = [e for e in service._writer.events if e[0] == "write"]  # noqa: SLF001
    assert summary.text_chunks == 24 and all(n <= 5 for _, _, n in writes) and len(writes) == 5
    assert all(len(d["content_vector"]) == 16 for d in writer.written["m-text"].values())


async def test_selective_kinds_skip_work_and_validate_against_the_collection(tmp_path):
    ocr = FakeOcr()
    service, writer, *_ = build(tmp_path, ocr=ocr)
    pdf = make_pdf(
        tmp_path / "a.pdf",
        BODY,
        table=[["h1", "h2"], ["a", "b"]],
        image_on_page=1,
        image_png=text_png(["words"]),
    )
    only_text = await service.ingest(spec(pdf, kinds=("text",)))
    assert only_text.table_chunks == only_text.image_chunks == 0 and ocr.reads == 0, (
        "unselected kinds are never produced, OCR never runs"
    )
    (tmp_path / "n.txt").write_text("Plain text long enough to be a chunk of its own, for sure.")
    with pytest.raises(InvalidRequestError, match="does not index"):
        await service.ingest(spec(tmp_path / "n.txt", collection="small", kinds=("image",)))


async def test_collection_parser_allow_list_is_enforced(tmp_path):
    service, *_ = build(tmp_path)
    (tmp_path / "a.csv").write_text("a,b\n1,2")
    with pytest.raises(UnsupportedTypeError, match="not accepted by this collection"):
        await service.ingest(spec(tmp_path / "a.csv", collection="small"))


async def test_ocr_disabled_skips_embedded_images_but_rejects_image_files(tmp_path):
    service, writer, *_ = build(tmp_path, ocr=None)
    pdf = make_pdf(tmp_path / "a.pdf", BODY, image_on_page=1, image_png=text_png(["words"]))
    summary = await service.ingest(spec(pdf))
    assert summary.image_chunks == 0 and "m-images" not in writer.written
    (tmp_path / "scan.png").write_bytes(text_png(["words"]))
    with pytest.raises(UnsupportedTypeError, match="OCR is disabled"):
        await service.ingest(spec(tmp_path / "scan.png"))


async def test_an_undecodable_embedded_image_is_reported_not_fatal(tmp_path):
    png = text_png(["words"])
    ocr = FakeOcr(fail=True)
    service, writer, *_ = build(tmp_path, ocr=ocr)
    pdf = make_pdf(tmp_path / "a.pdf", BODY, image_on_page=1, image_png=png)
    summary = await service.ingest(spec(pdf))
    assert summary.text_chunks > 0 and summary.image_chunks == 0
    assert len(summary.warnings) == 1 and "could not be decoded" in summary.warnings[0]


async def test_a_file_that_yields_nothing_but_errors_fails_instead_of_pretending(tmp_path):
    png = text_png(["words"])
    service, *_ = build(tmp_path, ocr=FakeOcr(fail=True))
    (tmp_path / "scan.png").write_bytes(png)
    with pytest.raises(ParseError, match="nothing could be extracted"):
        await service.ingest(spec(tmp_path / "scan.png"))


async def test_ocr_text_becomes_image_chunks_with_confidence_and_label(tmp_path):
    service, writer, *_ = build(tmp_path, ocr=FakeOcr("Operating margin improved to eighteen percent"))
    (tmp_path / "scan.png").write_bytes(text_png(["words"]))
    summary = await service.ingest(spec(tmp_path / "scan.png", image_language="hi"))
    [doc] = writer.written["m-images"].values()
    assert summary.image_chunks == 1 and doc["content_type"] == "image" and doc["kind"] == "image"
    assert (
        doc["metadata"]["ocr_confidence"] == 0.9
        and doc["language"] == "hi"
        and doc["metadata"]["image_label"] == "scan.png"
    )


async def test_blank_documents_are_recorded_as_empty_and_old_content_is_swept(tmp_path):
    service, writer, registry, events = build(tmp_path)
    pdf = make_pdf(tmp_path / "blank.pdf", [""])
    summary = await service.ingest(spec(pdf))
    assert summary.skipped_reason == "no_content" and summary.text_chunks == 0
    assert ("main", str(pdf)) in registry.records and any(e[0] == "sweep" for e in events), (
        "a document that became blank must lose its old chunks"
    )


async def test_bad_requests_fail_before_any_work(tmp_path):
    service, writer, *_ = build(tmp_path)
    with pytest.raises(NotFoundError, match="file not found"):
        await service.ingest(spec(tmp_path / "missing.pdf"))
    pdf = make_pdf(tmp_path / "a.pdf", BODY)
    with pytest.raises(InvalidRequestError, match="unsupported OCR language"):
        await service.ingest(spec(pdf, image_language="klingon"))
    with pytest.raises(NotFoundError, match="unknown collection"):
        await service.ingest(spec(pdf, collection="nope"))
    assert writer.writes == 0


async def test_concurrent_ingests_of_the_same_document_take_the_same_lock_key(tmp_path):
    service, writer, registry, events = build(tmp_path)
    pdf = make_pdf(tmp_path / "a.pdf", BODY)
    await asyncio.gather(service.ingest(spec(pdf)), service.ingest(spec(pdf)))
    keys = {e[1] for e in events if e[0] == "lock"}
    assert keys == {f"main:{pdf}"}


def test_registries_fixture_is_isolated():
    assert isinstance(build_registries(Settings(_env_file=None)), Registries)
