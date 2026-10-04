"""Ingestion: parse -> chunk -> (OCR) -> cross-reference -> embed -> index, idempotently.

Design points
* Streaming: the parser yields one page/section at a time; only extracted text (not images or
  vectors) is kept for the whole document, vectors exist only for the slices in flight.
* Selective: only the requested content kinds are produced - skipping ``image`` skips OCR entirely.
* Idempotent: chunk ids are content-addressed, so a retry overwrites instead of duplicating.
  The commit order is *write new -> sweep older generations -> record in the ledger*; a crash
  at any point leaves either the previous state or one a re-run completes.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.chunking.ids import chunk_id
from src.core.errors import InvalidRequestError, NotFoundError, ParseError, UnsupportedTypeError
from src.core.logger import logger
from src.core.specs import CollectionSpec, RagConfig
from src.core.types import (
    CONTENT_KINDS,
    ChunkRecord,
    ContentKind,
    DocumentRecord,
    JobSpec,
    content_type_label,
)
from src.ingestion.hints import normalize_language_hint
from src.ingestion.ocr import LazyOcr
from src.ingestion.storage import compute_checksum
from src.models.registry import ModelRegistry
from src.parsing.registry import ParserSet
from src.ports.indexing import DocumentRegistry, IndexDoc, IndexWriter
from src.ports.parsing import Chunker, ParsedUnit, Parser
from src.runtime.concurrency import run_all

if TYPE_CHECKING:
    from src.chunking.keywords import KeywordExtractor
    from src.core.config import Settings

MAX_REFERENCES = 10
LockFactory = Callable[[str], AbstractAsyncContextManager[Any]]


@dataclass(slots=True)
class IngestionSummary:
    collection: str
    source: str
    parser: str = ""
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    document_id: str = ""
    total_pages: int = 0
    cross_references: int = 0
    skipped_reason: str | None = None
    reindexed: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(slots=True)
class _Chunk:
    kind: ContentKind
    unit: int
    content: str
    language: str
    chunk_id: str
    metadata: dict[str, Any]
    keywords: list[str] = field(default_factory=list)
    siblings: list[str] = field(default_factory=list)
    adjacent: list[str] = field(default_factory=list)
    has_table: bool = False
    has_image: bool = False


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        config: RagConfig,
        parsers: ParserSet,
        chunkers: dict[str, Chunker],
        models: ModelRegistry,
        writer: IndexWriter,
        registry: DocumentRegistry,
        ocr: LazyOcr | None,
        keywords: KeywordExtractor,
        lock: LockFactory,
        on_change: Callable[[], Awaitable[object]],
    ) -> None:
        self._s = settings
        self._config = config
        self._parsers = parsers
        self._chunkers = chunkers
        self._models = models
        self._writer = writer
        self._registry = registry
        self._ocr = ocr
        self._keywords = keywords
        self._lock = lock
        self._on_change = on_change
        self._slice = min(settings.es_bulk_chunk_size, settings.embedding_batch_size)

    # --- entry point -----------------------------------------------------------------------
    async def ingest(self, spec: JobSpec) -> IngestionSummary:
        collection = self._config.collection(spec.collection)
        path = Path(spec.path)
        if not await asyncio.to_thread(path.is_file):
            raise NotFoundError(f"file not found: {path}")
        parser = self._parsers.for_path(path, collection.parsers)
        kinds = self._effective_kinds(collection, spec, parser)
        hint = self._language_hint(spec.image_language)
        source = str(path)

        async with self._lock(f"{collection.name}:{source}"):
            checksum = await asyncio.to_thread(compute_checksum, path)
            existing = await self._registry.get(collection.name, source)
            if (
                not spec.force
                and existing is not None
                and existing.status == "complete"
                and existing.file_checksum == checksum
                and sorted(existing.kinds) == sorted(kinds)
            ):
                logger.info("no changes for %s in %s; skipping", source, collection.name)
                return IngestionSummary(
                    collection.name,
                    source,
                    parser.name,
                    skipped_reason="no_changes_detected",
                    reindexed=False,
                    document_id=existing.document_id,
                )
            return await self._run(
                collection, parser, path, source, checksum, kinds, hint, reingest=existing is not None
            )

    def _effective_kinds(
        self, collection: CollectionSpec, spec: JobSpec, parser: Parser
    ) -> tuple[ContentKind, ...]:
        requested = tuple(spec.kinds) if spec.kinds else collection.kinds
        bad = [k for k in requested if k not in collection.kinds]
        if bad:
            raise InvalidRequestError(f"collection '{collection.name}' does not index {bad}")
        if "image" in requested and self._ocr is None:
            if parser.name == "image":
                raise UnsupportedTypeError(
                    "OCR is disabled (OCR_ENABLED=false); image files cannot be ingested"
                )
            return tuple(k for k in requested if k != "image")
        return requested

    def _language_hint(self, language: str | None) -> str | None:
        return normalize_language_hint(language, self._s.supported_ocr_languages)

    # --- the run ---------------------------------------------------------------------------
    async def _run(
        self,
        collection: CollectionSpec,
        parser: Parser,
        path: Path,
        source: str,
        checksum: str,
        kinds: tuple[ContentKind, ...],
        hint: str | None,
        *,
        reingest: bool,
    ) -> IngestionSummary:
        document_id = f"doc_{path.stem}_{uuid.uuid4().hex[:8]}"
        summary = IngestionSummary(collection.name, source, parser.name, document_id=document_id)
        logger.info(
            "ingesting %s into %s (parser=%s kinds=%s)", source, collection.name, parser.name, list(kinds)
        )

        chunks, units = await self._parse(parser, collection, path, source, kinds, hint, summary)
        summary.total_pages = units
        if not chunks and summary.warnings:
            raise ParseError(f"{path.name}: nothing could be extracted: {'; '.join(summary.warnings[:3])}")
        summary.cross_references = self._cross_reference(chunks, document_id)

        await self._index(collection, chunks, source, document_id, checksum, parser.name, kinds)
        counts = Counter(c.kind for c in chunks)
        summary.text_chunks, summary.table_chunks, summary.image_chunks = (
            counts["text"],
            counts["table"],
            counts["image"],
        )
        if not chunks:
            summary.skipped_reason = "no_content"

        all_indices = list(collection.index_names().values())
        if reingest and self._s.ingest_refresh == "interval":
            # the sweep only sees what is searchable: an earlier generation written moments ago may not be yet
            await self._writer.refresh(all_indices)
        await self._writer.delete_other_generations(all_indices, source, document_id)
        if self._s.ingest_refresh == "each":
            await self._writer.refresh(all_indices)
        await self._registry.put(
            DocumentRecord(
                collection=collection.name,
                source=source,
                file_checksum=checksum,
                document_id=document_id,
                text_chunks=summary.text_chunks,
                table_chunks=summary.table_chunks,
                image_chunks=summary.image_chunks,
                total_pages=units,
                parser=parser.name,
                kinds=sorted(kinds),
            )
        )
        await self._on_change()
        logger.info(
            "ingested %s: %d text, %d table, %d image chunks over %d units",
            source,
            summary.text_chunks,
            summary.table_chunks,
            summary.image_chunks,
            units,
        )
        return summary

    # --- phase 1: parse + chunk + OCR ------------------------------------------------------
    async def _parse(
        self,
        parser: Parser,
        collection: CollectionSpec,
        path: Path,
        source: str,
        kinds: tuple[ContentKind, ...],
        hint: str | None,
        summary: IngestionSummary,
    ) -> tuple[list[_Chunk], int]:
        chunker = self._chunkers[collection.name]
        loop = asyncio.get_running_loop()
        # one dedicated thread: parser objects (PyMuPDF) must not hop between threads mid-file
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parse")
        iterator = parser.iter_units(path)
        chunks: list[_Chunk] = []
        units = 0
        try:
            while (unit := await loop.run_in_executor(executor, next, iterator, None)) is not None:
                units += 1
                chunks.extend(await self._chunk_unit(parser, chunker, unit, source, kinds, hint, summary))
        finally:
            if (close := getattr(iterator, "close", None)) is not None:
                await loop.run_in_executor(executor, close)
            executor.shutdown(wait=True)
        return chunks, units

    async def _chunk_unit(
        self,
        parser: Parser,
        chunker: Chunker,
        unit: ParsedUnit,
        source: str,
        kinds: tuple[ContentKind, ...],
        hint: str | None,
        summary: IngestionSummary,
    ) -> list[_Chunk]:
        produced: dict[ContentKind, list[_Chunk]] = defaultdict(list)

        def add(kind: ContentKind, text: str, language: str | None, extra: dict[str, Any]) -> None:
            for draft in chunker.split(text):
                index = len(produced[kind])
                metadata = {
                    "source": source,
                    "page": unit.unit,
                    "language": language or "unknown",
                    "type": content_type_label(parser.name, kind),
                    "chunk_index": draft.index,
                    "total_chunks": draft.total,
                    **extra,
                }
                produced[kind].append(
                    _Chunk(
                        kind,
                        unit.unit,
                        draft.content,
                        metadata["language"],
                        chunk_id(source, kind, unit.unit, index, draft.content),
                        metadata,
                    )
                )

        if "text" in kinds:
            for block in unit.texts:
                add("text", block.text, block.language, {})
        if "table" in kinds:
            for table in unit.tables:
                add(
                    "table",
                    table.markdown,
                    table.language,
                    {
                        "table_summary": table.summary,
                        "table_columns": list(table.columns),
                        "table_row_count": table.row_count,
                    },
                )
        if "image" in kinds and unit.images:
            assert self._ocr is not None
            ocr = self._ocr

            async def read(image):
                try:
                    return image, await ocr.read(image.data, hint)
                except ParseError as exc:
                    summary.warnings.append(f"{image.label}: {exc}")
                    return image, None

            for image, result in await run_all(read(i) for i in unit.images):
                if result is not None and result.text.strip():
                    add(
                        "image",
                        result.text,
                        result.language,
                        {"image_label": image.label, "ocr_confidence": result.confidence},
                    )

        for kind_chunks in produced.values():
            if kind_chunks:
                keywords = await asyncio.to_thread(self._keywords.extract, [c.content for c in kind_chunks])
                for chunk, words in zip(kind_chunks, keywords, strict=True):
                    chunk.keywords = words
        return [c for kind in CONTENT_KINDS for c in produced.get(kind, [])]

    # --- phase 2: cross references ---------------------------------------------------------
    def _cross_reference(self, chunks: list[_Chunk], document_id: str) -> int:
        for chunk in chunks:
            chunk.metadata["document_id"] = document_id
        if not self._s.enable_cross_references:
            return 0
        by_unit: dict[int, list[_Chunk]] = defaultdict(list)
        for chunk in chunks:
            by_unit[chunk.unit].append(chunk)
        window = self._s.page_context_window
        references = 0
        for unit, members in by_unit.items():
            adjacent = [
                c.chunk_id
                for offset in range(-window, window + 1)
                if offset != 0
                for c in by_unit.get(unit + offset, ())
            ][:MAX_REFERENCES]
            kinds_here = {c.kind for c in members}
            for chunk in members:
                chunk.siblings = [c.chunk_id for c in members if c is not chunk][:MAX_REFERENCES]
                chunk.adjacent = adjacent
                chunk.has_table = "table" in kinds_here
                chunk.has_image = "image" in kinds_here
                chunk.metadata["page_chunk_count"] = len(members)
                references += len(chunk.siblings) + len(chunk.adjacent)
        return references

    # --- phase 3: embed + index ------------------------------------------------------------
    async def _index(
        self,
        collection: CollectionSpec,
        chunks: list[_Chunk],
        source: str,
        document_id: str,
        checksum: str,
        parser_name: str,
        kinds: tuple[ContentKind, ...],
    ) -> None:
        embedder = self._models.embedder(collection.embedding_model)
        gate = asyncio.Semaphore(self._s.ingest_pipeline_depth)

        async def write_slice(kind: ContentKind, items: list[_Chunk]) -> None:
            async with gate:
                vectors = await embedder.embed_documents([c.content for c in items])
                docs = [
                    IndexDoc(
                        c.chunk_id,
                        ChunkRecord(
                            chunk_id=c.chunk_id,
                            content=c.content,
                            kind=kind,
                            content_type=c.metadata["type"],
                            vector=vector,
                            keywords=c.keywords,
                            language=c.language,
                            source=source,
                            page=c.unit,
                            document_id=document_id,
                            file_checksum=checksum,
                            metadata=c.metadata,
                            sibling_chunk_ids=c.siblings,
                            adjacent_chunk_ids=c.adjacent,
                            has_table_on_page=c.has_table,
                            has_image_on_page=c.has_image,
                        ).to_source(),
                    )
                    for c, vector in zip(items, vectors, strict=True)
                ]
                await self._writer.write(collection.index_name(kind), docs)

        work = []
        for kind in kinds:
            of_kind = [c for c in chunks if c.kind == kind]
            work += [
                write_slice(kind, of_kind[i : i + self._slice]) for i in range(0, len(of_kind), self._slice)
            ]
        await run_all(work)
