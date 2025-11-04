from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


from langchain_core.documents import Document

from src.core.logger import logger
from src.services.embedding import EmbeddingService, KeywordExtractor, TextChunker
from src.services.elastic import ElasticIndexer
from src.services.ocr import OCRService
from src.utils.file_utils import DataFileManager

from src.utils.language import SUPPORTED_LANGS, detect_language
from src.utils.pdf_parser import PdfExtractionResult, extract_from_pdf


@dataclass
class IngestionSummary:
    source: Path
    text_chunks: int
    table_chunks: int
    image_chunks: int
    skipped_reason: str | None = None
    reindexed: bool = True


class IngestionPipeline:
    def __init__(self) -> None:
        self.data_manager = DataFileManager()
        self.chunker = TextChunker()
        self.keyword_extractor = KeywordExtractor()
        self.embedding = EmbeddingService()
        probe_vector = self.embedding.embed_query("dimension probe text")
        self.indexer = ElasticIndexer(len(probe_vector))
        self.ocr = OCRService()

    def _normalize_language_hint(self, language: str | None) -> str | None:
        if not language:
            return None
        normalized = language.strip().lower()
        if normalized in {"", "auto", "none"}:
            return None
        if normalized in SUPPORTED_LANGS:
            return normalized
        logger.warning("Unsupported image language hint '%s'; defaulting to auto-detect.", language)
        return None

    def ingest_upload(self, upload_file, force: bool = False, image_language: str | None = None) -> IngestionSummary:
        stored_path = self.data_manager.store_upload(upload_file)
        language_hint = self._normalize_language_hint(image_language)
        return self.ingest_path(stored_path, force=force, image_language=language_hint)

    def ingest_path(self, path: Path, force: bool = False, image_language: str | None = None) -> IngestionSummary:
        language_hint = self._normalize_language_hint(image_language)
        existing_docs = self.indexer.source_document_count(str(path))
        needs_reindex = self.data_manager.file_needs_reindex(path)
        if not force and not needs_reindex and existing_docs > 0:
            logger.info("No changes detected for %s; skipping reindex.", path)
            return IngestionSummary(
                source=path,
                text_chunks=0,
                table_chunks=0,
                image_chunks=0,
                skipped_reason="no_changes_detected",
                reindexed=False,
            )

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            summary = self._ingest_pdf(path, language_hint=language_hint)
        elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            summary = self._ingest_image(path, language_hint=language_hint)
        else:
            logger.warning("Unsupported file extension for %s", path)
            summary = IngestionSummary(
                source=path,
                text_chunks=0,
                table_chunks=0,
                image_chunks=0,
                skipped_reason="unsupported_extension",
                reindexed=False,
            )
        if summary.reindexed:
            self.data_manager.mark_indexed(path)
        return summary

    def _prepare_documents(self, documents: List[Document], metadata_overrides: dict | None = None):
        if not documents:
            return []
        texts = [doc.page_content for doc in documents]
        keywords_all = self.keyword_extractor.extract(texts)
        embeddings = self.embedding.embed_documents(documents)
        prepared = []
        for doc, keywords, vector in zip(documents, keywords_all, embeddings):
            metadata = {**doc.metadata}
            if metadata_overrides:
                metadata.update(metadata_overrides)
            prepared.append(
                {
                    "content": doc.page_content,
                    "content_vector": vector,
                    "keywords": keywords,
                    "metadata": metadata,
                    "language": metadata.get("language", detect_language(doc.page_content)),
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "created_at": int(time.time() * 1000),
                }
            )
        return prepared

    def _ingest_pdf(self, path: Path, language_hint: str | None = None) -> IngestionSummary:
        logger.info("Parsing PDF %s", path)
        extraction: PdfExtractionResult = extract_from_pdf(path)

        text_chunks_total = 0
        table_chunks_total = 0
        image_chunks_total = 0

        text_docs: List[Document] = []
        for block in extraction.text_blocks:
            chunk_docs = self.chunker.split(
                block.content,
                metadata={
                    "source": str(path),
                    "page": block.page,
                    "language": block.language,
                    "type": "pdf_text",
                },
            )
            text_docs.extend(chunk_docs)
        if text_docs:
            prepared = self._prepare_documents(text_docs)
            self.indexer.index_text_documents(prepared)
            text_chunks_total += len(prepared)

        table_docs: List[Document] = []
        for table in extraction.tables:
            table_markdown = table.dataframe.to_markdown(index=False)
            chunk_docs = self.chunker.split(
                table_markdown,
                metadata={
                    "source": str(path),
                    "page": table.page,
                    "language": table.language,
                    "type": "pdf_table",
                },
            )
            table_docs.extend(chunk_docs)
        if table_docs:
            prepared_tables = self._prepare_documents(table_docs)
            self.indexer.index_table_documents(prepared_tables)
            table_chunks_total += len(prepared_tables)

        image_docs: List[Document] = []
        for pdf_img in extraction.images:
            ocr_result = self.ocr.read(pdf_img.image, language_hint=language_hint)
            if not ocr_result.text.strip():
                continue
            chunk_docs = self.chunker.split(
                ocr_result.text,
                metadata={
                    "source": str(path),
                    "page": pdf_img.page,
                    "language": ocr_result.language,
                    "type": "pdf_image",
                    "image_label": pdf_img.label,
                },
            )
            image_docs.extend(chunk_docs)
        if image_docs:
            prepared_images = self._prepare_documents(image_docs)
            self.indexer.index_image_documents(prepared_images)
            image_chunks_total += len(prepared_images)

        return IngestionSummary(
            source=path,
            text_chunks=text_chunks_total,
            table_chunks=table_chunks_total,
            image_chunks=image_chunks_total,
        )

    def _ingest_image(self, path: Path, language_hint: str | None = None) -> IngestionSummary:
        logger.info("Processing image %s", path)
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            logger.error("Failed to load image %s", path)
            return IngestionSummary(
                source=path,
                text_chunks=0,
                table_chunks=0,
                image_chunks=0,
                skipped_reason="image_load_error",
                reindexed=False,
            )
        # Legacy preprocessing removed in favor of OCRService pipeline
        ocr_result = self.ocr.read(image, language_hint=language_hint)
        if not ocr_result.text.strip():
            logger.warning("No text detected in image %s", path)
            return IngestionSummary(
                source=path,
                text_chunks=0,
                table_chunks=0,
                image_chunks=0,
                skipped_reason="no_text_detected",
                reindexed=False,
            )
        docs = self.chunker.split(
            ocr_result.text,
            metadata={
                "source": str(path),
                "language": ocr_result.language,
                "type": "image",
            },
        )
        prepared = self._prepare_documents(docs)
        self.indexer.index_image_documents(prepared)
        return IngestionSummary(
            source=path,
            text_chunks=0,
            table_chunks=0,
            image_chunks=len(prepared),
        )


__all__ = ["IngestionPipeline", "IngestionSummary"]
