from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.errors import ParseError, ProviderUnavailableError
from src.parsing.language import detect_language
from src.ports.parsing import ImageRef, ParsedUnit, TableBlock, TextBlock

if TYPE_CHECKING:
    from src.core.config import Settings


class PdfParser:
    """PyMuPDF. One unit per page: text, detected tables (as markdown) and embedded images.

    Streams page by page and closes the file when done. An image that appears on many pages
    (a logo, a letterhead) is returned once - whether it is one shared object or many identical
    copies - and images smaller than ``ocr_min_image_side`` are
    skipped - both save a large amount of OCR time on typical documents.
    """

    name = "pdf"
    extensions = frozenset({".pdf"})

    def __init__(self, settings: Settings) -> None:
        self._extract_tables = settings.pdf_extract_tables
        self._min_side = settings.ocr_min_image_side

    def iter_units(self, path: Path) -> Iterator[ParsedUnit]:
        try:
            import pymupdf
        except ImportError as exc:
            raise ProviderUnavailableError(
                "PDF parsing needs PyMuPDF: pip install 'ai-rag-info[worker]'"
            ) from exc
        try:
            document: Any = pymupdf.open(path)  # PyMuPDF ships no type information
        except Exception as exc:  # pymupdf raises its own FileDataError/EmptyFileError
            raise ParseError(f"{path.name}: cannot open PDF: {exc}") from exc
        with document:
            if document.needs_pass:
                raise ParseError(f"{path.name}: PDF is password protected")
            seen_xrefs: set[int] = set()
            seen_content: set[str] = set()
            for number, page in enumerate(document, start=1):
                unit = ParsedUnit(unit=number)
                text = page.get_text("text")
                if text.strip():
                    unit.texts.append(TextBlock(text=text, language=detect_language(text)))
                if self._extract_tables:
                    unit.tables.extend(self._tables(page, path, number))
                for index, image in enumerate(page.get_images(full=True), start=1):
                    xref = image[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    info = document.extract_image(xref)
                    if not info or min(info["width"], info["height"]) < self._min_side:
                        continue
                    # producers often embed one logo under several object numbers: dedupe by content too
                    digest = hashlib.sha256(info["image"]).hexdigest()
                    if digest in seen_content:
                        continue
                    seen_content.add(digest)
                    unit.images.append(ImageRef(label=f"pdf-page-{number}-img-{index}", data=info["image"]))
                yield unit

    def _tables(self, page: Any, path: Path, number: int) -> list[TableBlock]:
        try:
            found = page.find_tables().tables
            frames = [table.to_pandas() for table in found]
        except Exception as exc:
            raise ParseError(
                f"{path.name}: table detection failed on page {number}: {exc} "
                "(set PDF_EXTRACT_TABLES=false to skip table extraction)"
            ) from exc
        blocks = []
        for frame in frames:
            if frame.empty:
                continue
            columns = [str(c) for c in frame.columns]
            language = detect_language(" ".join(map(str, frame.to_numpy().ravel())))
            blocks.append(
                TableBlock(
                    markdown=frame.to_markdown(index=False),
                    summary=f"Table with {len(frame)} rows and columns: {', '.join(columns[:5])}",
                    columns=tuple(columns[:20]),
                    row_count=len(frame),
                    language=language,
                )
            )
        return blocks
