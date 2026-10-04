from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    language: str | None = None


@dataclass(frozen=True, slots=True)
class TableBlock:
    markdown: str
    summary: str
    columns: tuple[str, ...]
    row_count: int
    language: str | None = None


@dataclass(frozen=True, slots=True)
class ImageRef:
    """An encoded image (PNG/JPEG/...) that still needs OCR."""

    label: str
    data: bytes


@dataclass(slots=True)
class ParsedUnit:
    """One page (PDF) or section (everything else). ``unit`` is 1-based and is the ``page``
    stored with each chunk; chunks of the same and neighbouring units are cross-referenced."""

    unit: int
    texts: list[TextBlock] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)


class Parser(Protocol):
    """Reads one file type. ``iter_units`` is synchronous and streams, so a 500-page scan is
    never held in memory; the ingestion service drives it from a dedicated thread."""

    name: str
    extensions: frozenset[str]

    def iter_units(self, path: Path) -> Iterator[ParsedUnit]: ...


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    language: str
    confidence: float


class OcrEngine(Protocol):
    """Blocking and CPU/GPU-bound; callers run it in a worker thread. Must be thread-safe."""

    def read(self, image: bytes, language_hint: str | None) -> OcrResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    content: str
    index: int
    total: int


class Chunker(Protocol):
    def split(self, text: str) -> list[ChunkDraft]: ...


Metadata = Mapping[str, Any]
