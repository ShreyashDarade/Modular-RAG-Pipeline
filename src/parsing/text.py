from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from src.core.errors import ParseError
from src.parsing.language import detect_language
from src.ports.parsing import ParsedUnit, TextBlock

SECTION_CHARS = 6000


def read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError(
            f"{path.name}: not valid UTF-8 ({exc.reason} at byte {exc.start}); convert it first"
        ) from exc


def split_sections(text: str, limit: int = SECTION_CHARS) -> Iterator[str]:
    """Group paragraphs into sections of roughly ``limit`` characters."""
    current: list[str] = []
    size = 0
    for paragraph in text.split("\n\n"):
        if current and size + len(paragraph) > limit:
            yield "\n\n".join(current)
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        yield "\n\n".join(current)


class TextParser:
    name = "text"
    extensions = frozenset({".txt", ".text", ".md", ".markdown", ".rst"})

    def __init__(self, settings: object | None = None) -> None:
        pass

    def iter_units(self, path: Path) -> Iterator[ParsedUnit]:
        for number, section in enumerate(split_sections(read_utf8(path)), start=1):
            if section.strip():
                yield ParsedUnit(unit=number, texts=[TextBlock(section, detect_language(section))])
