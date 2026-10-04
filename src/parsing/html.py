from __future__ import annotations

from collections.abc import Iterator
from html.parser import HTMLParser
from pathlib import Path

from src.parsing.language import detect_language
from src.parsing.markdown import table_block
from src.parsing.text import read_utf8
from src.ports.parsing import ParsedUnit, TableBlock, TextBlock

_SKIP = {"script", "style", "head", "noscript", "template", "svg", "title"}
_BREAK = {
    "p",
    "div",
    "br",
    "li",
    "ul",
    "ol",
    "section",
    "article",
    "header",
    "footer",
    "pre",
    "blockquote",
    "tr",
}
_SECTION_HEADINGS = {"h1", "h2"}


class _Collector(HTMLParser):
    """Collects text and tables, starting a new section at every <h1>/<h2>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[tuple[list[str], list[list[list[str]]]]] = [([], [])]
        self._skip_depth = 0
        self._table_depth = 0
        self._rows: list[list[str]] = []
        self._cell: list[str] | None = None

    @property
    def _text(self) -> list[str]:
        return self.sections[-1][0]

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP:
            self._skip_depth += 1
        elif tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._rows = []
        elif tag == "tr" and self._table_depth == 1:
            self._rows.append([])
        elif tag in ("td", "th") and self._table_depth == 1:
            self._cell = []
        elif tag in _SECTION_HEADINGS and self._table_depth == 0 and "".join(self._text).strip():
            self.sections.append(([], []))
        if tag in _BREAK or tag in _SECTION_HEADINGS or tag in ("h3", "h4", "h5", "h6"):
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in ("td", "th") and self._table_depth == 1 and self._cell is not None:
            if not self._rows:
                self._rows.append([])
            self._rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "table" and self._table_depth > 0:
            self._table_depth -= 1
            if self._table_depth == 0 and self._rows:
                self.sections[-1][1].append([r for r in self._rows if r])
                self._rows = []
        if tag in _BREAK or tag in _SECTION_HEADINGS or tag in ("h3", "h4", "h5", "h6"):
            self._emit("\n")

    def _emit(self, text: str) -> None:
        if self._skip_depth:
            return
        if self._cell is not None:
            self._cell.append(text)
        elif self._table_depth == 0:
            self._text.append(text)

    def handle_data(self, data: str) -> None:
        self._emit(data)


class HtmlParser:
    name = "html"
    extensions = frozenset({".html", ".htm", ".xhtml"})

    def __init__(self, settings: object | None = None) -> None:
        pass

    def iter_units(self, path: Path) -> Iterator[ParsedUnit]:
        collector = _Collector()
        collector.feed(read_utf8(path))
        collector.close()
        number = 0
        for text_parts, tables in collector.sections:
            text = "\n".join(line.strip() for line in "".join(text_parts).splitlines() if line.strip())
            blocks: list[TableBlock] = []
            for rows in tables:
                header, *body = rows
                blocks.append(
                    table_block(header, body, detect_language(" ".join(c for r in rows for c in r)))
                )
            if not text and not blocks:
                continue
            number += 1
            unit = ParsedUnit(unit=number, tables=blocks)
            if text:
                unit.texts.append(TextBlock(text, detect_language(text)))
            yield unit
