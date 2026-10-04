"""Shared helper: render rows as a GitHub-flavoured markdown table."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from src.ports.parsing import TableBlock


def _cell(value: Any) -> str:
    return (
        "" if value is None else str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()
    )


def markdown_table(header: Sequence[Any], rows: Iterable[Sequence[Any]]) -> str:
    head = [_cell(h) for h in header]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for row in rows:
        cells = [_cell(v) for v in row]
        cells += [""] * (len(head) - len(cells))
        lines.append("| " + " | ".join(cells[: len(head)]) + " |")
    return "\n".join(lines)


def table_block(
    header: Sequence[Any], rows: Sequence[Sequence[Any]], language: str | None, label: str = ""
) -> TableBlock:
    columns = tuple(_cell(h) for h in header)
    prefix = f"{label}: " if label else ""
    summary = f"{prefix}Table with {len(rows)} rows and columns: {', '.join(columns[:5])}"
    return TableBlock(
        markdown=markdown_table(header, rows),
        summary=summary,
        columns=columns[:20],
        row_count=len(rows),
        language=language,
    )
