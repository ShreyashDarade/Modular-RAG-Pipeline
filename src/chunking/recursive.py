from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.registry import Registries
from src.ports.parsing import ChunkDraft

if TYPE_CHECKING:
    from src.core.specs import ChunkerSpec


class RecursiveChunker:
    """Splits on paragraph, line, sentence (including the Devanagari danda) and word boundaries,
    in that order of preference."""

    name = "recursive"

    def __init__(self, spec: ChunkerSpec) -> None:
        self._min_chunk_size = spec.min_chunk_size
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=spec.chunk_size,
            chunk_overlap=spec.chunk_overlap,
            separators=["\n\n", "\n", "। ", ". ", "? ", "! ", "; ", ", ", " "],
            length_function=len,
            # keep sentence punctuation with the sentence it ends, not at the start of the next chunk
            keep_separator="end",
        )

    def split(self, text: str) -> list[ChunkDraft]:
        if not text or len(text.strip()) < self._min_chunk_size:
            return []
        pieces = self._splitter.split_text(text)
        return [ChunkDraft(content=piece, index=i, total=len(pieces)) for i, piece in enumerate(pieces)]


def register_builtin_chunkers(registries: Registries) -> None:
    registries.chunkers.register("recursive", RecursiveChunker)
