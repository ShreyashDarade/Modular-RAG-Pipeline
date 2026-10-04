from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.errors import ConfigError, UnsupportedTypeError
from src.core.registry import Registries
from src.ports.parsing import Parser

if TYPE_CHECKING:
    from src.core.config import Settings


class ParserSet:
    """Routes a file to the parser that declared its extension. Parser classes are cheap to
    construct (heavy libraries load on first parse), so building the full set costs nothing."""

    def __init__(
        self, registries: Registries, settings: Settings, names: Sequence[str] | None = None
    ) -> None:
        self._parsers: dict[str, Parser] = {}
        self._by_extension: dict[str, Parser] = {}
        for name in names if names is not None else registries.parsers.names():
            parser = registries.parsers.create(name, settings)
            self._parsers[name] = parser
            for extension in parser.extensions:
                if extension in self._by_extension:
                    raise ConfigError(
                        f"extension '{extension}' is claimed by both '{self._by_extension[extension].name}' and '{name}'"
                    )
                self._by_extension[extension] = parser

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset(self._by_extension)

    def names(self) -> list[str]:
        return sorted(self._parsers)

    def for_path(self, path: Path, allowed: Sequence[str] | None = None) -> Parser:
        extension = path.suffix.lower()
        parser = self._by_extension.get(extension)
        if parser is None:
            raise UnsupportedTypeError(
                f"no parser for '{extension or path.name}'. Supported: {', '.join(sorted(self._by_extension))}"
            )
        if allowed is not None and parser.name not in allowed:
            raise UnsupportedTypeError(
                f"'{extension}' ({parser.name}) is not accepted by this collection; it accepts: {', '.join(allowed)}"
            )
        return parser
