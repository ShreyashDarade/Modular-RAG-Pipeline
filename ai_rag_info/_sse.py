"""A small, spec-following server-sent-events parser (internal).

Fed one decoded line at a time (as ``httpx`` yields them), it returns a complete event when the blank line
that terminates it arrives. Comment lines (``:``), ``id`` and ``retry`` fields are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SSEEvent:
    event: str
    data: str


class SSEParser:
    def __init__(self) -> None:
        self._event = ""
        self._data: list[str] = []

    def feed(self, line: str) -> SSEEvent | None:
        line = line.rstrip("\r\n")
        if line == "":
            if not self._data and not self._event:
                return None
            event = SSEEvent(self._event or "message", "\n".join(self._data))
            self._event, self._data = "", []
            return event
        if line.startswith(":"):
            return None
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if name == "event":
            self._event = value
        elif name == "data":
            self._data.append(value)
        return None

    def finish(self) -> SSEEvent | None:
        """A stream that ends without the final blank line still delivers what it had."""
        return self.feed("")
