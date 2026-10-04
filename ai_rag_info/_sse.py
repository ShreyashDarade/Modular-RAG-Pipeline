"""A small, spec-following server-sent-events parser (internal).

Fed one decoded line at a time (as ``httpx`` yields them), it returns a complete event when the blank line
that terminates it arrives. Comment lines (``:``), ``id`` and ``retry`` fields are ignored.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SSEEvent:
    event: str
    data: str


class LineSplitter:
    """Bytes -> lines, split only on ``\\n``, ``\\r\\n`` and ``\\r`` as the SSE specification says.

    ``httpx.Response.aiter_lines`` follows ``str.splitlines`` instead, which also breaks on U+2028, U+2029,
    U+0085 and a few control characters - all of which can appear raw inside a JSON string (PDF text is full
    of them) and would cut an event in half. Multi-byte characters split across chunks are reassembled.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._buffer = ""

    def feed(self, data: bytes) -> list[str]:
        buffer = self._buffer + self._decoder.decode(data)
        lines: list[str] = []
        start = i = 0
        n = len(buffer)
        while i < n:
            char = buffer[i]
            if char == "\n":
                lines.append(buffer[start:i])
                i = start = i + 1
            elif char == "\r":
                if i + 1 >= n:  # a CRLF may be split across chunks: wait for the next one
                    break
                lines.append(buffer[start:i])
                i = start = i + (2 if buffer[i + 1] == "\n" else 1)
            else:
                i += 1
        self._buffer = buffer[start:]
        return lines

    def flush(self) -> list[str]:
        """What is left when the stream ends (a final line without a terminator)."""
        rest = (self._buffer + self._decoder.decode(b"", final=True)).rstrip("\r")
        self._buffer = ""
        return [rest] if rest else []


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
