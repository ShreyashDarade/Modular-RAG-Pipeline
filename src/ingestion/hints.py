from __future__ import annotations

from collections.abc import Sequence

from src.core.errors import InvalidRequestError

_AUTO = {"", "auto", "none"}


def normalize_language_hint(language: str | None, supported: Sequence[str]) -> str | None:
    """``None`` for auto-detect, the language code if supported, otherwise an error - an
    unrecognised hint is rejected, never silently turned into auto-detect."""
    if language is None or language.strip().lower() in _AUTO:
        return None
    normalized = language.strip().lower()
    if normalized not in supported:
        raise InvalidRequestError(
            f"unsupported OCR language '{language}'; use one of {list(supported)} or 'auto'"
        )
    return normalized
