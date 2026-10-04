from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from src.core.errors import InvalidRequestError
from src.core.types import CONTENT_KINDS, ContentKind


def parse_kinds(values: Iterable[str] | None) -> tuple[ContentKind, ...] | None:
    """Validate user-supplied content kinds (order kept, duplicates dropped); ``None`` stays ``None``."""
    if values is None:
        return None
    unique = tuple(dict.fromkeys(v.strip() for v in values if v.strip()))
    bad = [v for v in unique if v not in CONTENT_KINDS]
    if bad:
        raise InvalidRequestError(f"unknown content kind(s) {bad}; valid: {', '.join(CONTENT_KINDS)}")
    return cast("tuple[ContentKind, ...]", unique) or None
