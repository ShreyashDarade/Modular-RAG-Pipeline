from __future__ import annotations

import hashlib

from src.core.types import ContentKind


def chunk_id(source: str, kind: ContentKind, page: int, index: int, content: str) -> str:
    """Deterministic, content-addressed id (also used as the Elasticsearch ``_id``).

    Re-ingesting identical content yields identical ids, so retries and re-indexing overwrite
    instead of duplicating. The id is stable across processes and platforms.
    """
    digest = hashlib.sha256(f"{source}\x1f{kind}\x1f{page}\x1f{index}\x1f{content}".encode()).hexdigest()
    return digest[:40]
