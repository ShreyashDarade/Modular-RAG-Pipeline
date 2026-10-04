"""Candidate handling around the reranker, as pure functions: merge the per-variant result lists,
then pick the final result. Kept separate from retrieval (Elasticsearch) and from reranking
(a model call) so each rule is testable on its own.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from src.core.types import CONTENT_KINDS, RetrievedDocument

MAX_CHUNKS_PER_PAGE = 2


def _identity(doc: RetrievedDocument) -> tuple[str, object, object, str]:
    meta = doc.metadata
    return doc.collection, meta.get("source"), meta.get("page"), doc.content[:100]


def merge_variants(per_variant: Sequence[Sequence[RetrievedDocument]]) -> list[RetrievedDocument]:
    """One entry per distinct chunk, keeping its best fused score, best first."""
    best: dict[tuple[str, object, object, str], RetrievedDocument] = {}
    for docs in per_variant:
        for doc in docs:
            key = _identity(doc)
            current = best.get(key)
            if current is None or doc.score > current.score:
                best[key] = doc
    return sorted(best.values(), key=lambda d: d.score, reverse=True)


def select_result(
    ranked: Sequence[RetrievedDocument], limit: int, *, balance_kinds: bool = True
) -> list[RetrievedDocument]:
    """The best ``limit`` of ``ranked`` (best first, by final score).

    * With ``balance_kinds``, the best chunk of every content kind comes first, so an answer that
      lives in a table or a scanned image is not drowned out by denser text.
    * No page contributes more than ``MAX_CHUNKS_PER_PAGE`` chunks while other pages can fill the
      slots; only if slots are still free are the extra chunks of crowded pages used.
    """
    ordered = sorted(ranked, key=lambda d: d.final_score, reverse=True)
    chosen: list[RetrievedDocument] = []
    taken: set[int] = set()
    per_page: Counter[tuple[str, object, object]] = Counter()

    def take(doc: RetrievedDocument) -> None:
        chosen.append(doc)
        taken.add(id(doc))
        per_page[doc.collection, doc.metadata.get("source"), doc.metadata.get("page")] += 1

    if balance_kinds:
        for kind in CONTENT_KINDS:
            first = next((d for d in ordered if d.kind == kind), None)
            if first is not None and len(chosen) < limit:
                take(first)
    for doc in ordered:
        if len(chosen) >= limit:
            break
        page = (doc.collection, doc.metadata.get("source"), doc.metadata.get("page"))
        if id(doc) not in taken and per_page[page] < MAX_CHUNKS_PER_PAGE:
            take(doc)
    for doc in ordered:  # backfill from crowded pages rather than leave slots empty
        if len(chosen) >= limit:
            break
        if id(doc) not in taken:
            take(doc)
    return sorted(chosen, key=lambda d: d.final_score, reverse=True)
