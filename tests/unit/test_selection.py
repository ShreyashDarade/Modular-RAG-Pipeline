"""Merging per-variant candidates and the final cut."""

from __future__ import annotations

from src.core.types import RetrievedDocument
from src.retrieval.selection import merge_variants, select_result


def doc(
    content: str, score: float, *, page: int = 1, source: str = "/a.pdf", kind: str = "text", rerank=None
) -> RetrievedDocument:
    return RetrievedDocument(
        content=content,
        metadata={"source": source, "page": page},
        score=score,
        collection="main",
        kind=kind,  # type: ignore[arg-type]
        index="i",
        rerank_score=rerank,
    )


def test_merge_keeps_one_entry_per_chunk_with_its_best_score_best_first():
    merged = merge_variants(
        [
            [doc("alpha", 0.5), doc("beta", 0.2)],
            [doc("alpha", 0.9), doc("gamma", 0.4)],
        ]
    )
    assert [(d.content, d.score) for d in merged] == [("alpha", 0.9), ("gamma", 0.4), ("beta", 0.2)]


def test_a_zero_or_negative_rerank_score_is_a_score_not_a_missing_one():
    """Regression-proofing for model rerankers: logits can be exactly 0 or negative."""
    assert doc("x", 0.5, rerank=0.0).final_score == 0.0
    assert doc("x", 0.5, rerank=-3.0).final_score == -3.0
    assert doc("x", 0.5).final_score == 0.5


def test_select_orders_by_final_score_not_fused_score():
    low_fused_high_rerank = doc("a", 0.1, rerank=0.99)
    high_fused_low_rerank = doc("b", 0.9, rerank=0.01)
    chosen = select_result([high_fused_low_rerank, low_fused_high_rerank], 2)
    assert [d.content for d in chosen] == ["a", "b"]


def test_select_caps_chunks_per_page_then_backfills():
    ranked = [doc(f"p1-{i}", 1.0 - i / 100, page=1) for i in range(8)] + [
        doc(f"p2-{i}", 0.5 - i / 100, page=2) for i in range(3)
    ]
    pages = [d.metadata["page"] for d in select_result(ranked, 4)]
    assert len(pages) == 4 and pages.count(1) == 2 and pages.count(2) == 2


def test_select_backfills_when_only_one_page_matches():
    ranked = [doc(f"c{i}", 1.0 - i / 100) for i in range(5)]
    assert len(select_result(ranked, 4)) == 4, "slots are filled from crowded pages rather than left empty"


def test_best_chunk_of_each_kind_comes_first_when_balancing():
    ranked = [doc(f"t{i}", 1.0 - i / 100, page=i + 1) for i in range(6)]
    ranked += [doc("table", 0.01, kind="table", page=9), doc("image", 0.005, kind="image", page=10)]
    chosen = select_result(ranked, 4)
    assert {d.kind for d in chosen} == {"text", "table", "image"}
    unbalanced = select_result(ranked, 4, balance_kinds=False)
    assert {d.kind for d in unbalanced} == {"text"}


def test_select_never_returns_more_than_the_limit_or_duplicates():
    ranked = [doc(f"c{i}", 1.0 - i / 100, page=i) for i in range(10)]
    chosen = select_result(ranked, 3)
    assert len(chosen) == 3 and len({id(d) for d in chosen}) == 3
    assert select_result([], 5) == []
