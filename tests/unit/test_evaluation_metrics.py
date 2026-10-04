"""Ranking metrics checked against hand-computed values, and the statistics around them."""

from __future__ import annotations

import math

import pytest
from src.core.types import RetrievedDocument
from src.evaluation.dataset import Label
from src.evaluation.metrics import (
    judge_ranking,
    label_matches,
    mean_ci,
    paired_delta,
    percentile,
    ranking_metrics,
    source_matches,
)
from src.evaluation.runner import collapse_to_documents


def doc(source: str, *, page: int | None = 1, content: str = "text", chunk_id: str | None = None):
    return RetrievedDocument(
        content=content,
        metadata={"source": source, "page": page, "chunk_id": chunk_id},
        score=1.0,
        collection="c",
        kind="text",
        index="i",
    )


def metrics(ranking, labels, ks=(1, 2, 5)):
    return ranking_metrics(judge_ranking(ranking, labels), labels, ks)


# --- matching ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("wanted", "actual", "expected"),
    [
        ("/data/a/report.pdf", "/data/a/report.pdf", True),  # full path
        ("report.pdf", "/data/a/report.pdf", True),  # file name
        ("a/report.pdf", "/data/a/report.pdf", True),  # path suffix on a boundary
        ("port.pdf", "/data/a/report.pdf", False),  # not a boundary
        ("a.pdf", "/data/data.pdf", False),
        ("report.pdf", None, False),
        ("report.pdf", "", False),
    ],
)
def test_source_matching(wanted, actual, expected):
    assert source_matches(wanted, actual) is expected


def test_a_label_matches_only_when_every_key_it_sets_matches():
    d = doc("/x/r.pdf", page=3, content="Cloud revenue grew TWELVE percent", chunk_id="c1")
    assert label_matches(Label(source="r.pdf"), d)
    assert label_matches(Label(source="r.pdf", pages=(2, 3)), d)
    assert not label_matches(Label(source="r.pdf", pages=(4,)), d)
    assert label_matches(Label(contains="twelve percent"), d), "case-insensitive substring"
    assert not label_matches(Label(source="r.pdf", contains="thirteen"), d)
    assert label_matches(Label(chunk_id="c1"), d) and not label_matches(Label(chunk_id="c2"), d)
    assert not label_matches(Label(source="r.pdf", pages=(1,)), doc("/x/r.pdf", page=None)), (
        "no page, no page match"
    )


# --- metrics, by hand ---------------------------------------------------------------------------
def test_binary_relevance_metrics_match_hand_computation():
    labels = [Label(source="a"), Label(source="b"), Label(source="never")]
    ranking = [doc("x"), doc("a"), doc("y"), doc("b"), doc("z")]
    m = metrics(ranking, labels)
    assert m["hit@1"] == 0.0 and m["hit@2"] == 1.0
    assert m["recall@2"] == pytest.approx(1 / 3) and m["recall@5"] == pytest.approx(2 / 3)
    assert m["precision@5"] == pytest.approx(2 / 5) and m["precision@1"] == 0.0
    assert m["mrr@5"] == pytest.approx(1 / 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3) + 1 / 2
    assert m["ndcg@5"] == pytest.approx(dcg / ideal)


def test_graded_relevance_rewards_putting_the_better_evidence_first():
    labels = [Label(source="best", grade=3), Label(source="ok", grade=1)]
    good = metrics([doc("best"), doc("ok")], labels)
    bad = metrics([doc("ok"), doc("best")], labels)
    assert good["ndcg@2"] == pytest.approx(1.0)
    assert bad["ndcg@2"] == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    assert bad["ndcg@2"] < good["ndcg@2"]
    assert good["recall@2"] == bad["recall@2"] == 1.0


def test_several_chunks_of_one_label_count_once():
    labels = [Label(source="a"), Label(source="b")]
    ranking = [doc("a", content="1"), doc("a", content="2"), doc("a", content="3"), doc("b")]
    m = metrics(ranking, labels, ks=(4,))
    assert m["recall@4"] == 1.0
    assert m["precision@4"] == 1.0, "every one of the four chunks is relevant"
    assert m["ndcg@4"] == pytest.approx((1 + 1 / math.log2(5)) / (1 + 1 / math.log2(3))), (
        "the duplicates earn no gain: label b is found at rank 4, not rank 2"
    )
    assert m["ndcg@4"] < 1.0


def test_a_document_matching_two_labels_finds_both_without_exceeding_a_perfect_score():
    labels = [Label(contains="alpha"), Label(contains="beta")]
    both = doc("a", content="alpha and beta")
    m = metrics([both], labels, ks=(1,))
    assert m["recall@1"] == 1.0 and m["ndcg@1"] <= 1.0


def test_metrics_with_fewer_results_than_k_and_nothing_found():
    labels = [Label(source="a")]
    m = metrics([doc("x")], labels, ks=(1, 10))
    assert all(v == 0.0 for v in m.values())
    assert metrics([], labels, ks=(3,))["precision@3"] == 0.0
    assert metrics([doc("a")], labels, ks=(10,))["precision@10"] == pytest.approx(0.1), "denominator is k"


def test_collapse_to_documents_keeps_each_documents_best_ranked_chunk_in_order():
    ranking = [doc("a", content="1"), doc("b"), doc("a", content="2"), doc("c")]
    assert [(d.metadata["source"], d.content) for d in collapse_to_documents(ranking)] == [
        ("a", "1"),
        ("b", "text"),
        ("c", "text"),
    ]


# --- statistics --------------------------------------------------------------------------------
def test_mean_ci_brackets_the_mean_is_reproducible_and_narrows_with_more_data():
    few = [0.0, 1.0] * 5
    many = [0.0, 1.0] * 200
    a, b = mean_ci(few), mean_ci(many)
    assert a.low <= a.mean <= a.high and a.mean == pytest.approx(0.5)
    assert (b.high - b.low) < (a.high - a.low)
    assert mean_ci(few) == a, "fixed seed: the same data gives the same interval"
    assert mean_ci([0.7]).low == mean_ci([0.7]).high == 0.7
    with pytest.raises(ValueError):
        mean_ci([])


def test_paired_delta_detects_a_consistent_improvement_and_ignores_pure_noise():
    base = {f"q{i}": 0.5 for i in range(40)}
    better = {k: v + 0.1 for k, v in base.items()}
    d = paired_delta(base, better)
    assert d.diff == pytest.approx(0.1) and d.significant and d.p_value < 0.01 and d.n == 40

    same = paired_delta(base, dict(base))
    assert same.diff == 0.0 and not same.significant

    noisy_a = {f"q{i}": float(i % 2) for i in range(40)}
    noisy_b = {f"q{i}": float((i + 1) % 2) for i in range(40)}  # per-query swings that cancel out
    assert not paired_delta(noisy_a, noisy_b).significant


def test_a_handful_of_queries_is_never_flagged_significant():
    for n in (1, 2, 9):
        d = paired_delta({f"q{i}": 0.0 for i in range(n)}, {f"q{i}": 1.0 for i in range(n)})
        assert d.diff == 1.0 and not d.significant, "too few pairs for an interval to mean anything"
    assert paired_delta({f"q{i}": 0.0 for i in range(10)}, {f"q{i}": 1.0 for i in range(10)}).significant


def test_paired_delta_uses_only_queries_both_runs_scored():
    d = paired_delta({"a": 0.0, "b": 0.0, "only-a": 1.0}, {"a": 1.0, "b": 1.0, "only-b": 0.0})
    assert d.n == 2 and d.diff == 1.0 and not d.significant
    with pytest.raises(ValueError, match="share no queries"):
        paired_delta({"a": 1.0}, {"b": 1.0})


def test_percentile_is_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 0.5) == 50 and percentile(values, 0.95) == 95 and percentile([7], 0.99) == 7
    assert (
        percentile([1, 2, 3, 4], 0.5) == 2
        and percentile([1, 2, 3, 4], 1.0) == 4
        and percentile([1, 2, 3, 4], 0.0) == 1
    )
