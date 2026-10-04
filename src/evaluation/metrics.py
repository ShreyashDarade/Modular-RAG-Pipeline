"""Ranking metrics and the statistics around them - pure functions, standard library only.

Per query, a *ranking* is a list of retrieved documents and ``labels`` is the evidence a good
ranking contains. A document *matches* a label as defined in :mod:`src.evaluation.dataset`. Each
label can be found once: later documents that match only already-found labels earn no gain, so a
corpus that splits one page over five chunks cannot inflate recall or nDCG.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath

from src.core.types import RetrievedDocument
from src.evaluation.dataset import Label


def source_matches(wanted: str, actual: object) -> bool:
    """``wanted`` equals the chunk's source, its file name, or a trailing part of its path."""
    if not isinstance(actual, str) or not actual:
        return False
    if wanted == actual or PurePath(actual).name == wanted:
        return True
    return actual.endswith("/" + wanted.lstrip("/"))


def label_matches(label: Label, doc: RetrievedDocument) -> bool:
    meta = doc.metadata
    if label.source is not None and not source_matches(label.source, meta.get("source")):
        return False
    if label.pages and meta.get("page") not in label.pages:
        return False
    if label.chunk_id is not None and meta.get("chunk_id") != label.chunk_id:
        return False
    return label.contains is None or label.contains.lower() in doc.content.lower()


@dataclass(frozen=True, slots=True)
class Judged:
    """How one ranked document relates to the labels."""

    matched: tuple[int, ...]  # indices of every label the document matches
    gain: int  # the best grade among labels this document is the *first* to find (0 if none)


def judge_ranking(ranking: Sequence[RetrievedDocument], labels: Sequence[Label]) -> list[Judged]:
    found: set[int] = set()
    judged = []
    for doc in ranking:
        matched = tuple(i for i, label in enumerate(labels) if label_matches(label, doc))
        new = [i for i in matched if i not in found]
        judged.append(Judged(matched, max((labels[i].grade for i in new), default=0)))
        found.update(new)
    return judged


def _dcg(gains: Sequence[int]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ranking_metrics(judged: Sequence[Judged], labels: Sequence[Label], ks: Sequence[int]) -> dict[str, float]:
    """``hit@k``, ``recall@k``, ``precision@k``, ``ndcg@k`` for every k, and ``mrr@max(k)``."""
    out: dict[str, float] = {}
    ideal = sorted((label.grade for label in labels), reverse=True)
    for k in ks:
        top = judged[:k]
        found = {i for j in top for i in j.matched}
        out[f"hit@{k}"] = 1.0 if found else 0.0
        out[f"recall@{k}"] = len(found) / len(labels)
        out[f"precision@{k}"] = sum(1 for j in top if j.matched) / k
        out[f"ndcg@{k}"] = _dcg([j.gain for j in top]) / _dcg(ideal[:k])
    cutoff = max(ks)
    first = next((rank for rank, j in enumerate(judged[:cutoff], start=1) if j.matched), None)
    out[f"mrr@{cutoff}"] = 1.0 / first if first else 0.0
    return out


# --- statistics --------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int


def mean_ci(
    values: Sequence[float], *, confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> Interval:
    """Percentile-bootstrap interval for the mean. With one value it is degenerate (low = high)."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    mean = sum(values) / n
    if n == 1:
        return Interval(mean, mean, mean, 1)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    return Interval(
        mean, means[int(tail * resamples)], means[min(resamples - 1, int((1 - tail) * resamples))], n
    )


#: Fewer paired queries than this and a bootstrap interval says nothing; nothing is flagged significant.
MIN_PAIRS = 10


@dataclass(frozen=True, slots=True)
class PairedDelta:
    """``b - a`` over the queries both scored."""

    diff: float
    low: float
    high: float
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        """The 95% interval excludes zero (and there are enough queries for a bootstrap to mean anything).
        One of many comparisons: expect false positives."""
        return self.n >= MIN_PAIRS and (self.low > 0 or self.high < 0)


def paired_delta(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: int = 0,
) -> PairedDelta:
    """Bootstrap the mean per-query difference ``b - a`` (same queries, resampled together), which is
    far tighter than comparing two independent intervals because query difficulty cancels out."""
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("the two runs share no queries")
    diffs = [b[i] - a[i] for i in ids]
    n = len(diffs)
    diff = sum(diffs) / n
    if n == 1:
        return PairedDelta(diff, diff, diff, 1.0, 1)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    low, high = means[int(tail * resamples)], means[min(resamples - 1, int((1 - tail) * resamples))]
    below = sum(1 for m in means if m <= 0) / resamples
    above = sum(1 for m in means if m >= 0) / resamples
    p = min(1.0, 2 * min(below, above))
    return PairedDelta(diff, low, high, max(p, 1 / (resamples + 1)), n)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, ``q`` in [0, 1]."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]
