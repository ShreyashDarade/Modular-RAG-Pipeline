"""Run an evaluation dataset through a retrieval pipeline and score every query."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from src.core.errors import RagError
from src.core.types import RetrievalResult, RetrievalScope, RetrievedDocument
from src.evaluation.dataset import Dataset, EvalCase
from src.evaluation.metrics import judge_ranking, ranking_metrics
from src.retrieval.pipeline import RetrievalPipeline

Granularity = Literal["chunk", "document"]


@dataclass(slots=True)
class RankedItem:
    source: str | None
    page: int | None
    chunk_id: str | None
    score: float
    kind: str
    matched: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "kind": self.kind,
            "matched": list(self.matched),
        }


@dataclass(slots=True)
class CaseResult:
    id: str
    tags: tuple[str, ...]
    ranked: list[RankedItem]
    #: Retrieval metrics; empty when the case has no labelled evidence.
    metrics: dict[str, float]
    #: Indices of labels no ranked document matched within the largest cutoff.
    missed: list[int]
    latency_ms: float
    error: str | None = None
    #: Filled by the answer evaluator.
    answer: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tags": list(self.tags),
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "missed": self.missed,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "ranked": [item.to_dict() for item in self.ranked],
            **({"answer": self.answer} if self.answer is not None else {}),
        }


def collapse_to_documents(docs: Sequence[RetrievedDocument]) -> list[RetrievedDocument]:
    """Keep each (collection, source)'s best-ranked chunk, in rank order - for corpora where the
    unit of relevance is the whole document and several chunks of one document are one hit."""
    seen: set[tuple[str, object]] = set()
    out = []
    for doc in docs:
        key = (doc.collection, doc.metadata.get("source"))
        if key not in seen:
            seen.add(key)
            out.append(doc)
    return out


def zero_metrics(ks: Sequence[int]) -> dict[str, float]:
    names = [f"{m}@{k}" for k in ks for m in ("hit", "recall", "precision", "ndcg")]
    return {**dict.fromkeys(names, 0.0), f"mrr@{max(ks)}": 0.0}


def score_case(
    case: EvalCase, docs: Sequence[RetrievedDocument], ks: Sequence[int], granularity: Granularity
) -> tuple[list[RankedItem], dict[str, float], list[int]]:
    ranking = collapse_to_documents(docs) if granularity == "document" else list(docs)
    judged = judge_ranking(ranking, case.labels)
    items = [
        RankedItem(
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("chunk_id"),
            d.final_score,
            d.kind,
            j.matched,
        )
        for d, j in zip(ranking, judged, strict=True)
    ]
    if not case.labels:
        return items, {}, []
    found = {i for j in judged[: max(ks)] for i in j.matched}
    missed = [i for i in range(len(case.labels)) if i not in found]
    return items, ranking_metrics(judged, case.labels, ks), missed


class RetrievalEvaluator:
    def __init__(
        self,
        retrieval: RetrievalPipeline,
        *,
        ks: Sequence[int] = (1, 3, 5, 10),
        granularity: Granularity = "chunk",
        concurrency: int = 4,
        default_collection: str | None = None,
    ) -> None:
        if not ks or any(k < 1 for k in ks):
            raise ValueError("cutoffs must be positive integers")
        self._retrieval = retrieval
        self._ks = sorted(set(ks))
        self._granularity: Granularity = granularity
        self._slots = asyncio.Semaphore(concurrency)
        self._default_collection = default_collection

    def scope_for(self, case: EvalCase) -> RetrievalScope:
        name = case.collection or self._default_collection
        return self._retrieval.scope([name] if name else None, case.kinds)

    async def retrieve(self, case: EvalCase) -> tuple[RetrievalResult, float]:
        async with self._slots:
            started = time.perf_counter()
            result = await self._retrieval.retrieve(case.query, self.scope_for(case))
            return result, (time.perf_counter() - started) * 1000

    async def run_case(self, case: EvalCase) -> CaseResult:
        try:
            result, latency = await self.retrieve(case)
        except RagError as exc:
            # a failed query scores zero and is reported; it is never dropped from the average
            metrics = zero_metrics(self._ks) if case.labels else {}
            return CaseResult(
                case.id,
                case.tags,
                [],
                metrics,
                list(range(len(case.labels))),
                0.0,
                error=f"{exc.code}: {exc}",
            )
        items, metrics, missed = score_case(case, result.documents, self._ks, self._granularity)
        return CaseResult(case.id, case.tags, items, metrics, missed, latency)

    async def run(
        self, dataset: Dataset, progress: Callable[[int, int], None] | None = None
    ) -> list[CaseResult]:
        results: list[CaseResult] = []
        done = 0

        async def one(case: EvalCase) -> CaseResult:
            nonlocal done
            result = await self.run_case(case)
            done += 1
            if progress:
                progress(done, len(dataset))
            return result

        results = await asyncio.gather(*(one(case) for case in dataset.cases))
        return list(results)
