"""Evaluate a built engine: retrieval metrics and (optionally) judged answers, as one report.

Used by the ``rag eval`` CLI (one container per configuration variant) and by the SDK's
``evaluate()`` (the engine the application already runs). Building and closing the container is the caller's
concern; this function only measures.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.core.container import Container
from src.evaluation.answers import AnswerEvaluator
from src.evaluation.dataset import Dataset
from src.evaluation.judge import LlmJudge
from src.evaluation.report import EvalReport, build_report
from src.evaluation.runner import Granularity, RetrievalEvaluator

Progress = Callable[[str, int, int], None]


async def evaluate_container(
    container: Container,
    dataset: Dataset,
    *,
    name: str = "run",
    collection: str | None = None,
    ks: Sequence[int] = (1, 3, 5, 10),
    granularity: Granularity = "chunk",
    answers: bool = False,
    judge_model: str | None = None,
    answer_model: str | None = None,
    concurrency: int = 4,
    overrides: Mapping[str, str] | None = None,
    progress: Progress | None = None,
) -> EvalReport:
    s, c = container.settings, container.config
    cutoffs = sorted(set(ks))
    evaluator = RetrievalEvaluator(
        container.retrieval,
        ks=cutoffs,
        granularity=granularity,
        default_collection=collection,
        concurrency=concurrency,
    )
    results = await evaluator.run(dataset, (lambda d, t: progress("retrieval", d, t)) if progress else None)
    judge: LlmJudge | None = None
    if answers:
        judge = LlmJudge(container.models.chat(judge_model or c.utility_model))
        await AnswerEvaluator(container.answers, judge, evaluator, answer_model=answer_model).run(
            dataset, results, (lambda d, t: progress("answers", d, t)) if progress else None
        )
    spec = c.reranker_spec()
    meta: dict[str, Any] = {
        "collection": collection or c.default_collection,
        "ks": cutoffs,
        "granularity": granularity,
        "concurrency": concurrency,
        "short_rankings": sum(1 for r in results if r.error is None and len(r.ranked) < max(cutoffs)),
        "config": {
            "reranker": f"{spec.provider}:{spec.model}" if spec.model else spec.provider,
            "query_expander": c.query_expander,
            "hybrid_alpha": s.hybrid_alpha,
            "candidates": s.rerank_candidates,
            "top_k": s.retriever_top_k,
            "fuzziness": s.es_bm25_fuzziness,
            **({"overrides": dict(overrides)} if overrides else {}),
        },
        **({"judge_model": judge.model_id} if judge else {}),
    }
    return build_report(name, dataset, results, meta)
