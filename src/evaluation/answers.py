"""Answer-quality evaluation: generate with the real answer service, then score each answer."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from src.chat.answer import AnswerService
from src.core.errors import RagError
from src.core.types import RetrievedDocument
from src.evaluation.dataset import Dataset, EvalCase
from src.evaluation.judge import JudgeError, LlmJudge
from src.evaluation.metrics import source_matches
from src.evaluation.runner import CaseResult, RetrievalEvaluator

# the format the answer prompt asks for: [Source: report.pdf, Page: 3]
_CITATION = re.compile(r"\[Source:\s*([^,\]]+?)\s*(?:,\s*Page:\s*([^\]]*?))?\s*\]", re.IGNORECASE)


def citations(answer: str) -> list[tuple[str, str | None]]:
    return [(m.group(1).strip(), (m.group(2) or "").strip() or None) for m in _CITATION.finditer(answer)]


def citation_is_valid(source: str, page: str | None, documents: list[RetrievedDocument]) -> bool:
    """The cited source was actually retrieved (and, if a page is cited, that page of it)."""
    for doc in documents:
        if not source_matches(source, doc.metadata.get("source")):
            continue
        if page is None or str(doc.metadata.get("page")) == page:
            return True
    return False


class AnswerEvaluator:
    def __init__(
        self,
        answers: AnswerService,
        judge: LlmJudge,
        retrieval: RetrievalEvaluator,
        *,
        answer_model: str | None = None,
        concurrency: int = 2,
    ) -> None:
        self._answers = answers
        self._judge = judge
        self._retrieval = retrieval
        self._model = answer_model
        self._slots = asyncio.Semaphore(concurrency)

    async def _score(self, case: EvalCase, result: CaseResult) -> None:
        async with self._slots:
            try:
                answer = await self._answers.ask(case.query, self._retrieval.scope_for(case), self._model)
            except RagError as exc:
                result.answer = {"error": f"{exc.code}: {exc}"}
                return
            metrics: dict[str, float] = {}
            errors: list[str] = []
            detail: dict[str, Any] = {"text": answer.answer, "model": answer.model}

            cited = citations(answer.answer)
            if case.answerable:
                metrics["citation_rate"] = 1.0 if cited else 0.0
            if cited:
                valid = [citation_is_valid(s, p, answer.documents) for s, p in cited]
                metrics["citation_valid"] = sum(valid) / len(valid)

            failed = False

            async def judged(name: str, call: Callable[[], Any]) -> Any:
                nonlocal failed
                try:
                    return await call()
                except JudgeError as exc:  # an unparseable verdict: this case leaves that metric out
                    errors.append(f"{name}: {exc}")
                except RagError as exc:  # the judge model itself failed (rate limit, timeout, ...)
                    errors.append(f"{name}: {exc.code}: {exc}")
                    failed = True
                return None

            if case.answerable:
                faith = await judged(
                    "faithfulness",
                    lambda: self._judge.faithfulness(case.query, answer.documents, answer.answer),
                )
                if faith is not None and faith.score is not None:
                    metrics["faithfulness"] = faith.score
                    detail["claims"] = [{"claim": c, "supported": ok} for c, ok in faith.claims]
                if case.reference_answer:
                    score = await judged(
                        "correctness",
                        lambda: self._judge.correctness(
                            case.query, case.reference_answer or "", answer.answer
                        ),
                    )
                    if score is not None:
                        metrics["correctness"] = score
            else:
                abstained = await judged(
                    "abstention", lambda: self._judge.abstained(case.query, answer.answer)
                )
                if abstained is not None:
                    metrics["abstained"] = 1.0 if abstained else 0.0
            result.answer = {
                **detail,
                "metrics": metrics,
                **({"judge_errors": errors} if errors else {}),
                **({"failed": True} if failed else {}),
            }

    async def run(
        self, dataset: Dataset, results: list[CaseResult], progress: Callable[[int, int], None] | None = None
    ) -> None:
        by_id = {c.id: c for c in dataset.cases}
        done = 0

        async def one(result: CaseResult) -> None:
            nonlocal done
            await self._score(by_id[result.id], result)
            done += 1
            if progress:
                progress(done, len(results))

        await asyncio.gather(*(one(r) for r in results))
