"""Draft an evaluation set from the indexed corpus: sample chunks, have a chat model write a question
each chunk answers, and label that chunk's source (and page) as the evidence.

Be clear about what this gives you. It bootstraps a set quickly and is good for regression
tracking, but synthetic questions are written *from* the passage and tend to reuse its wording, which
flatters lexical retrieval; absolute scores run higher than on real user questions. Review a sample
of the questions and prefer a hand-labelled set, or real queries, for decisions that matter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from src.core.config import Settings
from src.core.specs import RagConfig
from src.core.types import ChatMessage, ContentKind, RawHit
from src.evaluation.dataset import EvalCase, Label
from src.evaluation.judge import JudgeError, extract_json
from src.ports.indexing import Searcher
from src.ports.models import ChatModel

GENERATE_SYSTEM = (
    "You write evaluation questions for a document search system. Given a passage, write ONE question that "
    "the passage answers completely, that a person who has not seen the passage might plausibly ask, and that "
    "does not copy the passage's distinctive phrases. Also give the short answer, using only the passage. "
    'Reply with JSON only: {"question": "...", "answer": "..."}'
)


@dataclass(slots=True)
class Generated:
    cases: list[EvalCase] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def _label_for(hit: RawHit) -> Label | None:
    source = hit.source.get("source")
    if not isinstance(source, str) or not source:
        return None
    page = hit.source.get("page")
    return Label(source=source, pages=(page,) if isinstance(page, int) and not isinstance(page, bool) else ())


async def generate_cases(
    searcher: Searcher,
    config: RagConfig,
    chat: ChatModel,
    *,
    collection: str,
    count: int,
    seed: int = 7,
    kind: ContentKind = "text",
    min_chars: int = 200,
    concurrency: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> Generated:
    index = config.collection(collection).index_name(kind)
    hits = await searcher.sample(index, size=max(count * 3, count + 10), seed=seed)
    result = Generated()
    usable = []
    for hit in hits:
        text = str(hit.source.get("content", ""))
        if len(text) < min_chars:
            result.skip("passage too short")
        elif _label_for(hit) is None:
            result.skip("chunk has no source")
        else:
            usable.append(hit)
    slots = asyncio.Semaphore(concurrency)
    drafted: dict[int, EvalCase] = {}

    async def draft(position: int, hit: RawHit) -> None:
        async with slots:
            if len(drafted) >= count:
                return
            try:
                reply = extract_json(
                    await chat.complete(
                        [
                            ChatMessage("system", GENERATE_SYSTEM),
                            ChatMessage("user", f"Passage:\n{hit.source['content']}"),
                        ]
                    )
                )
            except JudgeError:
                result.skip("model did not return JSON")
                return
            question, answer = reply.get("question"), reply.get("answer")
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
                result.skip("model reply missing question/answer")
                return
            label = _label_for(hit)
            assert label is not None
            drafted[position] = EvalCase(
                id=f"gen-{seed}-{position:04d}",
                query=question.strip(),
                labels=(label,),
                reference_answer=answer.strip() or None,
                tags=("synthetic",),
                collection=collection,
            )
            if progress:
                progress(min(len(drafted), count), count)

    await asyncio.gather(*(draft(i, hit) for i, hit in enumerate(usable)))
    result.cases = [drafted[i] for i in sorted(drafted)][:count]
    return result


def default_chat(settings: Settings, config: RagConfig) -> str:
    return config.utility_model
