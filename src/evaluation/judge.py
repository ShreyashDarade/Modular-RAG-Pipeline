"""LLM-as-judge for answers: faithfulness to the retrieved context, correctness against a reference,
and abstention on unanswerable questions.

An LLM judge is itself a model with its own errors and biases (it favours longer answers and its own
family's phrasing). Use a strong judge that is *not* the model being judged, treat small differences
as noise, and spot-check verdicts in the saved report. A reply that cannot be parsed raises
:class:`JudgeError`; the caller records it and the case is left out of that metric - never scored 0.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.chat.prompts import format_context
from src.core.errors import RagError
from src.core.types import ChatMessage, RetrievedDocument
from src.ports.models import ChatModel


class JudgeError(RagError):
    status_code = 502
    code = "judge_error"


FAITHFULNESS_SYSTEM = (
    "You check whether an answer is supported by the context it was written from. Split the answer into its "
    "individual factual claims. For each claim decide whether the context states it or directly implies it. "
    "A claim the context does not support is unsupported even if it is true in the real world. Ignore "
    "citations, greetings, and statements that the information is not available. "
    'Reply with JSON only: {"claims": [{"claim": "...", "supported": true}]}. '
    "Use an empty list when the answer makes no factual claims."
)
CORRECTNESS_SYSTEM = (
    "You compare an answer with a reference answer to the same question. Judge only whether the answer "
    "conveys the same facts as the reference: wording, length and extra correct detail do not matter; a "
    "missing key fact or a contradiction does. "
    'Reply with JSON only: {"verdict": "correct" | "partially_correct" | "incorrect", "reason": "..."}'
)
ABSTENTION_SYSTEM = (
    "You decide whether an answer declines to answer because the information is not available (for example "
    '"the documents do not say"), as opposed to actually answering. '
    'Reply with JSON only: {"abstained": true | false}'
)
VERDICT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict:
    """The JSON object in a model reply, tolerating a code fence or a sentence around it."""
    stripped = _FENCE.sub("", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise JudgeError(f"the judge did not return JSON: {text[:200]!r}")
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"the judge returned malformed JSON ({exc.msg}): {text[:200]!r}") from exc
    if not isinstance(value, dict):
        raise JudgeError(f"the judge returned {type(value).__name__}, not an object")
    return value


@dataclass(frozen=True, slots=True)
class Faithfulness:
    score: float | None  # None: the answer makes no factual claims, so there is nothing to support
    claims: tuple[tuple[str, bool], ...]


class LlmJudge:
    def __init__(self, chat: ChatModel) -> None:
        self._chat = chat
        self.model_id = chat.model_id

    async def _ask(self, system: str, user: str) -> dict:
        return extract_json(
            await self._chat.complete([ChatMessage("system", system), ChatMessage("user", user)])
        )

    async def faithfulness(
        self, question: str, documents: Sequence[RetrievedDocument], answer: str
    ) -> Faithfulness:
        data = await self._ask(
            FAITHFULNESS_SYSTEM,
            f"Question: {question}\n\nContext:\n{format_context(documents)}\n\nAnswer:\n{answer}",
        )
        raw = data.get("claims")
        if not isinstance(raw, list):
            raise JudgeError("the faithfulness reply has no `claims` list")
        claims: list[tuple[str, bool]] = []
        for entry in raw:
            if not isinstance(entry, dict) or not isinstance(entry.get("supported"), bool):
                raise JudgeError(f"malformed claim entry: {entry!r}")
            claims.append((str(entry.get("claim", "")), entry["supported"]))
        if not claims:
            return Faithfulness(None, ())
        return Faithfulness(sum(1 for _, ok in claims if ok) / len(claims), tuple(claims))

    async def correctness(self, question: str, reference: str, answer: str) -> float:
        data = await self._ask(
            CORRECTNESS_SYSTEM, f"Question: {question}\n\nReference answer:\n{reference}\n\nAnswer:\n{answer}"
        )
        verdict = data.get("verdict")
        if verdict not in VERDICT_SCORE:
            raise JudgeError(f"unknown verdict {verdict!r}; expected one of {sorted(VERDICT_SCORE)}")
        return VERDICT_SCORE[verdict]

    async def abstained(self, question: str, answer: str) -> bool:
        data = await self._ask(ABSTENTION_SYSTEM, f"Question: {question}\n\nAnswer:\n{answer}")
        value = data.get("abstained")
        if not isinstance(value, bool):
            raise JudgeError(f"`abstained` must be true or false, got {value!r}")
        return value
