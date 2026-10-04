"""Evaluation datasets: one JSON object per line.

.. code-block:: json

    {"id": "q1", "query": "How did cloud revenue change in Q2?",
     "relevant": [{"source": "report.pdf", "pages": [3, 4]}, {"contains": "twelve percent", "grade": 2}],
     "reference_answer": "It grew twelve percent.", "answerable": true, "tags": ["finance"]}

``relevant`` lists the *evidence* a good retrieval must surface. A label matches a retrieved chunk
when **every** key it sets matches: ``source`` (full path, file name or path suffix), ``pages``,
``chunk_id``, ``contains`` (case-insensitive substring of the chunk text). ``grade`` (default 1)
is the label's relevance for nDCG. A question the corpus cannot answer has ``"answerable": false``
and no labels: it is skipped by retrieval metrics and used to check that the answer abstains.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.errors import ConfigError

_CASE_KEYS = {"id", "query", "relevant", "reference_answer", "answerable", "tags", "collection", "kinds"}
_LABEL_KEYS = {"source", "pages", "chunk_id", "contains", "grade"}


@dataclass(frozen=True, slots=True)
class Label:
    """One piece of evidence. At least one matcher (source, pages, chunk_id, contains) must be set."""

    source: str | None = None
    pages: tuple[int, ...] = ()
    chunk_id: str | None = None
    contains: str | None = None
    grade: int = 1

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.source is not None:
            out["source"] = self.source
        if self.pages:
            out["pages"] = list(self.pages)
        if self.chunk_id is not None:
            out["chunk_id"] = self.chunk_id
        if self.contains is not None:
            out["contains"] = self.contains
        if self.grade != 1:
            out["grade"] = self.grade
        return out


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    query: str
    labels: tuple[Label, ...] = ()
    reference_answer: str | None = None
    answerable: bool = True
    tags: tuple[str, ...] = ()
    collection: str | None = None
    kinds: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "query": self.query}
        if self.labels:
            out["relevant"] = [label.to_dict() for label in self.labels]
        if self.reference_answer is not None:
            out["reference_answer"] = self.reference_answer
        if not self.answerable:
            out["answerable"] = False
        if self.tags:
            out["tags"] = list(self.tags)
        if self.collection:
            out["collection"] = self.collection
        if self.kinds:
            out["kinds"] = list(self.kinds)
        return out


@dataclass(frozen=True, slots=True)
class Dataset:
    cases: tuple[EvalCase, ...]
    #: Identifies the content, so reports can only be compared when they scored the same questions.
    fingerprint: str = field(default="")

    def __len__(self) -> int:
        return len(self.cases)


def _fail(where: str, message: str) -> ConfigError:
    return ConfigError(f"{where}: {message}")


def _label(raw: Any, where: str) -> Label:
    if not isinstance(raw, dict):
        raise _fail(where, "each `relevant` entry must be an object")
    unknown = sorted(set(raw) - _LABEL_KEYS)
    if unknown:
        raise _fail(where, f"unknown label key(s) {unknown}; allowed: {sorted(_LABEL_KEYS)}")
    pages = raw.get("pages", ())
    if not isinstance(pages, list | tuple) or not all(
        isinstance(p, int) and not isinstance(p, bool) for p in pages
    ):
        raise _fail(where, "`pages` must be a list of integers")
    grade = raw.get("grade", 1)
    if not isinstance(grade, int) or isinstance(grade, bool) or grade < 1:
        raise _fail(where, "`grade` must be an integer >= 1")
    for key in ("source", "chunk_id", "contains"):
        if key in raw and (not isinstance(raw[key], str) or not raw[key].strip()):
            raise _fail(where, f"`{key}` must be a non-empty string")
    label = Label(
        source=raw.get("source"),
        pages=tuple(pages),
        chunk_id=raw.get("chunk_id"),
        contains=raw.get("contains"),
        grade=grade,
    )
    if label.source is None and label.chunk_id is None and label.contains is None:
        raise _fail(
            where, "a label needs `source`, `chunk_id` or `contains` (`pages` alone matches too much)"
        )
    return label


def parse_case(raw: Any, where: str) -> EvalCase:
    if not isinstance(raw, dict):
        raise _fail(where, "each line must be a JSON object")
    unknown = sorted(set(raw) - _CASE_KEYS)
    if unknown:
        raise _fail(where, f"unknown key(s) {unknown}; allowed: {sorted(_CASE_KEYS)}")
    for key in ("id", "query"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise _fail(where, f"`{key}` is required and must be a non-empty string")
    labels_raw = raw.get("relevant", [])
    if not isinstance(labels_raw, list):
        raise _fail(where, "`relevant` must be a list")
    labels = tuple(_label(entry, where) for entry in labels_raw)
    answerable = raw.get("answerable", True)
    if not isinstance(answerable, bool):
        raise _fail(where, "`answerable` must be true or false")
    if not answerable and labels:
        raise _fail(where, "an unanswerable question cannot have relevant evidence")
    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise _fail(where, "`tags` must be a list of strings")
    reference = raw.get("reference_answer")
    if reference is not None and not isinstance(reference, str):
        raise _fail(where, "`reference_answer` must be a string")
    kinds = raw.get("kinds")
    if kinds is not None and not (isinstance(kinds, list) and all(isinstance(k, str) for k in kinds)):
        raise _fail(where, "`kinds` must be a list of strings")
    return EvalCase(
        id=raw["id"],
        query=raw["query"].strip(),
        labels=labels,
        reference_answer=reference,
        answerable=answerable,
        tags=tuple(tags),
        collection=raw.get("collection"),
        kinds=tuple(kinds) if kinds else None,
    )


def fingerprint_of(cases: tuple[EvalCase, ...]) -> str:
    canonical = json.dumps([c.to_dict() for c in cases], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_dataset(cases: list[EvalCase]) -> Dataset:
    if not cases:
        raise ConfigError("the evaluation dataset has no cases")
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ConfigError(f"duplicate case id '{case.id}'")
        seen.add(case.id)
    frozen = tuple(cases)
    return Dataset(frozen, fingerprint_of(frozen))


def load_dataset(path: Path) -> Dataset:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"cannot read dataset {path}: {exc}") from exc
    cases: list[EvalCase] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _fail(where, f"not valid JSON ({exc.msg})") from exc
        cases.append(parse_case(raw, where))
    return build_dataset(cases)


def write_dataset(cases: list[EvalCase], path: Path) -> None:
    path.write_text(
        "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in cases), encoding="utf-8"
    )
