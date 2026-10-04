from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from src.core.errors import RagError, UpstreamError
from src.core.types import JobRecord, JobSpec


def is_retryable(exc: BaseException) -> bool:
    """Transient failures (a dependency hiccup, an I/O error, a timeout) are worth another
    attempt; bad input, missing files and configuration errors are not."""
    return isinstance(exc, UpstreamError | OSError | TimeoutError)


def describe(exc: BaseException) -> str:
    """Client-visible failure text. Details of dependency or internal failures stay in the log."""
    if isinstance(exc, RagError):
        return f"{type(exc).__name__}: {exc.public_message}"
    return f"{type(exc).__name__}: internal error; details are in the server log"


def failure(record: JobRecord, exc: BaseException) -> None:
    """Record a terminal failure, keeping the typed error's code and HTTP status so a waiting
    API caller can answer with the right status instead of a generic 500."""
    record.status = "failed"
    record.error = describe(exc)
    record.error_code = getattr(exc, "code", "internal_error")
    record.error_status = getattr(exc, "status_code", 500)


def encode(record: JobRecord) -> str:
    return json.dumps(asdict(record), ensure_ascii=False)


def decode(raw: str | bytes) -> JobRecord:
    data: dict[str, Any] = json.loads(raw)
    spec = data.pop("spec")
    if spec.get("kinds") is not None:
        spec["kinds"] = tuple(spec["kinds"])
    return JobRecord(spec=JobSpec(**spec), **data)
