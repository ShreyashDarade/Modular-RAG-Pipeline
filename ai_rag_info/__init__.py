"""ai-rag-info: one SDK for the RAG backend.

Remote (thin - needs only ``httpx`` and ``pydantic``)::

    from ai_rag_info import AsyncRagClient, RagClient

    with RagClient("http://localhost:8000") as rag:
        answer = rag.ask("What changed in Q2?", collections=["finance"])

In-process (the engine inside your application - ``pip install 'ai-rag-info[engine]'``)::

    from ai_rag_info import AsyncRag, Rag

Both expose the same interface (:class:`AsyncRagAPI` / :class:`RagAPI`), the same models
(``ai_rag_info.models``) and the same typed errors (``ai_rag_info.errors``). What is public, and how it may
change, is in ``docs/framework.md``; everything outside this package is internal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_rag_info._compat import (
    EXPERIMENTAL,
    RagDeprecationWarning,
    RagFutureWarning,
    deprecated,
    experimental,
)
from ai_rag_info._facade import AsyncRagAPI
from ai_rag_info._sync import RagAPI
from ai_rag_info._version import __version__
from ai_rag_info.client import AsyncRagClient, RagClient

if TYPE_CHECKING:  # the engine is imported only when asked for, so `import ai_rag_info` stays thin
    from src.evaluation.report import EvalReport

    from ai_rag_info.embedded import AsyncRag, Rag

_LAZY = {
    "AsyncRag": "ai_rag_info.embedded",
    "Rag": "ai_rag_info.embedded",
    "EvalReport": "src.evaluation.report",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'ai_rag_info' has no attribute {name!r}")
    import importlib

    try:
        value = getattr(importlib.import_module(module), name)
    except ImportError as exc:
        raise ImportError(
            f"ai_rag_info.{name} needs the engine: pip install 'ai-rag-info[engine]' ({exc})"
        ) from exc
    globals()[name] = value
    return value


__all__ = [
    "EXPERIMENTAL",
    "AsyncRag",
    "AsyncRagAPI",
    "AsyncRagClient",
    "EvalReport",
    "Rag",
    "RagAPI",
    "RagClient",
    "RagDeprecationWarning",
    "RagFutureWarning",
    "__version__",
    "deprecated",
    "experimental",
]
