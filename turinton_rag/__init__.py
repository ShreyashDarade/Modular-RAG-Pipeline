"""turinton-rag: one SDK for the RAG backend.

Remote (thin - needs only ``httpx`` and ``pydantic``)::

    from turinton_rag import AsyncRagClient, RagClient

    with RagClient("http://localhost:8000") as rag:
        answer = rag.ask("What changed in Q2?", collections=["finance"])

In-process (the engine inside your application - ``pip install 'turinton-rag[engine]'``)::

    from turinton_rag import AsyncRag, Rag

Both expose the same interface (:class:`AsyncRagAPI` / :class:`RagAPI`), the same models
(``turinton_rag.models``) and the same typed errors (``turinton_rag.errors``). What is public, and how it may
change, is in ``docs/framework.md``; everything outside this package is internal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from turinton_rag._compat import (
    EXPERIMENTAL,
    RagDeprecationWarning,
    RagFutureWarning,
    deprecated,
    experimental,
)
from turinton_rag._facade import AsyncRagAPI
from turinton_rag._sync import RagAPI
from turinton_rag._version import __version__
from turinton_rag.client import AsyncRagClient, RagClient

if TYPE_CHECKING:  # the engine is imported only when asked for, so `import turinton_rag` stays thin
    from src.evaluation.report import EvalReport

    from turinton_rag.embedded import AsyncRag, Rag

_LAZY = {
    "AsyncRag": "turinton_rag.embedded",
    "Rag": "turinton_rag.embedded",
    "EvalReport": "src.evaluation.report",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'turinton_rag' has no attribute {name!r}")
    import importlib

    try:
        value = getattr(importlib.import_module(module), name)
    except ImportError as exc:
        raise ImportError(
            f"turinton_rag.{name} needs the engine: pip install 'turinton-rag[engine]' ({exc})"
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
