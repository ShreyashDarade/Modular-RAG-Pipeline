"""Stability tiers and the deprecation lifecycle (``docs/framework.md``, sections 6 and 7).

* ``@experimental`` marks public names that may change in any minor release.
* ``@deprecated(since=..., remove_in=..., alternative=...)`` is the *only* way to deprecate: the metadata is
  mandatory, the warning class is the SDK's own (so it can be filtered and tested), and ``stacklevel`` points
  at the caller.
"""

from __future__ import annotations

import functools
import inspect
import re
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

_VERSION = re.compile(r"^\d+\.\d+(\.\d+)?$")

#: Qualified names of every ``@experimental`` object, filled as modules import.
EXPERIMENTAL: set[str] = set()


class RagDeprecationWarning(DeprecationWarning):
    """A public name is deprecated. Hidden by default outside ``__main__`` (PEP 565): CI should turn it into an error."""


class RagFutureWarning(FutureWarning):
    """Like :class:`RagDeprecationWarning`, in the last minor release before removal - shown to end users too."""


def experimental[T](obj: T) -> T:
    """Mark ``obj`` as experimental: it may change or disappear in any minor release."""
    qualname = f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', repr(obj))}"
    EXPERIMENTAL.add(qualname)
    obj.__rag_experimental__ = True  # type: ignore[attr-defined]
    doc = inspect.getdoc(obj) or ""
    obj.__doc__ = f"{doc}\n\n.. warning:: Experimental: may change in any minor release.".strip()
    return obj


def internal_init[C: type](cls: C) -> C:
    """Mark a public class whose constructor is *not* part of the public API (it takes internal types; use
    the documented factory instead). The API-surface snapshot omits such constructors."""
    cls.__rag_internal_init__ = True  # type: ignore[attr-defined]
    return cls


def _check_version(label: str, value: str) -> None:
    if not _VERSION.match(value):
        raise ValueError(f"deprecated(): {label}={value!r} is not a version like '2.3' or '2.3.0'")


def deprecated(
    *, since: str, remove_in: str, alternative: str | None = None, reason: str | None = None
) -> Callable[[F], F]:
    """Deprecate a public function or method.

    ``since`` and ``remove_in`` are versions; ``alternative`` names the replacement - or give ``reason``
    when there is none. The window between them must respect the policy (at least two minor releases and
    removal only in a major release); that is checked here, not left to reviewers.
    """
    _check_version("since", since)
    _check_version("remove_in", remove_in)
    if not (alternative or reason):
        raise ValueError("deprecated(): give the replacement (`alternative`) or the `reason` there is none")
    since_major, since_minor = (int(x) for x in since.split(".")[:2])
    remove_major, remove_minor = (int(x) for x in remove_in.split(".")[:2])
    if remove_major <= since_major:
        raise ValueError(f"deprecated(): removal in {remove_in} is not a later major release than {since}")
    del since_minor, remove_minor

    def decorate(func: F) -> F:
        message = f"{func.__qualname__} is deprecated since {since} and will be removed in {remove_in}. " + (
            f"Use {alternative} instead." if alternative else reason or ""
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, RagDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__rag_deprecated__ = {"since": since, "remove_in": remove_in}  # type: ignore[attr-defined]
        wrapper.__doc__ = f"{inspect.getdoc(func) or ''}\n\n.. deprecated:: {since}\n   {message}".strip()
        return cast(F, wrapper)

    return decorate
