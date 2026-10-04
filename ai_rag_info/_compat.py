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

from ai_rag_info._version import __version__

F = TypeVar("F", bound=Callable[..., Any])

_VERSION = re.compile(r"^\d+\.\d+(\.\d+)?$")

#: Modules that must be importable for the in-process engine (``ai_rag_info.embedded``) to work.
ENGINE_MODULES = ("elasticsearch", "pydantic_settings", "redis", "langchain_core", "prometheus_client")

#: Qualified names of every ``@experimental`` object. Filled as the modules that define them are imported:
#: ``ai_rag_info.embedded`` and ``ai_rag_info.testing`` are loaded on first use, so it is empty until then.
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


def _as_tuple(version: str) -> tuple[int, int, int]:
    parts = [int(x) for x in re.findall(r"\d+", version)[:3]]
    return (*parts, *([0] * (3 - len(parts))))  # type: ignore[return-value]


def deprecated(
    *,
    since: str,
    remove_in: str,
    alternative: str | None = None,
    reason: str | None = None,
    escalate_in: str | None = None,
) -> Callable[[F], F]:
    """Deprecate a public function or method.

    ``since`` is the version that deprecates it, ``remove_in`` the major release that removes it
    (``3.0``; removal only happens in a major release - checked here). ``alternative`` names the replacement,
    or give ``reason`` when there is none. From ``escalate_in`` on - set it to the last minor release before
    removal - the warning becomes :class:`RagFutureWarning`, which is shown to end users and not only to
    developers. The floor of *two minor releases* between ``since`` and removal cannot be verified from
    versions alone: that part stays a review rule (``docs/framework.md``, section 7).
    """
    _check_version("since", since)
    _check_version("remove_in", remove_in)
    if not (alternative or reason):
        raise ValueError("deprecated(): give the replacement (`alternative`) or the `reason` there is none")
    removal = _as_tuple(remove_in)
    if removal[1] != 0 or removal[2] != 0:
        raise ValueError(
            f"deprecated(): remove_in={remove_in!r} must be a major release (like '3.0'): removal only happens in majors"
        )
    if removal[0] <= _as_tuple(since)[0]:
        raise ValueError(f"deprecated(): removal in {remove_in} is not a later major release than {since}")
    if escalate_in is not None:
        _check_version("escalate_in", escalate_in)
        if not _as_tuple(since) <= _as_tuple(escalate_in) < removal:
            raise ValueError(
                "deprecated(): escalate_in must lie between `since` (inclusive) and `remove_in` (exclusive)"
            )

    def decorate(func: F) -> F:
        message = f"{func.__qualname__} is deprecated since {since} and will be removed in {remove_in}. " + (
            f"Use {alternative} instead." if alternative else reason or ""
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            loud = escalate_in is not None and _as_tuple(__version__) >= _as_tuple(escalate_in)
            warnings.warn(message, RagFutureWarning if loud else RagDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        wrapper.__rag_deprecated__ = {"since": since, "remove_in": remove_in}  # type: ignore[attr-defined]
        wrapper.__doc__ = f"{inspect.getdoc(func) or ''}\n\n.. deprecated:: {since}\n   {message}".strip()
        return cast(F, wrapper)

    return decorate
