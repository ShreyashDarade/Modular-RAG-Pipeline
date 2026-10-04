"""Typed error hierarchy.

The pipeline never substitutes a degraded behaviour for a failed one: an upstream outage, a
missing optional dependency or an unknown component name surfaces as one of these errors.
Adapters translate third-party exceptions into this hierarchy (keeping the original as
``__cause__``) so callers - and the HTTP layer - can react to *what* went wrong without
knowing *which library* raised it.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class RagError(Exception):
    """Base class. ``status_code`` / ``code`` are hints for the HTTP and MCP layers.

    The ``code`` strings are part of the public contract: they never change meaning, and the SDK maps
    the same code to the same class whether the engine ran in-process or behind HTTP.
    """

    status_code: int = 500
    code: str = "internal_error"
    #: Set on errors rebuilt from an HTTP response (``None`` for errors raised in-process).
    request_id: str | None = None
    #: Extra fields of the error response (for example ``job_id`` and ``status_url`` of a failed ingest job).
    details: Mapping[str, Any] = MappingProxyType({})

    @property
    def public_message(self) -> str:
        """Text that is safe to show callers. Messages of caller-facing errors are written for
        them; errors that wrap a dependency's own text override this (see ``UpstreamError``)."""
        return str(self)


# --- configuration (raised at start-up, never per request) ---------------------------------
class ConfigError(RagError):
    code = "config_error"


class UnknownComponentError(ConfigError):
    """A registry was asked for a name nobody registered."""

    code = "unknown_component"


class ProviderUnavailableError(ConfigError):
    """The named provider/parser needs an optional package that is not installed."""

    code = "provider_unavailable"


# --- caller mistakes ------------------------------------------------------------------------
class InvalidRequestError(RagError):
    status_code = 400
    code = "invalid_request"


class NotFoundError(RagError):
    status_code = 404
    code = "not_found"


class UnsupportedTypeError(InvalidRequestError):
    status_code = 415
    code = "unsupported_type"


class PayloadTooLargeError(InvalidRequestError):
    status_code = 413
    code = "payload_too_large"


class RateLimitedError(RagError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, message: str, retry_after: int = 1) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# --- capacity -------------------------------------------------------------------------------
class OverloadedError(RagError):
    status_code = 503
    code = "overloaded"

    def __init__(self, message: str, retry_after: int = 1) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class QueueFullError(OverloadedError):
    code = "queue_full"


class RequestTimeoutError(RagError):
    status_code = 504
    code = "timeout"


# --- processing -----------------------------------------------------------------------------
class ParseError(RagError):
    status_code = 422
    code = "parse_error"


class JobFailedError(RagError):
    code = "job_failed"


# --- dependencies (Elasticsearch, Redis, model providers) -----------------------------------
class UpstreamError(RagError):
    status_code = 502
    code = "upstream_error"

    @property
    def public_message(self) -> str:
        # str(self) embeds the dependency's own exception text (which can contain request
        # details or fragments of credentials); that stays in the server log.
        return "a backing service failed; details are in the server log"


class IndexingError(UpstreamError):
    code = "indexing_error"


class SearchError(UpstreamError):
    code = "search_error"


class ModelError(UpstreamError):
    code = "model_error"


# --- client side (raised by the SDK, never sent by the server) ---------------------------------------
class ClientError(RagError):
    """A failure on the calling side of the SDK. ``status_code`` is 0: no HTTP status is involved."""

    status_code = 0
    code = "client_error"


class ConnectionFailedError(ClientError):
    """The server could not be reached (or the connection broke) after the configured retries."""

    code = "connection_failed"


class ResponseError(ClientError):
    """The server answered, but not with the documented shape. Never accepted as a best-effort object."""

    code = "invalid_response"


class UsageError(ClientError):
    """The SDK was used wrongly (for example a blocking call from inside a running event loop)."""

    code = "usage_error"


# --- the code catalog -------------------------------------------------------------------------------
def _walk(cls: type[RagError]) -> list[type[RagError]]:
    found = [cls]
    for sub in cls.__subclasses__():
        found.extend(_walk(sub))
    return found


def error_catalog() -> dict[str, type[RagError]]:
    """``code -> class`` for every error defined here. Codes are unique; a duplicate is a bug."""
    catalog: dict[str, type[RagError]] = {}
    for cls in _walk(RagError):
        if cls.code in catalog and catalog[cls.code] is not cls:
            raise RuntimeError(f"error code '{cls.code}' is used by both {catalog[cls.code]} and {cls}")
        catalog[cls.code] = cls
    return catalog


def error_from_code(
    code: str,
    message: str,
    *,
    status: int | None = None,
    retry_after: int | None = None,
    request_id: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> RagError:
    """Rebuild a typed error from its wire form (``code`` + public message).

    A known code becomes its class, so ``except NotFoundError`` works the same in-process and over HTTP.
    An unknown code (a newer server) becomes a plain :class:`RagError` that keeps the code and status, so
    nothing is lost and nothing is mistaken for a different error.
    """
    cls = error_catalog().get(code)
    error: RagError = cls(message) if cls is not None else RagError(message)
    if cls is None:
        error.code = code
    if status is not None:
        error.status_code = status
    if retry_after is not None and hasattr(error, "retry_after"):
        error.retry_after = retry_after
    error.request_id = request_id
    if details:
        error.details = MappingProxyType(dict(details))
    return error
