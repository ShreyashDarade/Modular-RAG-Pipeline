"""Typed error hierarchy.

The pipeline never substitutes a degraded behaviour for a failed one: an upstream outage, a
missing optional dependency or an unknown component name surfaces as one of these errors.
Adapters translate third-party exceptions into this hierarchy (keeping the original as
``__cause__``) so callers - and the HTTP layer - can react to *what* went wrong without
knowing *which library* raised it.
"""

from __future__ import annotations


class RagError(Exception):
    """Base class. ``status_code`` / ``code`` are hints for the HTTP and MCP layers."""

    status_code: int = 500
    code: str = "internal_error"

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
