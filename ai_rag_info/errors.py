"""The typed errors of the SDK - the same classes whether the engine runs in-process or behind HTTP.

Every failure derives from :class:`RagError` and carries a stable ``code`` (``not_found``, ``rate_limited``,
...). Over HTTP the SDK rebuilds the class from the error response's ``code``, so ``except NotFoundError``
means the same thing in both modes. Errors rebuilt from a response also carry ``request_id`` and ``details``
(for example the ``job_id`` of a failed ingestion). Nothing here is ever raised as a silent fallback.
"""

from src.core.errors import (
    ClientError,
    ConfigError,
    ConnectionFailedError,
    IndexingError,
    InvalidRequestError,
    JobFailedError,
    ModelError,
    NotFoundError,
    OverloadedError,
    ParseError,
    PayloadTooLargeError,
    ProviderUnavailableError,
    QueueFullError,
    RagError,
    RateLimitedError,
    RequestTimeoutError,
    ResponseError,
    SearchError,
    UnknownComponentError,
    UnsupportedTypeError,
    UpstreamError,
    UsageError,
    error_catalog,
    error_from_code,
)

__all__ = [
    "ClientError",
    "ConfigError",
    "ConnectionFailedError",
    "IndexingError",
    "InvalidRequestError",
    "JobFailedError",
    "ModelError",
    "NotFoundError",
    "OverloadedError",
    "ParseError",
    "PayloadTooLargeError",
    "ProviderUnavailableError",
    "QueueFullError",
    "RagError",
    "RateLimitedError",
    "RequestTimeoutError",
    "ResponseError",
    "SearchError",
    "UnknownComponentError",
    "UnsupportedTypeError",
    "UpstreamError",
    "UsageError",
    "error_catalog",
    "error_from_code",
]
