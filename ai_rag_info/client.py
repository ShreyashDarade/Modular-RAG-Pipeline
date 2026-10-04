"""The remote SDK: talk to a running RAG server over HTTP.

Needs only ``httpx`` and ``pydantic`` - nothing of the engine is imported (enforced by an import contract).
"""

from __future__ import annotations

from collections.abc import Mapping

import httpx

from ai_rag_info._facade import AsyncRagAPI
from ai_rag_info._http import HttpBackend
from ai_rag_info._sync import RagAPI, bridge_for
from ai_rag_info._version import __version__

#: Longer than the server's own request deadline (60 s by default), so a slow request ends with the server's
#: typed ``timeout`` error rather than a client-side transport timeout.
DEFAULT_TIMEOUT = 90.0
DEFAULT_MAX_RETRIES = 2


def _build_http_client(
    base_url: str,
    api_key: str | None,
    headers: Mapping[str, str] | None,
    auth: httpx.Auth | None,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None,
) -> httpx.AsyncClient:
    merged = {"User-Agent": f"ai-rag-info-python/{__version__}", **(headers or {})}
    if api_key:
        merged["Authorization"] = f"Bearer {api_key}"
    return httpx.AsyncClient(
        headers=merged,
        auth=auth,
        timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
        transport=transport,
    )


class AsyncRagClient(AsyncRagAPI):
    """Async client for the REST API.

    ``api_key`` is sent as ``Authorization: Bearer ...``; ``headers`` and ``auth`` (any ``httpx.Auth``) cover
    other gateways. ``max_retries`` bounds retries of transient failures (see ``docs/adr/0005``).
    Pass ``http_client`` to supply your own ``httpx.AsyncClient`` (proxies, TLS, tests); it is then yours to close.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        auth: httpx.Auth | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        client = http_client or _build_http_client(base_url, api_key, headers, auth, timeout, None)
        super().__init__(
            HttpBackend(client, base_url, max_retries=max_retries, owns_client=http_client is None)
        )


class RagClient(RagAPI):
    """Blocking client for the REST API: the same interface as :class:`AsyncRagClient`.

    It runs the async client on one background event loop, so there is a single code path. Do not call it
    from inside a running event loop (a ``UsageError`` says so) - use :class:`AsyncRagClient` there.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        auth: httpx.Auth | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        bridge = bridge_for("RagClient")

        async def make() -> AsyncRagClient:
            return AsyncRagClient(
                base_url,
                api_key=api_key,
                headers=headers,
                auth=auth,
                timeout=timeout,
                max_retries=max_retries,
            )

        super().__init__(bridge.run(make()), bridge)


__all__ = ["AsyncRagClient", "RagClient"]
