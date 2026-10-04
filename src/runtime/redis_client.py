from __future__ import annotations

import redis.asyncio as aioredis

DEFAULT_SOCKET_TIMEOUT = 5.0


def new_client(
    url: str, *, decode_responses: bool, socket_timeout: float = DEFAULT_SOCKET_TIMEOUT
) -> aioredis.Redis:
    """One place that builds Redis clients, so every one gets connect/read timeouts: a stalled Redis
    must fail a request quickly (as an ``UpstreamError``), not hang it until the proxy gives up."""
    return aioredis.from_url(
        url,
        decode_responses=decode_responses,
        health_check_interval=30,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_timeout,
    )
