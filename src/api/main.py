from __future__ import annotations

import uvicorn

from src.core.config import get_settings


def run() -> None:
    """Production entry point: ``rag-api`` or ``python -m src.api.main``.

    One process per container is the scaling unit (scale with replicas). ``WEB_CONCURRENCY``
    runs several processes - then use the Redis backends so state is shared between them.

    ``X-Forwarded-For`` is trusted only from the addresses in ``FORWARDED_ALLOW_IPS`` (uvicorn's
    own variable; default: localhost). Behind a load balancer set it to the balancer's addresses -
    trusting everyone would let any client spoof its IP and sidestep the per-client rate limit.
    """
    settings = get_settings()
    uvicorn.run(
        "src.api.server:app",
        host=settings.host,
        port=settings.port,
        workers=settings.web_concurrency,
        proxy_headers=True,
        timeout_graceful_shutdown=settings.shutdown_grace_seconds,
        access_log=False,
    )


if __name__ == "__main__":
    run()
