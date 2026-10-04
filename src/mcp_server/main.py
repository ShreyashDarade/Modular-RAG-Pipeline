from __future__ import annotations

import asyncio

from src.core.config import get_settings
from src.core.container import Container
from src.core.logger import configure_logging
from src.mcp_server.server import build_mcp_server


async def _serve() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    container = await Container.build(settings, role="mcp")
    try:
        await container.start()
        await build_mcp_server(lambda: container).run_stdio_async()
    finally:
        await container.close()


def main() -> None:
    """``rag-mcp``: serve the tools over stdio (for desktop MCP clients)."""
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
