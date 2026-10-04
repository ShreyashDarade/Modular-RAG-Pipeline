"""MCP tools over the same services the REST API uses (no parallel implementation).

All tools are read-only: ingestion and deletion are not exposed here. The server is stateless
over HTTP, so any replica can serve any request - no session affinity. Like the REST API it has no
authentication of its own: expose it only behind a gateway that provides it. Over HTTP each client
is rate limited (same limiter and limit as REST) and every tool call has the request deadline.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from src.core.errors import RagError
from src.core.types import RetrievedDocument
from src.runtime.concurrency import deadline
from src.runtime.metrics import RATE_LIMITED

if TYPE_CHECKING:
    from fastapi import FastAPI

    from src.core.config import Settings
    from src.core.container import Container

_READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


def _document(doc: RetrievedDocument) -> dict[str, Any]:
    meta = doc.metadata
    return {
        "content": doc.content,
        "score": round(doc.final_score, 6),
        "source": meta.get("source"),
        "page": meta.get("page"),
        "type": meta.get("type") or meta.get("content_type"),
        "collection": doc.collection,
        "kind": doc.kind,
    }


def _guarded[**P, R](
    get_container: Callable[[], Container],
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Every tool call gets the request deadline, and typed errors reach the MCP client as readable
    tool errors (the SDK shows only ``ToolError`` messages; any other exception becomes a generic
    "Error executing tool")."""

    def decorate(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                async with deadline(get_container().settings.request_timeout_seconds):
                    return await fn(*args, **kwargs)
            except RagError as exc:
                raise ToolError(f"{exc.code}: {exc.public_message}") from exc

        return wrapper

    return decorate


def build_mcp_server(get_container: Callable[[], Container]) -> MCPServer:
    guarded = _guarded(get_container)
    server = MCPServer(
        name="ai-rag-info",
        instructions=(
            "Search and question-answering over a private, multilingual (English/Hindi/Marathi) document "
            "corpus split into collections. Call list_collections first to see what is available; "
            "then search_documents for raw passages or ask_question for a cited answer."
        ),
    )

    @server.tool(annotations=_READ_ONLY)
    @guarded
    async def list_collections() -> list[dict[str, Any]]:
        """List the searchable collections, what each indexes and which one is the default."""
        config = get_container().config
        return [
            {
                "name": name,
                "description": spec.description,
                "kinds": list(spec.kinds),
                "embedding_model": spec.embedding_model,
                "default": name == config.default_collection,
            }
            for name, spec in sorted(config.collections.items())
        ]

    @server.tool(annotations=_READ_ONLY)
    @guarded
    async def list_models() -> dict[str, Any]:
        """List the chat models ask_question can use (pass the name as `model`)."""
        config = get_container().config
        return {
            "default": config.default_chat_model,
            "chat_models": [
                {"name": n, "provider": s.provider, "model": s.model}
                for n, s in sorted(config.chat_models.items())
            ],
        }

    @server.tool(annotations=_READ_ONLY)
    @guarded
    async def list_documents(
        collection: str | None = None, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """List the documents ingested into a collection (default collection if omitted)."""
        container = get_container()
        name = collection or container.config.default_collection
        records, total = await container.documents.list(
            name, limit=max(1, min(limit, 500)), offset=max(0, offset)
        )
        return {
            "collection": name,
            "total": total,
            "documents": [
                {
                    "source": r.source,
                    "parser": r.parser,
                    "kinds": r.kinds,
                    "pages": r.total_pages,
                    "chunks": r.text_chunks + r.table_chunks + r.image_chunks,
                }
                for r in records
            ],
        }

    @server.tool(annotations=_READ_ONLY)
    @guarded
    async def search_documents(
        query: str,
        collections: list[str] | None = None,
        kinds: list[str] | None = None,
        limit: int = 6,
    ) -> dict[str, Any]:
        """Hybrid (keyword + semantic) search. Returns the best passages with their source and page.

        collections: restrict to these collections (default collection if omitted).
        kinds: restrict to any of "text", "table", "image" (OCR'd images).
        """
        container = get_container()
        scope = container.retrieval.scope(collections, kinds)
        result = await container.retrieval.retrieve(query, scope)
        return {
            "query": result.query,
            "expanded_queries": result.expanded_queries,
            "documents": [_document(d) for d in result.documents[: max(1, limit)]],
        }

    @server.tool(annotations=_READ_ONLY)
    @guarded
    async def ask_question(
        question: str,
        collections: list[str] | None = None,
        kinds: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Answer a question using only retrieved context, with [Source, Page] citations.

        model: a chat model name from list_models (default model if omitted).
        """
        container = get_container()
        scope = container.retrieval.scope(collections, kinds)
        result = await container.answers.ask(question, scope, model)
        return {
            "answer": result.answer,
            "model": result.model,
            "sources": [_document(d) for d in result.documents],
        }

    return server


def mount_mcp(app: FastAPI, settings: Settings) -> None:
    """Serve MCP at ``settings.mcp_path`` (stateless streamable HTTP) inside the FastAPI app."""
    server = build_mcp_server(lambda: app.state.container)
    security = (
        TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=settings.mcp_allowed_hosts,
            allowed_origins=settings.mcp_allowed_origins,
        )
        if settings.mcp_allowed_hosts
        else None
    )
    mcp_app = server.streamable_http_app(
        streamable_http_path=settings.mcp_path,
        stateless_http=True,
        json_response=True,
        transport_security=security,
    )
    app.mount("/", _RateLimited(mcp_app, app, settings))

    def lifespan() -> AbstractAsyncContextManager[Any]:
        return server.session_manager.run()

    app.state.mcp_lifespan = lifespan


class _RateLimited:
    """Per-client rate limit in front of the MCP app, using the container's limiter (the same one,
    and the same limit, as the REST endpoints). Stdio has no client address and is not limited."""

    def __init__(self, inner: Any, app: FastAPI, settings: Settings) -> None:
        self._inner = inner
        self._app = app
        self._path = settings.mcp_path

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        container = getattr(self._app.state, "container", None)
        limiter = container.rate_limiter if container is not None else None
        if (
            container is not None
            and scope["type"] == "http"
            and limiter is not None
            and scope["path"].startswith(self._path)
        ):
            client = scope["client"][0] if scope.get("client") else "unknown"
            limit = container.settings.rate_limit_per_minute
            decision = await limiter.hit(f"{client}|mcp", limit, 60)
            if not decision.allowed:
                RATE_LIMITED.labels("mcp").inc()
                body = json.dumps(
                    {"detail": f"rate limit of {limit}/minute exceeded", "code": "rate_limited"}
                ).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 429,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"retry-after", str(decision.retry_after).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self._inner(scope, receive, send)
