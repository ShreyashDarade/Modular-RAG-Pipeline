# ADR 0005: One SDK with an embedded and an HTTP transport

Status: accepted

**Context.** One package must serve both applications that embed the pipeline and applications that call the
REST API. The review found no verified source on dual-mode SDKs; unverified leads (qdrant-client's local mode and
shared parity suite; a LanceDB issue where local and remote built different query objects from the same call)
point the same way: share the interface and models, define defaults once, and test both modes with the same
suite.

**Decision.** One public facade over a narrow internal `Backend` protocol; two backends (HTTP via httpx,
embedded via `RagService`). Async is the implementation, sync is a bridge over one background event loop.
Capability differences are explicit (`evaluate` exists only embedded). The thin client depends on `httpx` and
`pydantic` only; the engine is an optional extra. Retries cover connection errors and 408/429/502/503/504 for
idempotent operations (reads, ingest - which is idempotent by content checksum) and only connection-establishment
errors for chat turns. Unknown response fields are ignored; malformed required fields are a `ResponseError`.

**Consequences.** A background thread per sync client (documented; use the async classes inside an event loop).
Defaults live in the facade, once. The extras split changes the meaning of a bare `pip install ai-rag-info`.

**Enforced by.** `tests/sdk/` parity suite; thin-client import contract; a packaging test in a clean venv.
