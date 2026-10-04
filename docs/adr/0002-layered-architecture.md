# ADR 0002: Layered architecture, third-party confinement

Status: accepted

**Context.** The design is ports-and-adapters. Measured with Import Linter on the existing code, a strict layers
contract (api/cli/worker > mcp/evaluation > container > bootstrap > chat > retrieval/ingestion >
adapters > runtime > registry > ports > config > specs/kinds > types/errors) is *already satisfied*, and each
heavy third-party library is imported from exactly one layer.

**Decision.** Encode that as contracts: (1) a layers contract; (2) forbidden-import contracts confining
`elasticsearch`, `redis`, `langchain*`, `torch`/OCR/PDF libraries, `fastapi`, `typer` to their layers;
(3) a thin-client contract: the SDK's HTTP client may not import the engine or any of those libraries, directly
or indirectly. Every new top-level package must be assigned a layer (classification test).

**Consequences.** Import Linter's analysis is static: factories registered by dotted name or entry point are
invisible to it (see ADR 0007). Its exhaustive mode only checks direct children of a container, so the
classification test covers the gap. Type-checking-only imports count as dependencies (they are coupling).

**Enforced by.** `[tool.importlinter]` in `pyproject.toml`; `tests/architecture/`; CI `lint`.
