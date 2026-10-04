# Framework design

This is the contract the code is built to. It is written **before** the code that implements it and is
enforced by checks, not by goodwill: where a rule below says *enforced by*, a CI job fails when it is broken.
Decisions and their reasons are in [`docs/adr/`](adr/); this page is the map.

> Evidence base: a verified literature review (LangChain / LlamaIndex / Haystack release policies, Import
> Linter, Griffe, the Python typing guide, pandas PDEP-17, NumPy NEP 23, openai-python, Stripe) plus first-party
> measurement in this repo. Where a design choice rests on an *inference* rather than a verified source, the ADR
> says so. The review was thin on dual-mode (in-process + HTTP) SDKs, so those decisions are validated here by
> tests (`tests/sdk/`), not assumed.

## 1. What the framework is

A RAG backend is a pipeline of replaceable parts around a stable core. The framework fixes **four things** and
leaves everything else free to change:

| Fixed (the contract) | Free to change (the implementation) |
|---|---|
| the **ports** - small `Protocol`s every part implements | which Elasticsearch/Redis/LLM/OCR library sits behind a port |
| the **wire contract** - the `/api/v1` REST shapes and error codes | how a route or a service computes its answer |
| the **public SDK surface** - what `ai_rag_info` exports | everything under `src.*` |
| the **layering rule** - who may import whom | file layout inside a layer |

New capability enters as: (1) a port or contract change, reviewed as such (an ADR if it is not additive),
then (2) an adapter behind it, then (3) conformance tests. Code that skips step 1 fails the layering or
public-surface checks.

## 2. Layers and the dependency rule

Higher layers may import lower layers, never the reverse. Siblings on one row do not import each other.

```
 ai_rag_info.embedded (SDK, in-process)  |  ai_rag_info.client (SDK, HTTP)        <- public
 -----------------------------------------------------------------------------
 api | cli | worker                           driving adapters (HTTP, terminal, queue worker)
 mcp_server | evaluation                      driving adapters / offline tools
 application                                  use cases: one RagService - the only place a use case lives
 core.container                               composition root: builds everything, once
 core.bootstrap                               registers built-in components + plug-ins
 chat
 retrieval | ingestion                        services
 indexing | jobs | chunking | parsing | models   driven adapters (Elasticsearch, Redis, parsers, providers)
 runtime                                      cache, concurrency, rate limit, metrics
 core.registry                                name -> factory registries (the extension mechanism)
 ports                                        the Protocols
 core.config
 core.specs | core.kinds
 core.types | core.errors | core.logger       domain values and typed errors (stdlib only)
 contracts                                    wire models (pydantic only)  <- shared by api and the SDK
```

* **Dependencies point inward.** A port knows nothing about Elasticsearch; the Elasticsearch adapter knows
  the port. The composition root is the only place that names concrete adapters.
* **Third-party libraries are confined to one layer** - `elasticsearch` in `indexing`, `langchain*` in `models`
  and `chunking`, `torch`/OCR/PDF libraries in `parsing`, `fastapi` in `api` and `mcp_server`, `typer` in `cli`.
  Swapping a library touches one layer.
* **The thin client stays thin.** The SDK's HTTP client may import `contracts`, `core.errors` and `httpx` - and
  nothing that pulls in the engine.

*Enforced by:* Import Linter contracts in `pyproject.toml` (layers, third-party confinement, thin-client purity),
run in CI and in `tests/architecture/`. A test also fails if a new top-level package is not assigned a layer.

## 3. The ports (extension points)

Every extension point is a `Protocol` in `src/ports/`, a name -> factory `Registry`, and a conformance suite.

| Port | What a plug-in provides | Registry |
|---|---|---|
| `Parser` | a file type -> text/table/image units | `parsers` |
| `Chunker` | units -> chunks | `chunkers` |
| `OcrEngine` | image -> text | `ocr_engines` |
| `Embedder` | text -> vectors | `embedding_providers` |
| `ChatModel` | messages -> text / stream | `chat_providers` |
| `Reranker` | candidates + query -> ordered candidates | `rerankers` |
| `QueryExpander` | query -> variants | `query_expanders` |
| `Cache`, `RateLimiter`, `ConversationStore`, `JobBackend` | runtime services | `caches`, `rate_limiters`, `conversation_stores`, `job_backends` |

A **plug-in** is a module with `register(registries)`, enabled by `PLUGINS=["pkg.module"]`. Registration is by
*name*; asking for an unregistered name is an `UnknownComponentError` that lists the valid names. There is no
default and no fallback.

*Enforced by:* conformance checks (`ai_rag_info.testing`) that every built-in component passes in CI and that
plug-in authors can run on theirs. Dynamic loading is invisible to the import graph, so conformance - not the
linter - is what guards it.

## 4. The wire contract and the use-case layer

* **`src/contracts`** holds the request/response models of the REST API (pydantic, nothing else). The HTTP
  routes and the SDK both import them - there is one definition of every shape.
* **`src/application.RagService`** is the only place a use case (retrieve, ask, chat, ingest, list/delete
  documents, jobs, catalog) is implemented. HTTP routes, the embedded SDK and (later) the CLI/MCP server are
  *driving adapters*: they translate their input into a `RagService` call and its output into their format.
  They contain no retrieval, ranking or persistence logic.
* **The OpenAPI document is a build artefact that is checked in** (`docs/openapi.json`). A test regenerates it
  from the app and fails on any difference, so a change to the wire contract is always a visible diff.
* Within `/api/v1`, changes are **additive only** (new optional request fields, new response fields, new
  endpoints). Removing or re-typing a field requires `/api/v2`. The snapshot makes every change *visible*; that a
  change is additive is judged in review - there is no automatic breaking-change detector for the wire contract yet.
* **Error codes are part of the contract.** `RagError.code` strings (`not_found`, `rate_limited`, ...) never
  change meaning; the SDK maps the same code to the same exception class in both transports.

*Enforced by:* the OpenAPI snapshot test, the layering contracts (routes cannot reach adapters), and the SDK
parity suite.

## 5. One SDK, two transports

`pip install ai-rag-info` gives a **thin client**; extras add the engine (see the README for the extras map).

```python
from ai_rag_info import AsyncRagClient, RagClient      # HTTP, thin: httpx + pydantic only
from ai_rag_info import AsyncRag, Rag                  # in-process engine (needs ai-rag-info[engine])
```

* **One facade, written once** (`documents`, `chat`, `jobs`, `collections`, `retrieve`, `ask`) over a narrow
  internal `Backend` protocol. The HTTP backend and the embedded backend implement the protocol; the public
  interface cannot differ between them because it is the same code.
* **Same models, same errors.** Both transports return `src.contracts` models and raise `src.core.errors`
  classes. A `404 not_found` over HTTP and a `NotFoundError` in-process are the same exception.
* **Capability differences are explicit, never stubbed.** `evaluate()` exists only on the embedded classes; the
  HTTP client has no such attribute (a type checker says so) rather than a method that fails at runtime.
* **Async is the primary implementation; sync is a thin bridge** (`Rag`, `RagClient`) over one background event
  loop, so there is a single code path to test. Calling a sync method from inside a running event loop is a
  typed `UsageError`, not a silent block.
* **Forward compatibility without silent loss.** Responses may gain fields: older SDKs ignore unknown fields
  (documented) but validate required ones strictly - a malformed response is a `ResponseError`, never a
  best-effort object.

*Enforced by:* the parity suite (`tests/sdk/`): the same test body runs against the embedded engine and the
HTTP client (through the real FastAPI app), and one test points both at the *same* engine and compares outputs.

## 6. Public API surface and stability tiers

* **Public** = names exported in `ai_rag_info.__all__` (and its documented submodules `models`, `errors`,
  `extend`, `testing`). Everything under `src.*` and every underscore-prefixed name is **internal** with no
  compatibility promise. (`src` is the engine's historical top-level name; renaming it is tracked in ADR 0003.)
* **Tiers.** *Stable* is the default for public names. *Experimental* names are marked `@experimental` (greppable;
  listed in `ai_rag_info.EXPERIMENTAL`) and may change in any minor release: today `evaluate`, `testing`.
* **Typed.** The package ships `py.typed`; the public interface is fully annotated and checked with strict mypy
  settings (`[[tool.mypy.overrides]]` for `ai_rag_info.*`: no untyped defs, no implicit re-exports, no `Any` returns).
  A private type never appears in a public signature.

*Enforced by:* an API-surface snapshot (`tests/sdk/api_surface.txt`, regenerated explicitly) and `griffe check`
against the latest release tag in CI; strict mypy settings on `ai_rag_info`; a test that every `__all__` name
resolves, is documented and is in the snapshot.

## 7. Versioning and deprecation

* **SemVer for the public surface and the wire contract.** Breaking changes only in a major release; patch
  releases never deprecate or remove.
* **Deprecation window:** a public name stays working, unchanged, for **at least two minor releases** after it is
  deprecated, and is removed no earlier than the next major (the floor from pandas PDEP-17 and NumPy NEP 23).
  People upgrade slowly; when in doubt the window is longer.
* **Every deprecation carries metadata** - the version it was deprecated in, the version it will be removed in,
  the replacement (or the reason there is none) - through one decorator, `ai_rag_info.deprecated(...)`. It
  raises `RagDeprecationWarning` (a `DeprecationWarning` subclass) with a correct `stacklevel`. The author sets
  `escalate_in` to the last minor before removal; from that version on it warns with `RagFutureWarning`, which
  is visible outside `__main__`.
* **What the decorator can and cannot check.** It refuses malformed versions, removal anywhere but a later
  *major* release, and an `escalate_in` outside `[since, remove_in)`. The two-minor-release *floor* cannot be
  derived from version numbers alone (it depends on what is released in between), so it is a review rule, and
  `escalate_in` is something the author must remember to set; neither is machine-enforced.
* **The wire contract has its own window:** a deprecated endpoint or field is announced in the OpenAPI document
  (`deprecated: true`) and the release notes for at least two minors before `/api/v2` drops it.
* The test suite runs with `-W error::ai_rag_info.RagDeprecationWarning`: nothing in this repo may use a name
  it has deprecated.

## 8. Governance: what runs where

| Rule | Check | Where |
|---|---|---|
| layering; third-party confinement; thin-client purity | Import Linter | CI `lint`, `tests/architecture/` |
| every package has a layer | classification test | `tests/architecture/` |
| wire contract changes are visible | OpenAPI snapshot | `tests/contract/` |
| public surface changes are visible | API-surface snapshot; `griffe check` vs last tag | `tests/sdk/`, CI `api-compat` |
| public interface is typed | strict mypy settings on `ai_rag_info`; `py.typed` in the wheel | CI `lint`, packaging test |
| embedded and HTTP transports agree | parity suite | `tests/sdk/` |
| plug-in contracts hold | conformance checks | `tests/conformance/` |
| deprecations are well-formed and unused | decorator validation; `-W error` | unit tests |
| a design decision has a record | ADR for any change to a row of section 1's "fixed" column | review |

## 9. Changing the framework

A change to a *fixed* item (section 1) starts with an ADR in `docs/adr/` - context, decision, consequences, and
"enforced by" - and updates this page in the same change. Additive changes (a new optional field, a new port
implementation) need no ADR but do need the snapshot updates that make them visible. A change that edits an
implementation but not the contract needs neither.

## 10. Known debt (stated, not hidden)

* The engine's import namespace is the generic `src`. It is internal by declaration; renaming it is a breaking
  change to the documented 1.x entry points (`uvicorn src.api.server:app`) and is tracked in ADR 0003.
* The CLI and the MCP server predate the application layer and still call the container directly. New use
  cases must go through `RagService`; migrating these two is incremental (ADR 0004).
* Plug-in loading by dotted name is invisible to static import analysis (ADR 0007).
