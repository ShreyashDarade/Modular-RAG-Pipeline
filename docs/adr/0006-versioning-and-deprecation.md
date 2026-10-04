# ADR 0006: Versioning and deprecation policy

Status: accepted

**Context.** LangChain only pledged SemVer at 1.0 and still broke a sibling package in a patch release because a
dependency was not capped; pandas and NumPy publish numeric, checkable windows.

**Decision.** SemVer for the public surface and the wire contract; no deprecation or removal in a patch release;
a deprecated name works unchanged for at least two minor releases and is removed no earlier than the next major.
Deprecations go through `turinton_rag.deprecated(since, remove_in, alternative)` (metadata is mandatory) and
warn with `RagDeprecationWarning`, escalating to `RagFutureWarning` in the last minor. Internal packages that
ship together are pinned to the same version.

**Consequences.** Slower removals. The SDK tracks the server's deprecations through `openapi.json`.

**Enforced by.** The decorator validates its metadata; the suite runs with the warning as an error; `griffe check`.
