# ADR 0006: Versioning and deprecation policy

Status: accepted

**Context.** LangChain only pledged SemVer at 1.0 and still broke a sibling package in a patch release because a
dependency was not capped; pandas and NumPy publish numeric, checkable windows.

**Decision.** SemVer for the public surface and the wire contract; no deprecation or removal in a patch release;
a deprecated name works unchanged for at least two minor releases and is removed no earlier than the next major.
Deprecations go through `ai_rag_info.deprecated(since, remove_in, alternative, escalate_in)` (metadata is
mandatory) and warn with `RagDeprecationWarning`; from the `escalate_in` version - the last minor before removal -
they warn with `RagFutureWarning`. Internal packages that ship together are pinned to the same version.

**Consequences.** Slower removals. The SDK tracks the server's deprecations through `openapi.json`.

**Enforced by.** The decorator validates its metadata and that removal is in a later major; the suite runs with
the warning as an error; `griffe check`. Not machine-enforced: the two-minor-release floor and remembering to set
`escalate_in` - those are review rules.
