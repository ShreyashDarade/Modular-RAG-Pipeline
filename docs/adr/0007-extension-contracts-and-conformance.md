# ADR 0007: Extension contracts and conformance checks

Status: accepted

**Context.** Components are registered by name and may come from third-party plug-ins, loaded by dotted path.
Static import analysis cannot see that, so layering checks alone do not guard it.

**Decision.** Each port has a conformance check in `turinton_rag.testing` (plain functions, no pytest
dependency) that states what any implementation must do - for example an embedder returns vectors of its
declared size and is independent of batching; a reranker sets a score on every candidate and returns the same
candidates; a cache returns what was stored. Built-in components and the fake test plug-in must pass them in CI;
plug-in authors run the same functions.

**Consequences.** `turinton_rag.testing` is experimental. A contract the checks do not express remains a
convention.

**Enforced by.** `tests/conformance/`.
