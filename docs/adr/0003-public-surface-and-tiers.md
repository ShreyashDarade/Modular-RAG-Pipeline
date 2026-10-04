# ADR 0003: Public API surface, stability tiers, the `src` namespace

Status: accepted

**Context.** Users and plug-in authors currently import `src.*` modules. A promise of stability needs a small,
declared surface; LangChain's tiers and the Python typing guidance both define public by declaration and by
underscore-private modules, and note that type-completeness is judged on the public interface only.

**Decision.** The public surface is `ai_rag_info.__all__` plus the documented submodules `models`, `errors`,
`extend`, `testing`. Everything under `src.*` is internal. Experimental names carry `@experimental`
(greppable, listed in `ai_rag_info.EXPERIMENTAL`). The package is typed (`py.typed`, strict mypy settings on `ai_rag_info.*`).
The engine keeps the namespace `src` for now: renaming it would break the documented 1.x entry points
(`uvicorn src.api.server:app`, `python -m src.cli`); it is declared internal and the rename is scheduled for the
next major.

**Consequences.** Public classes keep their defining module (for example `src.core.errors.NotFoundError`) and
are *re-exported*; their `__module__` is not rewritten. Plug-in authors move from `src.ports.*` to
`ai_rag_info.extend` (the old paths keep working but carry no promise).

**Enforced by.** API-surface snapshot, `griffe check` against the last tag, strict mypy settings, the `__all__` test.
