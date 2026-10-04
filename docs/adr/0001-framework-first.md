# ADR 0001: Framework first: contracts before code, enforced mechanically

Status: accepted

**Context.** The code base grew by adding features where they were convenient. Rules that exist only as prose
(guidelines, README sections, review comments) erode: the review of this repo's own history found a bare
`.gitignore` rule that silently dropped a source package from three commits, and layering that was only true by
accident. The literature agrees that stated architecture without an automated check drifts, and that mature
LLM frameworks made stability promises only late, after repeated breaking releases.

**Decision.** Fix the contract first (`docs/framework.md`), record each decision as an ADR, and give every rule an
automated check that fails the build (section 8 of the framework). New capability starts as a change to a port,
a contract or a snapshot, then an adapter, then conformance tests - never as code that bypasses them.

**Consequences.** More up-front writing; changes to contracts are slower and visible in diffs. A rule without a
check is a convention and is labelled as one. Debt is listed (framework section 10), not hidden.

**Enforced by.** CI jobs listed in framework section 8.
