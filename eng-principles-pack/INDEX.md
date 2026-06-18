# Agent Skills Index

Load the smallest sufficient set. Hooks are lifecycle gates; skills are deeper references.

## Skill routing

| Trigger | Load |
|---|---|
| architecture, module boundaries, surprising behavior, patterns | `skills/design-review.md` |
| exceptions, validation, error propagation, external inputs | `skills/error-handling.md` |
| public API, function signatures, data model, idempotency | `skills/api-design.md` |
| tests, test design, regression coverage, property tests | `skills/testing-strategy.md` |
| scientific/math code, floating point, reproducibility, memory layout | `skills/numerical-code.md` |
| bug diagnosis, logs, observability, production failure | `skills/debugging.md` |
| AI-written code, AI review, AI-generated tests, AI debugging | `skills/ai-code-verification.md` |
| commits, CI, code review, technical debt, performance, reliability | `skills/refactoring.md` |

## Hook routing

| Lifecycle moment | Run |
|---|---|
| before commit | `hooks/pre-commit-checklist.md` |
| before PR | `hooks/pre-pr-checklist.md` |
| after a bug fix | `hooks/post-bugfix.md` |
| before merging numerical/scientific code | `hooks/pre-merge-numerical.md` |

## Loading policy

Prefer `INDEX.md` first. Load a hook for the lifecycle moment, then at most one or two skills unless the task genuinely crosses boundaries. Do not load the entire pack by default.
