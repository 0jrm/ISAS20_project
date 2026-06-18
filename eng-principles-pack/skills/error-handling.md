# Error Handling

Source: Fail Fast, Explicit Error Types, Don't Swallow Exceptions, Defensive Programming in Boundaries.

Error Handling
Fail Fast, Fail Loudly
Detect errors at the earliest possible point and surface them clearly.
Silent failures that propagate through a system are the hardest bugs to diagnose. Validate inputs at boundaries. Assert invariants in computational code. Prefer exceptions or explicit error types over sentinel return values (returning -1 or null to signal error). In scientific pipelines, an incorrect result that silently propagates is far worse than a crash.
Explicit Error Types
Errors should carry enough information to be acted upon.
Generic errors like 'something went wrong' are useless. Error types should encode what failed, why, and ideally how to fix it. Use typed exceptions or result types (Result<T, E>) over raw strings. Distinguish between operational errors (expected: network timeout, invalid input) and programmer errors (unexpected: assertion violations). Handle each category differently.
Don't Swallow Exceptions
Every caught exception must be handled or re-raised. Empty catch blocks are bugs.
An empty `except:` block silently discards information. At minimum, log the exception with full context before continuing. Prefer letting exceptions propagate to a centralized error handler over scattered, inconsistent local handling. If you truly can't handle an error, let it crash — a visible crash is better than corrupted state.
Defensive Programming in Boundaries
Validate all data crossing system boundaries — files, APIs, user input, databases.
Never trust data from outside your module. Validate types, ranges, formats, and invariants at ingestion points. This is especially critical in scientific code where malformed data (wrong units, out-of-range sensor readings) can produce plausible-looking but wrong results. Consider using schema validation libraries at all external interfaces.
