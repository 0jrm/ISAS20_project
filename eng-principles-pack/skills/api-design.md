# API & Data Design

Source: API Design Principles + Data Modeling (Make Illegal States Unrepresentable, Prefer Immutability, Normalize/Denormalize Deliberately).

API Design Principles
APIs Are Forever
Public interfaces are commitments. Design them as if you can't change them.
Once published, changing an API breaks callers. Invest heavily in API design upfront. Prefer conservative initial designs — it's always possible to add, rarely possible to remove. Version APIs from day one. Deprecation cycles take time; breaking changes have real costs downstream. In scientific libraries, API stability is critical because papers cite specific versions.
Make the Right Thing Easy
The 'pit of success' — design APIs where correct usage is the natural path.
If the correct way to use your API requires reading documentation, the API is wrong. Unsafe operations should be explicitly named (`unsafe_`, `force_`, `raw_`). Required parameters belong in constructors, not optional setters. Thread-safe operations should be the default. The API should make it hard to accidentally do the wrong thing.
Consistency Over Cleverness
Users learn patterns once. Consistent APIs reduce cognitive load across calls.
Use the same parameter ordering, naming conventions, and return types across similar functions. If `sort(collection, key)` and `filter(collection, predicate)` exist, `map(collection, transform)` should follow the same pattern. Inconsistencies force users to re-read docs for every function. Follow language idioms and ecosystem conventions unless there's a compelling reason not to.
Narrow Interfaces
Expose only what's necessary. Every public symbol is a maintenance obligation.
The interface segregation principle at the API level: don't force callers to depend on things they don't use. Small, focused interfaces are easier to implement, test, and evolve. Keep implementation details private. Mark things as internal/private aggressively — it's easier to make something public later than to retract it.
Idempotency and Safety
HTTP GET should never mutate. Idempotent operations can be safely retried.
Distinguish read operations (safe: no side effects) from mutations. Make mutations idempotent where possible — calling the same operation twice produces the same result as calling it once. This property is essential in distributed systems and retry-heavy environments. Include idempotency keys for non-idempotent operations that must not be duplicated (e.g., payments, job submissions).
Data Modeling
Make Illegal States Unrepresentable
Use the type system to enforce domain invariants at compile time.
If a temperature cannot be negative in your domain, use a `NonNegativeFloat` type, not a raw float with runtime checks scattered throughout. If a user must be either authenticated or anonymous, model it as a union type, not two nullable fields. The more the type system enforces correctness, the fewer runtime errors are possible.
Prefer Immutability
Immutable data structures eliminate an entire class of bugs related to shared state.
Mutable shared state is the source of most concurrency bugs and many single-threaded ones. Functions that don't mutate their inputs are referentially transparent — easier to test, reason about, and parallelize. Use immutable data structures by default; opt into mutability explicitly and locally. In numerical computing, in-place mutations may be necessary for performance but should be clearly marked.
Normalize or Denormalize Deliberately
Choose data redundancy based on read/write patterns, not habit.
Normalized data (no redundancy) is the correct default for relational data — updates propagate automatically, integrity is enforced. Denormalized data (controlled redundancy) is correct when read performance dominates — precomputed fields, embedded documents. The worst outcome is unintentional denormalization where duplicated data diverges. Make the choice consciously and document it.
