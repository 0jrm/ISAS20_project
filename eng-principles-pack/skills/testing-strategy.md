# Testing Strategy

Source: Testing Strategy + Scientific & Numerical Testing.

Testing Strategy
The Test Pyramid
Many unit tests, fewer integration tests, few end-to-end tests.
Unit tests are fast, cheap, and precise — test one function or class in isolation. Integration tests verify that components work together: database queries, service calls, external APIs. End-to-end tests verify entire user journeys — slow, fragile, valuable for critical paths. An inverted pyramid (many E2E, few unit) is a common anti-pattern that creates slow, flaky test suites.
Test Behavior, Not Implementation
Tests should survive refactoring. Tests tied to implementation break when you improve code.
Test what a function does (its observable output given inputs), not how it does it (which private methods it calls). Tests that assert on internal state or private methods break whenever you refactor, punishing improvement. The test suite should be a safety net, not a cage. Mock external dependencies, not internal logic.
Test-Driven Development (TDD)
Write the test before the code. Red → Green → Refactor.
TDD forces you to think about the interface before the implementation. Tests become a design tool, not just a verification tool. The discipline of writing tests first reveals awkward APIs before they're built in. TDD works best for algorithmic logic. It's less naturally suited to UI, exploratory research code, or integrations — in these cases, 'test shortly after' is a pragmatic compromise.
Property-Based Testing
Describe invariants; let the framework generate hundreds of test cases.
Instead of testing `sort([3,1,2]) == [1,2,3]`, test that for any list, the output is non-decreasing and a permutation of the input. Frameworks like Hypothesis, QuickCheck, or fast-check generate random inputs that satisfy given constraints and shrink failures to minimal examples. Invaluable in scientific computing where invariants (energy conservation, probability sums to 1) can be expressed and verified.
Regression Tests Are Permanent
Every bug fixed must produce a test that would have caught it.
A regression test encodes the knowledge that this specific failure is possible. It documents the bug, prevents its recurrence, and signals if a future change re-introduces it. A bug fixed without a test is a bug that will likely return. The test suite should grow monotonically; deleting tests requires explicit justification.
Scientific & Numerical Testing
Verify Against Known Solutions
Test numerical code against analytical solutions, manufactured solutions, or benchmarks.
The Method of Manufactured Solutions (MMS): construct an exact solution, derive what source terms it implies, run the solver with those terms, compare output to the exact solution. This verifies correctness even when no real-world analytical solution exists. For differential equations solvers, verify convergence order — a second-order method should show O(h²) error reduction.
Numerical Tolerance in Assertions
Floating-point equality is always approximate. Use relative and absolute tolerances.
Never assert `result == 1.0`. Assert `abs(result - 1.0) < 1e-10`. Choose tolerances based on the expected precision of the algorithm, not arbitrarily. Be aware of catastrophic cancellation (subtracting nearly equal numbers loses precision). Test edge cases: near-zero values, very large values, NaN propagation, and inf handling.
Reproducibility Testing
Fix random seeds and pin dependencies for reproducible results.
Stochastic algorithms must be seeded for testing. Version pin all numerical libraries — numpy, scipy, LAPACK versions affect floating-point results. Store test fixture data in version control. For long-running simulations, test that checkpointing and restarting produces bit-for-bit identical results to uninterrupted runs.
