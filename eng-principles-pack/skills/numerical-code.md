# Numerical & Scientific Code

Source: Scientific & Numerical Testing + Memory Access Patterns Matter (performance section).

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
Memory Access Patterns Matter
Cache-friendly code can be orders of magnitude faster than cache-unfriendly code.
Modern CPUs are memory-bound, not compute-bound. Sequential memory access (arrays) is dramatically faster than random access (pointer chasing). Structure-of-Arrays often outperforms Array-of-Structures for vectorizable code. In numerical computing, matrix traversal order (row-major vs column-major) can affect performance by 10x. Understand your language's memory layout.
