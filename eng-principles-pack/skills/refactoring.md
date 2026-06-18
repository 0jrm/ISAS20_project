# Engineering Process: Commits, Review, Debt

Source: Version Control & CI/CD + Code Review + Technical Debt.

Version Control & CI/CD
Every Change Is a Commit
Commits are the atomic unit of history. Make them small, coherent, and well-described.
A good commit contains one logical change: a bug fix, a feature addition, a refactoring. Commits that mix concerns make bisection hard and code review harder. Write commit messages in the imperative mood ('Fix off-by-one in integrator'), explain the why when it's not obvious, and reference issue trackers. History is an asset — treat it that way.
Continuous Integration
Every commit should be automatically built and tested.
CI catches integration failures immediately. The faster the feedback loop, the cheaper the fix. Keep CI fast — a slow pipeline gets skipped. Build and test in a clean environment to catch 'works on my machine' failures. Enforce: linting, formatting, type checking, unit tests, and integration tests at minimum. Never merge a broken build to the main branch.
Feature Flags Over Long-Lived Branches
Merge incomplete features behind flags rather than maintaining parallel branches.
Long-lived feature branches accumulate merge debt that compounds daily. Feature flags let you ship code continuously while controlling what users see. They also enable A/B testing, gradual rollouts, and instant rollbacks. The flag becomes the branch. Clean up old flags aggressively — flag accumulation is its own form of technical debt.
Code Review
Reviews Are for Knowledge Transfer, Not Just Bug-Finding
Code review spreads understanding of the codebase across the team.
The bus factor (how many people need to be hit by a bus before a component is unmaintainable) is a real risk. Code review ensures at least two people understand every significant change. The knowledge transfer function often outweighs the bug-catching function. Senior engineers should prioritize reviewing code from less experienced engineers — it multiplies their impact.
Review the Design, Not the Syntax
Linters catch style issues. Human reviewers should focus on correctness and design.
Don't spend review time on whitespace, naming conventions, or formatting — that's what automated tools are for. Focus on: correctness of the logic, appropriate algorithm choice, API design decisions, security implications, performance characteristics, test coverage, and whether the change solves the stated problem. Be specific in feedback — 'I think this is wrong' is not useful; 'this loop will skip the last element because...' is.
Approve Fast, Block Rarely
Code review latency is a key developer productivity metric. Keep it low.
Reviews that take days demoralize authors and create integration debt. Aim to review within one business day. Distinguish blocking issues (correctness, security, API design) from suggestions (style improvements, minor optimizations). Mark suggestions clearly — authors should not need to guess if a comment is blocking approval. Use 'Request Changes' sparingly; prefer constructive suggestions on approved PRs.
Technical Debt
Debt Must Be Tracked and Paid
Untracked technical debt is the most dangerous kind.
Every shortcut taken knowingly should be tracked in the issue tracker with a clear description of what was compromised and why. 'Deliberate' debt (tactical shortcut to meet a deadline) is manageable; 'accidental' debt (not knowing a better approach existed) is avoidable through learning. Schedule debt repayment — allocate 20% of engineering cycles to improvement work, not just feature work.
Refactor Continuously, Not in Big Bangs
Continuous small refactors are safer and more effective than periodic rewrites.
The 'we'll clean this up in the rewrite' plan almost never works — rewrites take longer than expected, accumulate their own debt, and frequently reproduce the same design mistakes. Instead, apply the Boy Scout Rule: leave every file slightly better than you found it. Rename the misleading variable. Extract the repeated logic. Small, continuous improvements compound into significantly better codebases.
Measure Before Optimizing
Premature optimization is the root of much evil. Profile first, then optimize.
Human intuition about performance bottlenecks is reliably wrong. 90% of execution time is typically in 10% of the code, and it's rarely where you expect. Profile first to find the actual bottleneck. Only then optimize. Document what you measured, what you changed, and what the improvement was. Unmeasured optimizations create complexity without verified benefit.
Algorithmic Complexity First
An O(n²) algorithm cannot be micro-optimized into an O(n log n) one.
The most impactful performance improvements almost always come from algorithm changes, not micro-optimizations. Before optimizing constants, ensure the asymptotic complexity is correct. A hash map lookup (O(1)) beats a sorted list binary search (O(log n)) at scale regardless of constants. Understand the growth characteristics of your code with respect to input size.
Reliability & Resilience
Design for Failure
Every external dependency will fail. Design every interaction with this assumption.
Network calls timeout or return errors. Disks fill up. Databases become unavailable. Services deploy bad versions. Resilient systems handle these cases gracefully: retry with exponential backoff and jitter, implement circuit breakers to stop calling failing services, use bulkheads to isolate failure domains. Test failure paths explicitly with chaos engineering or fault injection.
Idempotency Enables Recovery
Operations that can be safely retried enable automatic recovery from transient failures.
When a network request fails, you often don't know if the server received it. Idempotent operations can be safely retried without double-processing. Build idempotency keys into mutation APIs. In data pipelines, design stages to be restartable from checkpoints. The ability to safely retry is one of the most valuable reliability properties a system can have.
Define and Measure SLOs
Service Level Objectives quantify the reliability you are committing to provide.
Without defined SLOs, 'reliable' means different things to different people. Define concrete targets: 99.9% availability, p99 latency under 200ms, error rate below 0.1%. Measure against them continuously. Use error budgets — the gap between 100% and your SLO target — to make explicit decisions about stability vs. velocity. An exhausted error budget is a signal to pause feature work and improve reliability.
