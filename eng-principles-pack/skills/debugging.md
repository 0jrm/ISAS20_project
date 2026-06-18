# Debugging & Observability

Source: Debugging Methodology + Observability & Logging.

Debugging Methodology
Reproduce First
A bug you cannot reproduce reliably cannot be debugged reliably.
Before attempting to fix anything, establish a minimal reproducible example (MRE). Reduce the input, strip the code, and isolate the failure to its smallest form. This process often reveals the cause. An MRE is also the foundation of a bug report and the eventual regression test. If you can't reproduce it, add logging until you can.
Scientific Debugging
Debug like a scientist: form hypotheses, design experiments, confirm or reject.
Resist the urge to randomly change things. Form a hypothesis ('I think the error occurs when the input array is empty'). Design a minimal test that would confirm or deny it. Run it. Update your model. This approach is slower at each step but faster overall — random changes burn time without building understanding. Keep a debug log if the problem is complex.
Binary Search Debugging
Bisect the problem space to find the failure point in O(log n) steps.
If a pipeline of 100 steps produces wrong output, check step 50. If that's wrong, check step 25. If right, check step 75. This applies to commit history (git bisect), to data pipelines, and to multi-step computations. Methodical bisection outperforms intuition-driven search when the search space is large.
Understand Before Fixing
A fix applied without understanding the root cause usually creates a new bug.
If you don't know why the fix works, you don't know what else it might break. Understanding the bug fully often reveals that the fix you planned is wrong or incomplete. Write down your understanding of the root cause before touching the code. If you can't explain the bug, keep investigating. A correct explanation predicts where else the problem might manifest.
Observability & Logging
Structured Logging
Log key-value pairs, not free-form strings. Logs should be machine-parseable.
Structured logs enable querying, filtering, and alerting. Log `{user_id: 42, action: 'login', duration_ms: 120}` not `'User 42 logged in after 120ms'`. Use log levels consistently: DEBUG for internal state, INFO for notable events, WARN for recoverable anomalies, ERROR for failures. Log at decision points, not at every line.
The Three Pillars of Observability
Logs, metrics, and traces give different views of system behavior.
Logs: timestamped records of discrete events. Metrics: numeric measurements over time (latency, error rate, throughput). Traces: the path of a single request through a distributed system. Together they answer: what happened (logs), how often and how fast (metrics), where in the system (traces). No single pillar is sufficient — you need all three for production systems.
Alerts Should Be Actionable
Every alert that fires should require a human decision. Alert fatigue kills on-call effectiveness.
An alert that fires and requires no action is noise that trains engineers to ignore alerts. Before adding an alert, define the runbook: what does the engineer do when this fires? If the answer is 'check and see', the threshold is wrong. Set thresholds based on SLO impact, not arbitrary percentage thresholds.
