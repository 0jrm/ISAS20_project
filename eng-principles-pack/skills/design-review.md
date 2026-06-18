# Design Review

Source: SOLID, DRY, KISS, YAGNI, Separation of Concerns, Least Astonishment, Architecture Patterns.

SOLID
Five object-oriented design principles that keep codebases maintainable.
Single Responsibility (one reason to change), Open/Closed (open for extension, closed for modification), Liskov Substitution (subtypes must be substitutable), Interface Segregation (clients shouldn't depend on unused interfaces), Dependency Inversion (depend on abstractions, not concretions). These apply beyond OOP — think in terms of modules, functions, and services.
DRY — Don't Repeat Yourself
Every piece of knowledge must have a single, authoritative representation.
Duplication is the root of most maintenance debt. When logic exists in two places, they will eventually diverge. However, premature abstraction to avoid repetition is its own trap — two nearly identical pieces of code may represent genuinely different concepts. Ask: is this the same knowledge, or just similar-looking code?
KISS — Keep It Simple
Simplicity is the ultimate sophistication. Solve the actual problem, no more.
Complexity is the enemy of reliability. Every abstraction, layer, and pattern adds cognitive overhead and failure points. The simplest correct solution is almost always the best long-term. Avoid speculative complexity — don't add configurability, extensibility, or generality until there's a concrete reason to.
YAGNI — You Aren't Gonna Need It
Don't implement something until you actually need it.
Anticipated requirements are wrong more often than they are right. Building for imagined futures incurs real costs today: complexity, maintenance burden, and misleading abstractions. Build what's needed now; refactor when the real need emerges. This pairs directly with iterative development cycles.
Separation of Concerns
Divide a system so each component addresses a distinct concern.
Data access, business logic, and presentation belong in different layers. Mixing them creates code that is hard to test in isolation, hard to reason about, and brittle to change. The boundary between concerns should be explicit and narrow — ideally a well-defined interface or API.
Principle of Least Astonishment
Code should behave the way a reader would expect it to.
Functions named `get_user()` should not delete rows. A method that sorts in-place should say so, or not sort in-place. Surprising behavior is a bug even when it's technically documented. Design APIs, names, and behaviors so the obvious assumption is the correct one.
Architecture Patterns
Layered / N-Tier Architecture
Organizes code into horizontal layers: presentation, domain, data.
Dependencies flow downward only. The presentation layer never touches the database directly. Each layer can be tested, replaced, or scaled independently. Common in enterprise software. Risk: 'lasagna code' where changes percolate through every layer even for simple additions.
Hexagonal / Ports & Adapters
The core domain is isolated from infrastructure via defined ports.
Business logic knows nothing about HTTP, databases, or file systems. It communicates through interfaces (ports). Concrete implementations (adapters) can be swapped without touching the core. This makes testing trivial — swap real databases for in-memory fakes. Critical for long-lived scientific software where storage backends change.
Event-Driven Architecture
Components communicate by emitting and reacting to events.
Decouples producers from consumers in time and space. Enables high scalability and resilience. Introduces complexity in tracing, ordering, and exactly-once delivery. Use event sourcing (persisting events as the source of truth) when audit trails or temporal queries matter — common in financial and scientific pipelines.
Microservices vs Monolith
Choose based on team size, domain boundaries, and operational maturity.
A well-structured monolith is almost always preferable to a poorly-designed microservices system. Microservices win at independent scaling, independent deployment, and team autonomy at scale. They lose at latency, distributed transaction complexity, and debugging. The 'modular monolith' is a middle path: clean internal boundaries without the operational overhead of distributed systems.
Domain-Driven Design (DDD)
Model software around the business domain using a ubiquitous language.
Bounded contexts establish explicit boundaries between subsystems with their own models. Aggregates group related entities with clear ownership rules. The ubiquitous language — shared by developers and domain experts — is the most valuable DDD artifact. Particularly useful in scientific software where domain complexity is high and domain experts exist.
