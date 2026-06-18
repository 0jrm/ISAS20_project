# AI-Assisted Development

Source: Using AI Effectively + AI in Testing & Debugging.

Using AI Effectively
AI as a Pair Programmer, Not an Oracle
AI generates plausible code; you are responsible for understanding and verifying it.
AI tools excel at boilerplate, common patterns, and routine transformations. They confidently produce wrong answers in edge cases, numerical code, and security-sensitive contexts. Every AI-generated snippet must be read, understood, and tested as if you wrote it yourself — because you are responsible for it. Use AI to speed up implementation of understood design, not to skip the understanding step.
Prompt for Constraints, Not Just Output
Include requirements, constraints, and edge cases in your prompts for better results.
Vague prompts yield vague code. Specify: language, performance constraints, error handling requirements, security considerations, and what not to do. 'Write a sort function' yields a naive solution. 'Write a sort function that is stable, handles None values by placing them last, and runs in O(n log n) worst case' yields a useful one. The quality of AI output is highly correlated with the specificity of the prompt.
AI-Assisted Code Review
Use AI to surface common issues before human review, not to replace it.
AI code review tools catch many common issues: null pointer risks, common security patterns (SQL injection, XSS), resource leaks, and deviation from conventions. They are less reliable on architectural concerns, domain-specific correctness, and subtle race conditions. Treat AI review as a first pass that raises the floor — human review remains essential for semantic correctness and design quality.
Verifying AI-Generated Numerical Code
AI-generated numerical algorithms require especially rigorous verification.
Language models learn from text, not from mathematical understanding. AI-generated numerical code may have subtle precision issues, incorrect algorithm choices, or off-by-one errors in loop bounds. Always test AI-generated math against known solutions. Verify algorithm choice explicitly — an AI may use a method that is correct but inappropriate for your convergence or precision requirements.
AI in Testing & Debugging
AI-Generated Test Cases
AI is excellent at generating edge cases you might miss.
Given a function signature and description, AI tools can systematically enumerate boundary conditions, type errors, empty inputs, extreme values, and special cases. This is a high-value, low-risk application of AI — generated tests don't run until you approve them, and their correctness is easy to verify. Use AI to augment your test suite, especially for functions with complex input spaces.
Rubber Duck Debugging with AI
Explaining a bug to AI often surfaces the solution — and AI can respond with targeted questions.
Describing a bug forces you to articulate your assumptions — and incorrect assumptions become visible in the articulation. AI assistants can ask clarifying questions that expose hidden assumptions, suggest debugging approaches, and identify common failure patterns. However, AI descriptions of what code 'should' do may not match what it actually does — always run the code, don't reason from AI descriptions alone.
AI for Log Analysis
AI can quickly parse stack traces, identify error patterns, and suggest root causes.
Pasting a stack trace or error log to an AI assistant is often the fastest path to a hypothesis. AI tools recognize common library errors, framework-specific patterns, and well-known failure modes. They are less useful for proprietary systems, domain-specific numerical failures, or errors with environmental context not present in the log. Always verify the suggested fix rather than applying it blindly.
Know What AI Cannot Reliably Do
Concurrency bugs, race conditions, and security vulnerabilities require human expertise.
AI tools struggle reliably with: timing-dependent bugs, emergent behavior in distributed systems, subtle side-channel security vulnerabilities, and failures that only manifest under specific hardware or OS configurations. Do not rely on AI to audit code for production security vulnerabilities. Use dedicated static analysis tools, penetration testing, and expert human review for security-critical paths.
