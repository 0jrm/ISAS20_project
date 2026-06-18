# Pre-commit Checklist

Before committing, verify the change is one coherent unit; formatting/lint/type checks pass; names expose intent; no exception is swallowed; changed behavior has tests; every bug fix adds a regression test. If numerical code changed, verify tolerances, seeds, dependency pins, and no exact floating-point equality.
