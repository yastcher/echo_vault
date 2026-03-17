---
name: review
description: Self-review checklist before finishing a task
disable-model-invocation: true
user-invocable: true
---

# Self-Review Checklist

Run through every item before reporting task as complete.

## Code quality

1. `uv run ruff check --fix` — zero warnings, no `noqa` comments added
2. `uv run ruff format` — no formatting changes
3. `uv run pytest` — all pass, coverage >= 85%

## Conventions

4. Stdlib imports: `import datetime`, `import typing` — module style, not `from`
5. Constants: magic strings use `src/const.py`, accessed as `const.X`
6. No circular import hacks (no local imports inside functions for dependency resolution)
7. DDD structure: each domain module has router/schemas/models/service/dependencies/exceptions

## User-facing

8. User-facing strings use localization (all 4 languages), not hardcoded
9. Documentation updated: README + localized READMEs, ADMIN.md if needed

## Security

10. No secrets/tokens in logs, responses, or error messages
11. Input validated via Pydantic before use
12. Auth checks on all endpoints
13. Rate limits on new public endpoints

## Summary

14. Run `git diff --stat` — list all changed files, verify diff under ~500 lines

Report pass/fail for each item.
