---
name: spec
description: Execute a task specification file with TDD workflow
disable-model-invocation: true
user-invocable: true
argument-hint: [path/to/spec.md]
---

# Execute Task Spec

Read the task spec at `$ARGUMENTS`.

## Workflow

1. Read the spec file completely
2. List all tasks and acceptance criteria — confirm the plan with user before starting
3. For each task:
   - Write tests first (TDD)
   - Implement the code
   - Run `uv run pytest` to verify no regressions
4. After all tasks complete:
   - `uv run ruff check --fix`
   - `uv run ruff format`
   - `uv run pytest` — full suite, verify coverage >= 85%
   - Update documentation (README, localization) if functionality changed
5. Report summary: files changed, tests added, coverage

## Rules

- Do NOT make git commits, checkout, reset, or any git write operations
- Do NOT delete files not tracked in git
- Do NOT remove existing providers, configs, or feature flags
- Stop and propose a commit when diff approaches ~500 lines
- Fix one file at a time when dealing with linter/import issues
