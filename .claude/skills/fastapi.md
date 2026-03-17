# FastAPI Conventions

## Module structure

Each domain module in src/ contains:

- router.py — endpoints
- schemas.py — Pydantic models (request/response)
- models.py — DB models
- service.py — business logic
- dependencies.py — FastAPI Depends
- exceptions.py — domain-specific exceptions

## Async

- Never use blocking operations (time.sleep, sync HTTP) in async def
- Use `def` for sync I/O or `await asyncio/run_in_threadpool`
- CPU-bound tasks — separate process

## Pydantic

- Use validators: Field, EmailStr, regex
- Create CustomBaseModel as project base
- Split BaseSettings per module

## Dependencies

- Use for data validation (check existence in DB)
- Chain dependencies
- Prefer async def
- Unified path parameter names for dependency reuse

## Database

- SQL-first: joins and aggregations in DB, not in Python
- Naming: lower_snake_case, singular
- Explicit index names
