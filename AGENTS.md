# Repository Guidelines

## Project Structure & Module Organization
- `brokle/` is the main SDK package. Core client/config lives in `brokle/_client.py`, `brokle/_base_client.py`, and `brokle/config.py`. Public patterns are in `brokle/wrappers/` (client wrappers), `brokle/decorators.py` (`@observe`), and `brokle/evaluate.py` (evaluation helpers).
- `tests/` holds pytest suites plus documentation (`tests/README.md`).
- `examples/` and `docs/` contain usage samples and guides.
- Build artifacts land in `dist/` and metadata in `brokle.egg-info/`.

## Build, Test, and Development Commands
Use the Makefile for repeatable workflows:
- `make install-dev` installs dev extras.
- `make test` runs the full pytest suite.
- `make test-coverage` runs tests with coverage reports.
- `make format` formats with Black + isort.
- `make lint` runs Flake8.
- `make type-check` runs mypy.
- `make dev-check` runs lint + type-check + coverage.
- `make build` builds distributions after cleaning.

## Coding Style & Naming Conventions
- Python with 4-space indentation; keep lines <= 88 characters (Black default).
- Formatting: Black + isort (`profile = black`).
- Linting: Flake8. Type checking: mypy with strict settings.
- Tests follow `test_*.py` files, `Test*` classes, and `test_*` functions.

## Testing Guidelines
- Framework: pytest (see `pytest.ini`). Markers: `unit`, `integration`, `slow`, `asyncio`.
- Naming: `test_<functionality>_<scenario>` where possible.
- Run a single file: `python -m pytest tests/test_streaming_wrappers.py -v`.

## Compatibility Notes
- Backward compatibility is not required yet; there is no production data because the product has not been released.

## Commit & Pull Request Guidelines
- Recent history uses conventional prefixes like `feat:`, `fix:`, `refactor(scope):`, `chore:`.
- Preferred format (from `CONTRIBUTING.md`): `<type>: <description>` with optional body.
- PRs should include a clear description, linked issues, testing notes, and highlight breaking changes. Add screenshots or examples when behavior changes.

## Configuration & Security Notes
- Local runs require `BROKLE_API_KEY`; optional `BROKLE_BASE_URL` defaults to `http://localhost:8080`.
- Mask sensitive data using the masking utilities in `brokle/utils/` before sending telemetry.

## Known Gotchas

1. **Lazy module loading via `__getattr__`** — `brokle/__init__.py` uses `__getattr__` + `importlib` for 150+ exports. `from brokle import *` won't work. Must import specific names: `from brokle import observe, get_client`.
2. **First-write-wins singleton** — First `Brokle()` call auto-registers as the global client. A second `Brokle()` call does NOT override — use `set_client()` explicitly. `get_client()` returns the singleton.
3. **`BROKLE_` env var prefix, uppercase only** — All env vars must be `BROKLE_API_KEY`, `BROKLE_BASE_URL`, `BROKLE_ENABLED` etc. Lowercase `brokle_api_key` does not work.
4. **No atexit flush** — SDK does not register process exit handlers. Serverless/CLI apps must call `brokle.flush()` before process exit or traces are silently lost.
5. **mypy selectively disabled** — `pyproject.toml` has `ignore_errors = true` overrides for `brokle.wrappers.*`, `brokle.config`, `brokle._client`, and `brokle.decorators`. Don't assume full type safety in those modules.
6. **Sync vs Async client is a one-time choice** — `Brokle` (sync) and `AsyncBrokle` (async) share `BaseBrokleClient` but are not interchangeable. Sync span context won't work in async code and vice versa. Pick one at initialization.
7. **`@observe` decorator requires initialized client** — The decorator uses `_client_context` (ContextVar) implicitly. If used without prior `get_client()` or `Brokle()`, it fails silently or uses the wrong client.
8. **Provider wrapper reflection** — Wrappers in `brokle/wrappers/` use reflection to hook into provider SDK internals. Breaking changes in provider SDK versions (e.g., OpenAI v1 → v3) require wrapper updates. The wrapper doesn't validate provider SDK version at import.
9. **BrokleConfig is a dataclass, not Pydantic** — Despite `pydantic>=2.0.0` in dependencies, `BrokleConfig` is a plain dataclass. No automatic validation, coercion, or JSON schema generation on config objects.
10. **`enabled=False` disables everything** — Master switch disables all telemetry including metrics. For cost optimization, use `sample_rate` instead.

## Lessons Learned

- 2026-04-14: The root repo's AGENTS.md documents cross-cutting SDK gotchas (submodule workflow, singleton pattern, optional peer deps). Check it for platform-wide context before making changes.

## Compatibility Notes
- Backward compatibility is not required yet; there is no production data because the product has not been released.