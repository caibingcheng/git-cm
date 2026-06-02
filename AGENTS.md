# Agent Instructions for git-cm

## Development Setup

```bash
pip install -e ".[dev]"
```

Or activate the existing `.venv` at repo root.

## Running Tests

```bash
pytest tests/ -v
pytest tests/test_cli.py -v                     # single test file
pytest tests/test_cli.py::TestClass -v           # single test class
```

Tests run against `src/` via `pythonpath = ["src"]` in `pyproject.toml`.

## Project Structure

- Source: `src/git_cm/` — `cli.py` is the entry point
- Tests: `tests/` — uses temp dirs and mocked configs; some tests create real git repos
- Entry point: `git-cm` (defined in `pyproject.toml` scripts)

## Tooling

- **No lint, format, or typecheck tools are configured** (no ruff, black, mypy, etc.)
- pytest + pytest-cov are the only dev dependencies
- Coverage config in `pyproject.toml` targets `src/git_cm`

## Conventions

- Commit messages in this repo use **Conventional Commits** (`feat:`, `fix:`, etc.)
