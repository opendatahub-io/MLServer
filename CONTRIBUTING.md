# Contributing to MLServer

Thank you for your interest in contributing to MLServer. This guide covers the
development setup, coding conventions, and PR process.

## Development Setup

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Runtime |
| Poetry | 1.x | Dependency management |
| Make | any | Task runner |
| Docker | 20+ | Container builds and integration tests |

### Clone and Install

```bash
git clone https://github.com/opendatahub-io/MLServer.git
cd MLServer
poetry install
```

To work on a specific runtime plugin:

```bash
cd runtimes/sklearn
poetry install
```

### Running Locally

```bash
# Start the server (looks for models in current directory)
poetry run mlserver start .
```

## Code Style

- **Python** — follow PEP 8. The codebase uses type annotations throughout.
- **Docstrings** — use triple-quoted docstrings on all public classes,
  methods, and functions. Keep them concise — explain *why*, not *what*,
  unless the behaviour is non-obvious.
- **Imports** — group into stdlib, third-party, and local. Use relative
  imports within the `mlserver` package.
- **Settings** — all configuration uses Pydantic Settings classes in
  `mlserver/settings.py`. New settings should include a docstring and
  support environment variable override via the `MLSERVER_` prefix.

## Commit Message Convention

Format: `type(scope): description`

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Build, CI, dependency updates |

Examples:
```
feat(batching): add configurable batch timeout per model
fix(parallel): handle worker crash during model reload
docs: update architecture diagram with kafka flow
```

## Running Tests

```bash
# Full test suite
make test

# Core tests only
poetry run pytest tests/

# Single file
poetry run pytest tests/batch_processing/test_rest.py

# Runtime-specific tests
cd runtimes/sklearn && poetry run pytest tests/
```

## Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make changes and ensure tests pass.
3. Add docstrings to any new public functions or classes.
4. Choose a PR template from [`/docs/PULL_REQUEST_TEMPLATE/`](./docs/PULL_REQUEST_TEMPLATE/).
5. Submit the PR targeting `master`.

### Before Opening a PR

- Is the change important and ready enough to ask the community to spend
  time reviewing?
- Have you searched for existing, related issues and pull requests?
- Is the change being proposed clearly explained and motivated?

## Adding a New Runtime

1. Create a new directory under `runtimes/` with a Poetry project.
2. Subclass `mlserver.MLModel` and implement `load()` and `predict()`.
3. Add the canonical import path to `ALLOWED_MODEL_IMPLEMENTATIONS` in
   `mlserver/settings.py`.
4. Add tests validating allowlist behaviour.
5. Add documentation in `docs/runtimes/`.

For custom runtimes not shipped with this repository, use the
`mlserver build` workflow with `--allow-runtime` — do not extend the global
allowlist.

## Engineering Documentation

For architectural context, design decisions, and the security model, see:

- [Architecture](./docs/engineering/architecture.md)
- [ADRs](./docs/engineering/adr/)
- [Security Model](./docs/engineering/security.md)
- [Developer Onboarding](./docs/engineering/onboarding.md)

## License

When you contribute code, you affirm that the contribution is your original
work and that you license the work to the project under the project's open
source license.
