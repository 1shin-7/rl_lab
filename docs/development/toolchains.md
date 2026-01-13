# Toolchains & Guidelines

## Environment

*   **Python**: >= 3.12 (Uses modern syntax like `|` for types).
*   **Package Manager**: `uv`. Always run commands through `uv run` to ensure consistency.

## Code Quality (`clippy`)

We enforce strict linting and formatting via **Ruff**.

### Rulesets Enabled
*   `E`, `F`: Standard Error/Failure rules.
*   `B`: Bugbear (design flaws).
*   `UP`: Pyupgrade (modern syntax enforcement).
*   `I`: Isort (import sorting).
*   `SIM`: Simplify (code simplification suggestions).

### Usage
Run the check and auto-fix:
```bash
uv run ruff check . --fix
```

## Development Guidelines

1.  **Type Hinting**: Mandatory for all function signatures. Use Python 3.10+ native pipe syntax (`str | None`) instead of `Optional`.
2.  **Pathlib**: Always use `pathlib.Path` for type annotations. Use `utils.paths` for actual path resolution.
3.  **Exception Handling**: Use `contextlib.suppress()` instead of empty `try-except` blocks when the failure is expected and safe to ignore.
4.  **Decoupling**: Keep TUI logic strictly separate from the Task/Agent core. Use callbacks or properties for communication.
5.  **Clean Code**: Avoid long lines (> 88 characters). Use parenthesized expressions or temporary variables to break them up.
