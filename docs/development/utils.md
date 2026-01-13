# Utilities Reference

The `drl_lab.utils` module provides unified APIs for common operations.

## Logging (`logger`)

We use a pre-configured `loguru` instance. Use the unified import to ensure consistent formatting (e.g., Bold Magenta for `INFO`).

```python
from drl_lab.utils import logger

logger.info("Task started")
logger.error("Failed to load weights")
```

## Path Management (`paths`)

Always use the `paths` module for file system operations to ensure cross-platform compatibility and maintainable directory structures.

### Constants
*   `PROJECT_ROOT`: Absolute path to the repository root.
*   `WORK_DIR`: The current working directory.
*   `OUTPUTS_DIR`: Standard location for models and plots (`./outputs`).

### Helpers
*   `ensure_dir(path)`: Ensures the parent directory of a file (or the directory itself) exists.
*   `get_model_path(task_name)`: Returns the standard `.pth` path.
*   `resolve_task_paths(task_name, output_path)`: Smartly infers model and plot paths from user input.

## Task Matching (`matching`)

The `fuzzy_match` utility enables the "smart" CLI experience:
1.  **Exact Match**: Direct hit.
2.  **Unique Prefix**: `cliff` -> `cliff_walking`.
3.  **Fuzzy Match**: `cartpol` -> `cartpole` (with warning).

## Configuration (`Config`)

A centralized `dataclass` for hyperparameters. Each `BaseTask` instance maintains its own `config` attribute, allowing for task-specific overrides.
