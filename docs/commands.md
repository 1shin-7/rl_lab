# CLI Reference

`drl_lab` provides a robust command-line interface via `rlab`.

## Global Options

*   `--debug`: Enable debug logging output.

## Commands

### `train`

Train an agent on a specific task.

```bash
rlab train [OPTIONS] [TASK]
```

**Arguments:**
*   `TASK`: Name of the task (e.g., `cartpole`, `cliff_walking`). Default: `cliff_walking`.

**Options:**
*   `--episodes INTEGER`: Number of episodes to train. Default: 500.
*   `--output TEXT`: Path to save the model (`.pth`).
*   `--visual`: Enable TUI visualization during training (includes real-time plots and logs).
*   `--visual-logs INTEGER`: Number of log lines to show in visual mode. Default: 5.

### `infer`

Run inference using a trained agent.

```bash
rlab infer [OPTIONS] [TASK]
```

**Arguments:**
*   `TASK`: Name of the task. Default: `cliff_walking`.

**Options:**
*   `--episodes INTEGER`: Number of episodes to run inference. Default: 5.
*   `--weight TEXT`: Path to load model weights from.
*   `--visual`: Enable TUI visualization during inference.

### `clean`

Clean up generated artifacts (models, plots) for a task.

```bash
rlab clean [TASK_NAME]
```

**Arguments:**
*   `TASK_NAME`: Name of the task to clean.
