# Task Development Guide

In `rl_lab`, a **Task** is the central unit of development. it encapsulates the RL environment logic, state preprocessing, hyperparameter fine-tuning, and visual representation.

## BaseTask API

All tasks must inherit from `drl_lab.tasks.base.BaseTask`.

### Required Methods & Properties

*   **`get_env(self) -> gymnasium.Env`**: Returns the initialized environment instance. This is where you instantiate the gym environment and apply any necessary wrappers.
*   **`state_size(self) -> int`**: Dimension of the state vector.
*   **`action_size(self) -> int`**: Number of possible actions.
*   **`create_model(self) -> torch.nn.Module`**: Returns the PyTorch model architecture. It is recommended to use `DuelingMLP` from `drl_lab.models` for better stability.

### Optional Methods

*   **`preprocess_state(self, state: Any) -> Any`**: Transform raw observations (e.g., one-hot encoding, normalization) before they reach the Agent.
*   **`render(self) -> BaseTaskTUI`**: Provide a custom TUI interface. Defaults to `DefaultTaskTUI`.

---

## Inheritance & Customization

Subclassing `BaseTask` allows you to customize every aspect of the training process for a specific environment.

### 1. Environment Patching (Reward Shaping)
Often, standard environment rewards are too sparse for quick convergence. You should use `gymnasium.Wrapper` to "patch" the environment behavior within `get_env`.

**Example: Centered Reward for CartPole**
```python
from gymnasium import Wrapper

class CenteredRewardWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Penalize distance from center
        x = obs[0]
        penalty = abs(x) / self.env.unwrapped.x_threshold
        return obs, reward - (penalty * 0.5), terminated, truncated, info

# Inside your Task class:
def get_env(self):
    env = gym.make("CartPole-v1")
    return CenteredRewardWrapper(env)
```

### 2. Hyperparameter Fine-Tuning
Each task instance holds its own `self.config` (an instance of `drl_lab.utils.Config`). You can fine-tune RL parameters directly in the task's `__init__` to optimize performance for that specific environment.

```python
def __init__(self, config=None):
    super().__init__("MyTask-v0", config)
    # Fine-tune parameters for this environment
    self.config.learning_rate = 0.0005  # Smaller LR for stability
    self.config.gamma = 0.95           # Focus on short-term rewards
    self.config.epsilon_decay = 0.999  # Explore longer
```

---

## Lifecycle Hooks

Hooks allow you to inject logic into the training loop managed by the `Trainer`.

| Hook | Timing | Use Case |
| :--- | :--- | :--- |
| `pre_training()` | Before the first episode. | Initialize external logs, allocate buffers. |
| `post_training()`| After training ends or stops. | Close environment, save custom heatmaps. |
| `pre_episode(idx)`| Before `env.reset()`. | Adjust curriculum (e.g., randomize start pos). |
| `post_episode(idx, r)` | After an episode finishes. | Log task-specific metrics to console. |
| `sync_data(data)` | Custom intervals. | Send telemetry to a database or dashboard. |

---

## Task Structure Conventions

### Simple Tasks (Single File)
For tasks without complex visual requirements.
*   **Location**: `drl_lab/tasks/my_task.py`

### Complex Tasks (Module)
Required when using custom CSS or advanced TUI widgets.
*   **Location**: `drl_lab/tasks/my_task/`
*   **Structure**:
    ```text
    my_task/
    ├── __init__.py  # Export the Task class
    ├── task.py      # Core BaseTask logic
    ├── tui.py       # BaseTaskTUI implementation
    └── styles.tcss  # Textual CSS styles
    ```

## Registration

Register your task in `drl_lab/tasks/__init__.py`:

```python
from .my_task import MyTask
registry.register("my_task", MyTask)
```
Now available via `rlab train my_task`.