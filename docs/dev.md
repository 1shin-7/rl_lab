# Developer Guide

## Architecture Overview

`drl_lab` is designed with modularity and extensibility in mind. The core components are decoupled to allow easy addition of new tasks and algorithm improvements.

### Core Components

*   **`drl_lab.tasks`**: Contains task definitions. All tasks inherit from `BaseTask`.
*   **`drl_lab.agent`**: Implements the RL agent (currently `BaseDQNAgent` with DDQN/Dueling support).
*   **`drl_lab.train.Trainer`**: Manages the training loop, lifecycle hooks, and callbacks.
*   **`drl_lab.cli`**: Handles command-line arguments and TUI application launching.

## Developing New Tasks

To add a new environment/task to `drl_lab`, you need to implement the `BaseTask` interface and optionally a `BaseTaskTUI` for visualization.

### 1. The `BaseTask` Class

Located in `drl_lab/tasks/base.py`.

```python
from drl_lab.tasks.base import BaseTask
import gymnasium as gym

class MyNewTask(BaseTask):
    def get_env(self) -> gym.Env:
        """Return the gymnasium environment instance."""
        return gym.make("MyEnv-v0")

    @property
    def state_size(self) -> int:
        """Return state dimension."""
        return 4

    @property
    def action_size(self) -> int:
        """Return action dimension."""
        return 2

    def create_model(self) -> nn.Module:
        """Return the PyTorch model (e.g., DuelingMLP)."""
        return DuelingMLP(self.state_size, self.action_size)
    
    def preprocess_state(self, state):
        """Optional: Preprocess state before passing to agent."""
        return state
```

### 2. Registering the Task

Register your new task in `drl_lab/tasks/__init__.py`:

```python
from .my_task import MyNewTask
registry.register("my_task", MyNewTask)
```

### 3. TUI Visualization (Optional but Recommended)

Implement `BaseTaskTUI` in `drl_lab/tasks/visual.py` (or a dedicated `tui.py` in your task module).

```python
from drl_lab.tasks.visual import BaseTaskTUI
from textual.widget import Widget

class MyTaskTUI(BaseTaskTUI):
    def get_main_widget(self) -> Widget:
        return MyCustomWidget() # Textual Widget implementation

    def update_state(self, state, info=None):
        # Update widget based on state
        self.widget.update(state)
```

Then override `render()` in your Task class:

```python
def render(self) -> BaseTaskTUI:
    return MyTaskTUI(self.name)
```

## Lifecycle Hooks

`BaseTask` provides hooks for advanced control:

*   `pre_training()`: Called before training starts.
*   `post_training()`: Called after training ends.
*   `pre_episode(episode)`: Called before each episode.
*   `post_episode(episode, reward)`: Called after each episode.

These are useful for curriculum learning, custom logging, or resource management.
