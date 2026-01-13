from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from textual.app import ComposeResult
from textual.widgets import Label
from textual.containers import Container
from textual.reactive import reactive
from textual.widget import Widget

class TaskHeader(Container):
    """
    A standard header for RL tasks displaying Episode, Step, and Reward.
    """
    DEFAULT_CSS = """
    TaskHeader {
        layout: horizontal;
        height: 3;
        dock: top;
        background: $primary-background;
        border-bottom: solid $accent;
        padding: 0 1;
    }
    
    TaskHeader Label {
        width: 1fr;
        content-align: center middle;
        text-style: bold;
    }
    """

    episode = reactive(0)
    step = reactive(0)
    reward = reactive(0.0)
    
    def compose(self) -> ComposeResult:
        yield Label(f"Episode: {self.episode}", id="episode")
        yield Label(f"Step: {self.step}", id="step")
        yield Label(f"Reward: {self.reward:.2f}", id="reward")

    def watch_episode(self, value: int):
        try:
            self.query_one("#episode", Label).update(f"Episode: {value}")
        except Exception: 
            pass

    def watch_step(self, value: int):
        try:
            self.query_one("#step", Label).update(f"Step: {value}")
        except Exception: 
            pass

    def watch_reward(self, value: float):
        try:
            self.query_one("#reward", Label).update(f"Reward: {value:.2f}")
        except Exception: 
            pass

class BaseTaskTUI(ABC):
    """
    Abstract interface for Task TUIs.
    Maintains the structure of Header + Content.
    """
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.header = TaskHeader()

    @abstractmethod
    def get_main_widget(self) -> Widget:
        """Returns the main content widget for the task visualization."""
        pass

    def compose_view(self) -> ComposeResult:
        """
        Composes the full task view (Header + Main Widget).
        Helper for the main App to mount the visual components.
        """
        yield self.header
        yield Container(self.get_main_widget(), id="task-content")

    def update_state(self, state: Any, info: Optional[Dict[str, Any]] = None):
        """
        Update the visualization with new state.
        Override this to update your custom widgets.
        """
        pass

    def update_stats(self, episode: int, step: int, reward: float):
        """Update the header stats."""
        self.header.episode = episode
        self.header.step = step
        self.header.reward = reward

class DefaultTaskTUI(BaseTaskTUI):
    """
    Default TUI implementation if a task doesn't provide one.
    Displays the task name in the center.
    """
    
    def get_main_widget(self) -> Widget:
        label = Label(f"Task: {self.task_name}\n\n(No custom visualization available)", id="default-label")
        label.styles.align = ("center", "middle")
        label.styles.width = "100%"
        label.styles.height = "100%"
        return label
