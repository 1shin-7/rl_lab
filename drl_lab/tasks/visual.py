from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from textual.app import ComposeResult
from textual.widgets import Label
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widget import Widget

class TaskHeader(Horizontal):
    """
    A compact, single-line header for RL tasks.
    Layout: [Episode/Step] -- [Task Name] -- [Reward]
    """
    DEFAULT_CSS = """
    TaskHeader {
        height: 1;
        dock: top;
        background: $accent;
        color: $text;
        align: center middle;
        padding: 0 1;
    }
    
    .stats-left {
        width: 1fr;
        text-align: left;
    }
    
    .title-center {
        width: 2fr;
        text-align: center;
        text-style: bold;
    }
    
    .stats-right {
        width: 1fr;
        text-align: right;
    }
    """

    episode = reactive(0)
    step = reactive(0)
    reward = reactive(0.0)
    
    def __init__(self, task_name: str):
        super().__init__()
        self.task_name = task_name

    def compose(self) -> ComposeResult:
        yield Label(f"Ep: {self.episode} | St: {self.step}", classes="stats-left", id="stats-left")
        yield Label(self.task_name, classes="title-center")
        yield Label(f"Rw: {self.reward:.2f}", classes="stats-right", id="stats-right")

    def watch_episode(self, value: int):
        self._update_left()

    def watch_step(self, value: int):
        self._update_left()

    def watch_reward(self, value: float):
        try:
            self.query_one("#stats-right", Label).update(f"Rw: {value:.2f}")
        except Exception: 
            pass

    def _update_left(self):
        try:
            self.query_one("#stats-left", Label).update(f"Ep: {self.episode} | St: {self.step}")
        except Exception:
            pass

class BaseTaskTUI(ABC):
    """
    Abstract interface for Task TUIs.
    Maintains the structure of Header + Content.
    """
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.header = TaskHeader(task_name)

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
        # The main content container fills the rest of the screen
        container = Container(self.get_main_widget(), id="task-content")
        container.styles.height = "1fr"
        container.styles.align = ("center", "middle")
        yield container

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