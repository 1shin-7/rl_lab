from abc import ABC, abstractmethod
import numpy as np

class BaseTask(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def make_env(self, render_mode=None):
        """Creates and returns the gymnasium environment."""
        pass

    @property
    @abstractmethod
    def state_size(self) -> int:
        """Returns the dimension of the state vector."""
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Returns the number of possible actions."""
        pass

    def preprocess_state(self, state):
        """
        Preprocesses the state from the environment into a format suitable for the agent.
        Default implementation returns the state as is.
        """
        return state

    def render_tui(self, state, info=None):
        """
        Returns a renderable (string, Rich object, etc.) for the TUI.
        """
        return f"State: {state}\nInfo: {info}"

    def create_visual_widget(self):
        """
        Optional: Returns a Textual Widget instance for more complex visualizations.
        If implemented, the App will use this widget and call its `update_state(state, info)` method
        instead of `render_tui`.
        """
        return None