from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import gymnasium as gym
import torch.nn as nn

from ..utils import Config
from .visual import BaseTaskTUI, DefaultTaskTUI

class BaseTask(ABC):
    """
    Abstract Base Class for Reinforcement Learning Tasks.
    Enforces strong typing and separation of concerns (Logic vs Visuals).
    """

    def __init__(self, name: str, config: Optional[Config] = None):
        """
        Initialize the Task.
        
        Args:
            name: The name of the task.
            config: Optional configuration override. If None, uses default Config.
        """
        self.name: str = name
        self.config: Config = config if config else Config()
        self.config.env_name = name
        
        self._env: Optional[gym.Env] = None
        self._tui: Optional[BaseTaskTUI] = None

    @property
    def env(self) -> gym.Env:
        """
        Access the persistent gymnasium environment.
        Lazily initialized via `make_env` if not already set.
        """
        if self._env is None:
            self._env = self.get_env()
        return self._env

    @property
    def tui(self) -> BaseTaskTUI:
        """
        Access the TUI interface.
        Lazily initialized via `render` if not already set.
        """
        if self._tui is None:
            self._tui = self.render()
        return self._tui

    @abstractmethod
    def get_env(self) -> gym.Env:
        """
        Creates and returns the gymnasium environment instance.
        This should be called only once by the base class to populate self._env.
        """
        pass
    
    def render(self) -> BaseTaskTUI:
        """
        Returns the TUI interface for this task.
        Defaults to DefaultTaskTUI if not overridden.
        """
        return DefaultTaskTUI(self.name)

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

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Creates and returns the neural network model for this task."""
        pass

    def preprocess_state(self, state: Any) -> Any:
        """
        Preprocesses the state from the environment into a format suitable for the agent.
        Default implementation returns the state as is.
        """
        return state

    # --- Hooks ---
    
    def pre_training(self) -> None:
        """
        Hook called before training starts.
        Useful for initializing resources, logging, or setting up curriculum.
        """
        pass

    def post_training(self) -> None:
        """
        Hook called after training finishes.
        Useful for cleaning up resources or saving final artifacts.
        """
        if self._env:
            self._env.close()

    def pre_episode(self, episode: int) -> None:
        """
        Hook called before an episode starts.
        """
        pass

    def post_episode(self, episode: int, reward: float) -> None:
        """
        Hook called after an episode finishes.
        """
        pass

    def sync_data(self, data: Dict[str, Any]) -> None:
        """
        Hook for syncing arbitrary data (e.g., custom logging metrics).
        """
        pass