from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import torch.nn as nn

from ..utils import Config
from .visual import BaseTaskTUI, DefaultTaskTUI

class BaseTask(ABC):
    """
    Abstract Base Class for Reinforcement Learning Tasks.
    Enforces strong typing and separation of concerns (Logic vs Visuals).
    """

    def __init__(self, name: str, config: Config | None = None):
        """
        Initialize the Task.
        
        Args:
            name: The name of the task.
            config: Optional configuration override. If None, uses default Config.
        """
        self.name: str = name
        self.config: Config = config if config else Config()
        self.config.env_name = name
        
        self._env: gym.Env | None = None
        self._tui: BaseTaskTUI | None = None

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
        Preprocesses the state into a format suitable for the agent.
        """
        return state

    # --- Hooks ---
    
    def pre_training(self) -> None:
        """
        Hook called before training starts.
        """
        pass

    def post_training(self) -> None:
        """
        Hook called after training finishes.
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

    def sync_data(self, data: dict[str, Any]) -> None:
        """
        Hook for syncing arbitrary data (e.g., custom logging metrics).
        """
        pass
