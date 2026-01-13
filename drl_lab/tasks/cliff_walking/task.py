import gymnasium as gym
import numpy as np
import torch.nn as nn
from typing import Any

from ..base import BaseTask
from ..visual import BaseTaskTUI
from ...models import DuelingMLP
from .tui import CliffWalkingTUI

class CliffWalkingTask(BaseTask):
    def __init__(self, config=None):
        super().__init__("CliffWalking-v1", config)
        # Pre-calculate sizes without keeping env open
        temp_env = gym.make("CliffWalking-v1")
        self._action_size = int(temp_env.action_space.n)
        self._n_states = int(temp_env.observation_space.n)
        temp_env.close()

    def get_env(self) -> gym.Env:
        return gym.make("CliffWalking-v1")

    @property
    def state_size(self) -> int:
        return self._n_states

    @property
    def action_size(self) -> int:
        return self._action_size

    def create_model(self) -> nn.Module:
        return DuelingMLP(self.state_size, self.action_size)

    def preprocess_state(self, state: Any) -> Any:
        if isinstance(state, (np.ndarray, list)):
            state = state[0] if len(state) > 0 else 0
        
        state = int(state)
        one_hot = np.zeros(self._n_states, dtype=np.float32)
        if 0 <= state < self._n_states:
            one_hot[state] = 1.0
        return one_hot

    def render(self) -> BaseTaskTUI:
        return CliffWalkingTUI(self.name)