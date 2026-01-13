import gymnasium as gym
import torch.nn as nn

from ..base import BaseTask
from ..visual import BaseTaskTUI
from ...models import SimpleMLP
from .tui import CartPoleTUI

class CartPoleTask(BaseTask):
    def __init__(self, config=None):
        super().__init__("CartPole-v1", config)
        temp_env = gym.make("CartPole-v1")
        self._state_size = int(temp_env.observation_space.shape[0])
        self._action_size = int(temp_env.action_space.n)
        temp_env.close()

    def get_env(self) -> gym.Env:
        return gym.make("CartPole-v1")

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def action_size(self) -> int:
        return self._action_size

    def create_model(self) -> nn.Module:
        return SimpleMLP(self.state_size, self.action_size)
    
    def render(self) -> BaseTaskTUI:
        return CartPoleTUI(self.name)
