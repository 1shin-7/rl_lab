import gymnasium as gym
import torch.nn as nn
from gymnasium import Wrapper

from ...models import DuelingMLP
from ..base import BaseTask
from ..visual import BaseTaskTUI
from .tui import CartPoleTUI

class CenteredRewardWrapper(Wrapper):
    """
    Modifies CartPole reward to penalize distance from center.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # obs: [x, x_dot, theta, theta_dot]
        x = obs[0]
        x_threshold = self.env.unwrapped.x_threshold
        
        dist_penalty = abs(x) / x_threshold
        shaped_reward = reward - (dist_penalty * 0.5)
        
        return obs, shaped_reward, terminated, truncated, info

class CartPoleTask(BaseTask):
    def __init__(self, config=None):
        super().__init__("CartPole-v1", config)
        temp_env = gym.make("CartPole-v1")
        self._state_size = int(temp_env.observation_space.shape[0])
        self._action_size = int(temp_env.action_space.n)
        temp_env.close()

    def get_env(self) -> gym.Env:
        env = gym.make("CartPole-v1")
        return CenteredRewardWrapper(env)

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def action_size(self) -> int:
        return self._action_size

    def create_model(self) -> nn.Module:
        return DuelingMLP(self.state_size, self.action_size)
    
    def render(self) -> BaseTaskTUI:
        return CartPoleTUI(self.name)
