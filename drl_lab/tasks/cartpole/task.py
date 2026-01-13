import gymnasium as gym
import torch.nn as nn
from gymnasium import Wrapper

from ..base import BaseTask
from ..visual import BaseTaskTUI
from ...models import SimpleMLP
from .tui import CartPoleTUI

class CenteredRewardWrapper(Wrapper):
    """
    Modifies CartPole reward to penalize distance from center.
    Standard CartPole-v1 gives +1 for every step alive.
    This wrapper adds a penalty proportional to |x|/x_threshold.
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # obs: [x, x_dot, theta, theta_dot]
        x = obs[0]
        x_threshold = self.env.unwrapped.x_threshold
        
        # Calculate penalty based on distance from center (0 to 1.0)
        # We want the agent to stay alive (+1) BUT also prefer center.
        # Let's subtract up to 0.5 depending on distance.
        dist_penalty = abs(x) / x_threshold
        
        # New reward: 1.0 - (0.0 to 1.0) * 0.5
        # At center: 1.0
        # At edge: 0.5
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
        return SimpleMLP(self.state_size, self.action_size)
    
    def render(self) -> BaseTaskTUI:
        return CartPoleTUI(self.name)