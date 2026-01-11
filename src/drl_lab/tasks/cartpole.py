import gymnasium as gym
from .base import BaseTask

class CartPoleTask(BaseTask):
    def __init__(self):
        super().__init__("CartPole-v1")
        # We need a dummy env to get spaces, or just hardcode if known.
        # Better to instantiate once to check.
        env = gym.make("CartPole-v1")
        self._state_size = int(env.observation_space.shape[0])
        self._action_size = int(env.action_space.n)
        env.close()

    def make_env(self, render_mode=None):
        return gym.make("CartPole-v1", render_mode=render_mode)

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def action_size(self) -> int:
        return self._action_size
