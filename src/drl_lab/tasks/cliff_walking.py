import gymnasium as gym
import numpy as np
from .base import BaseTask

class CliffWalkingTask(BaseTask):
    def __init__(self):
        super().__init__("CliffWalking-v0")
        env = gym.make("CliffWalking-v0")
        self._action_size = int(env.action_space.n)
        # Observation space is Discrete(48)
        self._n_states = int(env.observation_space.n)
        env.close()

    def make_env(self, render_mode=None):
        return gym.make("CliffWalking-v0", render_mode=render_mode)

    @property
    def state_size(self) -> int:
        return self._n_states

    @property
    def action_size(self) -> int:
        return self._action_size

    def preprocess_state(self, state):
        # One-hot encoding
        # Ensure state is an integer
        if isinstance(state, (np.ndarray, list)):
            state = state[0] if len(state) > 0 else 0
        
        state = int(state)
        one_hot = np.zeros(self._n_states, dtype=np.float32)
        if 0 <= state < self._n_states:
            one_hot[state] = 1.0
        return one_hot
