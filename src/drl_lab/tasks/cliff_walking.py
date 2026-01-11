import gymnasium as gym
import numpy as np
from rich.table import Table
from rich.text import Text
from rich import box
from .base import BaseTask

class CliffWalkingTask(BaseTask):
    def __init__(self):
        super().__init__("CliffWalking-v1")
        env = gym.make("CliffWalking-v1")
        self._action_size = int(env.action_space.n)
        self._n_states = int(env.observation_space.n)
        env.close()

    def make_env(self, render_mode=None):
        return gym.make("CliffWalking-v1", render_mode=render_mode)

    @property
    def state_size(self) -> int:
        return self._n_states

    @property
    def action_size(self) -> int:
        return self._action_size

    def preprocess_state(self, state):
        if isinstance(state, (np.ndarray, list)):
            state = state[0] if len(state) > 0 else 0
        
        state = int(state)
        one_hot = np.zeros(self._n_states, dtype=np.float32)
        if 0 <= state < self._n_states:
            one_hot[state] = 1.0
        return one_hot

    def render_tui(self, state, info=None):
        # Determine agent position
        agent_idx = -1
        if isinstance(state, (np.ndarray, list)):
            if len(state) == self._n_states: # One-hot
                agent_idx = np.argmax(state)
            else:
                agent_idx = int(state[0]) if len(state) > 0 else 0
        else:
             agent_idx = int(state)
             
        rows = 4
        cols = 12
        
        # Create a table for grid layout
        table = Table(show_header=False, show_edge=True, box=box.SQUARE, padding=0)
        for _ in range(cols):
            table.add_column(width=4, justify="center") # Width 4 for "sqaure-ish" look with text

        for r in range(rows):
            row_cells = []
            for c in range(cols):
                idx = r * cols + c
                
                # Determine cell type
                style = "on black"
                char = "    "
                
                if idx == 36: # Start
                    style = "on yellow"
                elif idx == 47: # Goal
                    style = "on green"
                elif r == 3 and 1 <= c <= 10: # Cliff
                    style = "on red"
                else:
                    style = "on white" # Ground
                
                if idx == agent_idx:
                    style = "on blue"
                    char = " 馃槆 " # Agent icon

                row_cells.append(Text(char, style=style))
            
            table.add_row(*row_cells)
            
        return table