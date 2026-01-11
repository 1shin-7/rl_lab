import gymnasium as gym
import numpy as np
from textual.widgets import Placeholder
from textual.containers import Container
from .base import BaseTask

class CliffWalkingWidget(Container):
    CSS = """
    CliffWalkingWidget {
        layout: grid;
        grid-size: 12 4;
        grid-gutter: 1;
        width: auto;
        height: auto;
        border: solid $accent;
        align: center middle;
        background: $surface;
    }

    Placeholder {
        width: 7;  /* Approx square-ish in terminal (2:1 font ratio typically) */
        height: 3; 
        background: $surface-lighten-1;
        color: $text;
    }

    /* State modifiers */
    .start {
        background: $warning;
        color: black;
    }
    
    .goal {
        background: $success;
        color: black;
    }
    
    .cliff {
        background: $error;
    }
    
    .ground {
        background: white;
        color: black;
    }

    .agent {
        background: blue;
        color: white;
    }
    """

    def __init__(self):
        super().__init__()
        self.placeholders = []
        self.agent_idx = -1
        self.total_cells = 48 

    def compose(self):
        for i in range(self.total_cells):
            classes = ""
            
            r = i // 12
            c = i % 12
            
            if i == 36: # Start
                classes = "start"
            elif i == 47: # Goal
                classes = "goal"
            elif r == 3 and 1 <= c <= 10: # Cliff
                classes = "cliff"
            else:
                classes = "ground"
            
            # Label is the index
            p = Placeholder(label=str(i), id=f"p-{i}", classes=classes)
            self.placeholders.append(p)
            yield p

    def update_state(self, state, info=None):
        idx = -1
        if isinstance(state, (np.ndarray, list)):
            if len(state) == 48: 
                idx = np.argmax(state)
            else:
                idx = int(state[0]) if len(state) > 0 else 0
        else:
            idx = int(state)
            
        if idx != self.agent_idx:
            # Clear old agent class
            if self.agent_idx != -1 and 0 <= self.agent_idx < self.total_cells:
                self.placeholders[self.agent_idx].remove_class("agent")

            # Set new agent class
            self.agent_idx = idx
            if 0 <= idx < self.total_cells:
                self.placeholders[idx].add_class("agent")

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
    
    def create_visual_widget(self):
        return CliffWalkingWidget()

    def render_tui(self, state, info=None):
        return str(state)