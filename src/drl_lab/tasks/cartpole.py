import gymnasium as gym
import numpy as np
from rich.text import Text
from rich.panel import Panel
from .base import BaseTask

class CartPoleTask(BaseTask):
    def __init__(self):
        super().__init__("CartPole-v1")
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

    def render_tui(self, state, info=None):
        # State: [cart_position, cart_velocity, pole_angle, pole_velocity]
        if isinstance(state, (list, np.ndarray)):
            # If batch or tuple, extract
             if isinstance(state, tuple): state = state[0]
             state = np.array(state).flatten()
        
        x = state[0]
        theta = state[2]
        
        # Visualize cart position on a track
        track_width = 50
        track_range = 4.8 # approx range [-2.4, 2.4]
        
        # Normalize x to [0, track_width]
        norm_x = (x + 2.4) / 4.8
        pos = int(norm_x * track_width)
        pos = max(0, min(track_width - 1, pos))
        
        # Pole Drawing
        pole_height = 5
        # Initialize canvas with spaces
        canvas = [[" " for _ in range(track_width)] for _ in range(pole_height)]
        
        # Scale for horizontal deflection: 
        # A 45 degree tilt (approx 0.8 rad) should probably deflect significantly but not break the screen.
        # Let's say 1 unit of height corresponds to 2 units of width for 45 deg?
        # offset = height * tan(theta) * scale
        scale = 3.0 
        
        for r in range(pole_height):
            # Distance from base (r=0 is bottom of pole, strictly speaking base is at cart)
            # visual row 0 is just above cart
            dist = r + 1
            col_offset = int(dist * np.tan(theta) * scale)
            col = pos + col_offset
            
            if 0 <= col < track_width:
                # Determine char based on local slope or just position
                if col_offset == 0:
                    char = "|"
                elif col_offset > 0:
                    char = "/"
                else:
                    char = "\\"
                
                # Canvas row: 0 is top, pole_height-1 is bottom
                # We want r=0 to be bottom (just above cart), so canvas row is pole_height - 1 - r
                canvas_row = pole_height - 1 - r
                canvas[canvas_row][col] = char

        # Combine lines
        lines = ["".join(row) for row in canvas]
        
        # Track line
        track_line = ["-"] * track_width
        if 0 <= pos < track_width:
            track_line[pos] = "O" # Cart
        lines.append("".join(track_line))
        
        # Construct display
        display = Text()
        display.append(f"Cart X: {x:.2f} | Angle: {theta:.2f}\n", style="bold")
        display.append("\n")
        
        # Color the pole/cart
        grid_str = "\n".join(lines)
        display.append(grid_str, style="bold cyan")
        
        return Panel(display, title="CartPole State", expand=False)
