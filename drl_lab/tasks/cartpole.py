import gymnasium as gym
import numpy as np
import torch.nn as nn
from rich.text import Text
from rich.panel import Panel
from textual.widget import Widget
from textual.widgets import Static
from typing import Any, Dict, Optional

from .base import BaseTask
from .visual import BaseTaskTUI
from ..models import SimpleMLP

class CartPoleWidget(Static):
    def __init__(self):
        super().__init__()
        
    def _braille_line(self, x0, y0, x1, y1, width, height, canvas):
        """Draws a line using Bresenham's algorithm on a virtual 2x4 dot grid per char."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            # Set pixel at (x0, y0)
            char_x = x0 // 2
            char_y = y0 // 4
            
            if 0 <= char_x < width and 0 <= char_y < height:
                dot_x = x0 % 2
                dot_y = y0 % 4
                
                mapping = [
                    [0x01, 0x08],
                    [0x02, 0x10],
                    [0x04, 0x20],
                    [0x40, 0x80]
                ]
                canvas[char_y][char_x] |= mapping[dot_y][dot_x]
            
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def render_state(self, state):
        # State: [cart_position, cart_velocity, pole_angle, pole_velocity]
        if isinstance(state, (list, np.ndarray)):
             if isinstance(state, tuple): 
                 state = state[0]
             state = np.array(state).flatten()
        
        x = state[0]
        theta = state[2]
        
        W, H = 60, 10
        vW, vH = W * 2, H * 4
        canvas = [[0 for _ in range(W)] for _ in range(H)]
        
        scale_x = vW / 4.8
        center_x = vW / 2
        
        cart_pixel_x = int(center_x + x * scale_x)
        cart_pixel_y = vH - 5 
        
        # Draw track
        track_y = cart_pixel_y + 2
        self._braille_line(0, track_y, vW-1, track_y, W, H, canvas)
        
        # Draw Cart
        for dx in range(-2, 3):
             for dy in range(-2, 3):
                 self._braille_line(cart_pixel_x+dx, cart_pixel_y+dy, cart_pixel_x+dx, cart_pixel_y+dy, W, H, canvas)
        
        # Draw Pole
        pole_len = 30
        tip_x = cart_pixel_x + int(pole_len * np.sin(theta))
        tip_y = cart_pixel_y - int(pole_len * np.cos(theta))
        
        self._braille_line(cart_pixel_x, cart_pixel_y, tip_x, tip_y, W, H, canvas)
        
        lines = []
        for row in canvas:
            line_str = ""
            for val in row:
                line_str += chr(0x2800 + val)
            lines.append(line_str)
            
        display = Text()
        display.append(f"Cart X: {x:.2f} | Angle: {theta:.2f}\n", style="bold")
        display.append("\n".join(lines), style="green")
        
        self.update(Panel(display, title="CartPole (Braille)", expand=False))

class CartPoleTUI(BaseTaskTUI):
    def __init__(self, task_name: str):
        super().__init__(task_name)
        self.widget = CartPoleWidget()

    def get_main_widget(self) -> Widget:
        return self.widget

    def update_state(self, state: Any, info: Optional[Dict[str, Any]] = None):
        self.widget.render_state(state)

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
