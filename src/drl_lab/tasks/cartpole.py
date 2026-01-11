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

    def _braille_line(self, x0, y0, x1, y1, width, height, canvas):
        """Draws a line using Bresenham's algorithm on a virtual 2x4 dot grid per char."""
        # Canvas dimensions in characters: (width, height)
        # Virtual pixel dimensions: (width*2, height*4)
        
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
                
                # Braille dot mapping
                # 0x01 (0,0)  0x08 (1,0)
                # 0x02 (0,1)  0x10 (1,1)
                # 0x04 (0,2)  0x20 (1,2)
                # 0x40 (0,3)  0x80 (1,3)
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

    def render_tui(self, state, info=None):
        # State: [cart_position, cart_velocity, pole_angle, pole_velocity]
        if isinstance(state, (list, np.ndarray)):
             if isinstance(state, tuple): state = state[0]
             state = np.array(state).flatten()
        
        x = state[0]
        theta = state[2]
        
        # Dimensions (characters)
        W, H = 60, 10
        
        # Virtual dimensions (dots)
        vW, vH = W * 2, H * 4
        
        # Initialize canvas (integers representing unicode offset from 0x2800)
        canvas = [[0 for _ in range(W)] for _ in range(H)]
        
        # Map world to virtual pixels
        # X range [-2.4, 2.4] -> [0, vW]
        # X Center at vW / 2
        # Scale: vW corresponds to approx 4.8 units width? 
        # Let's verify standard range. CartPole triggers done at 2.4.
        
        scale_x = vW / 4.8
        center_x = vW / 2
        
        cart_pixel_x = int(center_x + x * scale_x)
        cart_pixel_y = vH - 5 # slightly up from bottom
        
        # Draw track
        track_y = cart_pixel_y + 2
        self._braille_line(0, track_y, vW-1, track_y, W, H, canvas)
        
        # Draw Cart (simple box around center)
        # Just a small blob
        for dx in range(-2, 3):
             for dy in range(-2, 3):
                 self._braille_line(cart_pixel_x+dx, cart_pixel_y+dy, cart_pixel_x+dx, cart_pixel_y+dy, W, H, canvas)
        
        # Draw Pole
        pole_len = 30 # dots
        # Tip position
        # theta 0 is up. +theta is right.
        tip_x = cart_pixel_x + int(pole_len * np.sin(theta))
        tip_y = cart_pixel_y - int(pole_len * np.cos(theta))
        
        self._braille_line(cart_pixel_x, cart_pixel_y, tip_x, tip_y, W, H, canvas)
        
        # Convert to string
        lines = []
        for row in canvas:
            line_str = ""
            for val in row:
                line_str += chr(0x2800 + val)
            lines.append(line_str)
            
        display = Text()
        display.append(f"Cart X: {x:.2f} | Angle: {theta:.2f}\n", style="bold")
        display.append("\n".join(lines), style="green")
        
        return Panel(display, title="CartPole (Braille)", expand=False)