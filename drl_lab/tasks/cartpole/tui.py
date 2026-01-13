from typing import Any, Dict, Optional
import numpy as np
import math
from pathlib import Path
from textual.widgets import Label, Static
from textual.containers import Container
from textual.widget import Widget
from rich.text import Text

from ..visual import BaseTaskTUI

class BrailleCanvas:
    """Helper to draw on a virtual dot matrix using Braille characters."""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # 2x4 dots per character
        self.v_width = width * 2
        self.v_height = height * 4
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def clear(self):
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]

    def set_pixel(self, x: int, y: int):
        if not (0 <= x < self.v_width and 0 <= y < self.v_height):
            return
        
        char_x = x // 2
        char_y = y // 4
        
        dot_x = x % 2
        dot_y = y % 4
        
        # Braille dot mapping
        mask = 0
        if dot_x == 0:
            if dot_y == 0:
                mask = 0x01
            elif dot_y == 1:
                mask = 0x02
            elif dot_y == 2:
                mask = 0x04
            elif dot_y == 3:
                mask = 0x40
        else:
            if dot_y == 0:
                mask = 0x08
            elif dot_y == 1:
                mask = 0x10
            elif dot_y == 2:
                mask = 0x20
            elif dot_y == 3:
                mask = 0x80
            
        self.grid[char_y][char_x] |= mask

    def draw_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            self.set_pixel(x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def render(self) -> Text:
        lines = []
        for row in self.grid:
            line_str = ""
            for val in row:
                line_str += chr(0x2800 + val)
            lines.append(line_str)
        return Text("\n".join(lines))

class CartPoleWidget(Container):
    def __init__(self):
        super().__init__()
        self.canvas_width = 60
        self.canvas_height = 10
        self.braille = BrailleCanvas(self.canvas_width, self.canvas_height)
        self.info_label = Label("Initializing...", classes="info-text")
        self.canvas_display = Static(id="canvas")

    def compose(self):
        yield self.info_label
        yield self.canvas_display

    def on_mount(self):
        css_path = Path(__file__).parent / "styles.tcss"
        with open(css_path) as f:
            self.app.stylesheet.add_source(f.read())

    def update_state(self, state):
        # State: [cart_position, cart_velocity, pole_angle, pole_velocity]
        if isinstance(state, (list, np.ndarray)):
             if isinstance(state, tuple):
                 state = state[0]
             state = np.array(state).flatten()
        
        x = state[0]
        theta = state[2]
        
        # Update Info Text
        self.info_label.update(f"Cart X: {x:.2f} | Angle: {theta:.2f} rad")

        # Draw
        self.braille.clear()
        
        # Virtual dimensions
        vW = self.braille.v_width
        vH = self.braille.v_height
        
        # Scale: viewport is approx -2.4 to 2.4 => width 4.8
        # vW pixels represent 4.8 units
        scale_x = vW / 4.8
        center_x = vW / 2
        
        cart_pixel_x = int(center_x + x * scale_x)
        cart_pixel_y = vH - 10 # Base line
        
        # 1. Draw Track
        self.braille.draw_line(0, cart_pixel_y, vW-1, cart_pixel_y)
        
        # 2. Draw Cart (Box)
        w_cart = 6
        h_cart = 4
        x_left = cart_pixel_x - w_cart // 2
        x_right = cart_pixel_x + w_cart // 2
        y_top = cart_pixel_y - h_cart
        y_bot = cart_pixel_y
        
        # Top/Bottom
        self.braille.draw_line(x_left, y_top, x_right, y_top)
        self.braille.draw_line(x_left, y_bot, x_right, y_bot)
        # Sides
        self.braille.draw_line(x_left, y_top, x_left, y_bot)
        self.braille.draw_line(x_right, y_top, x_right, y_bot)
        
        # 3. Draw Pole
        pole_len = 35
        # Tip position: theta=0 is UP. 
        # sin(theta) gives x component (right), cos(theta) gives y component (up)
        # screen y is down, so minus cos.
        tip_x = cart_pixel_x + int(pole_len * math.sin(theta))
        tip_y = cart_pixel_y - int(pole_len * math.cos(theta))
        
        # To make it "bold", draw two lines slightly offset
        self.braille.draw_line(cart_pixel_x, cart_pixel_y, tip_x, tip_y)
        # Offset 1 pixel to right for thickness
        self.braille.draw_line(cart_pixel_x+1, cart_pixel_y, tip_x+1, tip_y)

        # Render
        self.canvas_display.update(self.braille.render())

class CartPoleTUI(BaseTaskTUI):
    def __init__(self, task_name: str):
        super().__init__(task_name)
        self.widget = CartPoleWidget()

    def get_main_widget(self) -> Widget:
        return self.widget

    def update_state(self, state: Any, info: Optional[Dict[str, Any]] = None):
        self.widget.update_state(state)