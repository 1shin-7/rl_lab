from typing import Any, Dict, Optional
import numpy as np
from textual.widgets import Static
from textual.containers import Container
from textual.widget import Widget
from pathlib import Path

from ..visual import BaseTaskTUI

class CliffWalkingWidget(Container):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.agent_idx = -1
        self.total_cells = 48 

    def on_mount(self):
        # Load CSS
        css_path = Path(__file__).parent / "styles.tcss"
        with open(css_path) as f:
            self.app.stylesheet.add_source(f.read())

    def compose(self):
        with Container(id="grid"):
            for i in range(self.total_cells):
                classes = "cell"
                
                # Bottom row: 36 (Start), 37-46 (Cliff), 47 (Goal)
                # Cliff is 37-46. 47 is the Goal and should be white.
                if 37 <= i <= 46:
                    classes += " cliff"
                
                cell = Static(str(i), id=f"cell-{i}", classes=classes)
                self.cells.append(cell)
                yield cell

    def update_agent(self, idx: int):
        if idx != self.agent_idx:
            # Clear old agent class
            if self.agent_idx != -1 and 0 <= self.agent_idx < self.total_cells:
                self.cells[self.agent_idx].remove_class("agent")

            # Set new agent class
            self.agent_idx = idx
            if 0 <= idx < self.total_cells:
                self.cells[idx].add_class("agent")

class CliffWalkingTUI(BaseTaskTUI):
    def __init__(self, task_name: str):
        super().__init__(task_name)
        self.widget = CliffWalkingWidget()

    def get_main_widget(self) -> Widget:
        return self.widget

    def update_state(self, state: Any, info: Optional[Dict[str, Any]] = None):
        idx = -1
        if isinstance(state, (np.ndarray, list)):
            if len(state) == 48: 
                idx = np.argmax(state)
            else:
                idx = int(state[0]) if len(state) > 0 else 0
        else:
            idx = int(state)
            
        self.widget.update_agent(idx)