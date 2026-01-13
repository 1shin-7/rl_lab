from typing import Any

import numpy as np
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import Static

from ...utils import paths
from ..visual import BaseTaskTUI

class CliffWalkingWidget(Container):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.agent_idx = -1
        self.total_cells = 48 

    def on_mount(self):
        css_path = (
            paths.PROJECT_ROOT / "drl_lab" / "tasks" / 
            "cliff_walking" / "styles.tcss"
        )
        with css_path.open() as f:
            self.app.stylesheet.add_source(f.read())

    def compose(self):
        with Container(id="grid"):
            for i in range(self.total_cells):
                classes = "cell"
                if 37 <= i <= 46:
                    classes += " cliff"
                cell = Static(str(i), id=f"cell-{i}", classes=classes)
                self.cells.append(cell)
                yield cell

    def update_agent(self, idx: int):
        if idx != self.agent_idx:
            if self.agent_idx != -1 and 0 <= self.agent_idx < self.total_cells:
                self.cells[self.agent_idx].remove_class("agent")
            self.agent_idx = idx
            if 0 <= idx < self.total_cells:
                self.cells[idx].add_class("agent")

class CliffWalkingTUI(BaseTaskTUI):
    def __init__(self, task_name: str):
        super().__init__(task_name)
        self.widget = CliffWalkingWidget()

    def get_main_widget(self) -> Widget:
        return self.widget

    def update_state(self, state: Any, info: dict[str, Any] | None = None):
        idx = -1
        if isinstance(state, (np.ndarray, list)):
            if len(state) == 48: 
                idx = int(np.argmax(state))
            else:
                idx = int(state[0]) if len(state) > 0 else 0
        else:
            idx = int(state)
        self.widget.update_agent(idx)
