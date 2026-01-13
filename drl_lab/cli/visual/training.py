import contextlib
from collections import deque
from typing import Any

import torch
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import RichLog
from textual.worker import get_current_worker

from ...tasks import get_task
from ...train import Trainer, TrainingCallbacks
from ...utils import logger, setup_logger

class TrainingAppCallback(TrainingCallbacks):
    def __init__(self, app: "VisualTrainApp"):
        self.app = app
        self._current_episode = 0

    def on_step(
        self, step: int, state: Any, reward: float, info: dict[str, Any]
    ) -> None:
        if self.app.is_running:
            self.app.update_task_view(state, info)
            self.app.update_header(self._current_episode, step, reward)

    def on_episode_end(self, episode: int, steps: int, reward: float) -> None:
        self._current_episode = episode + 1
        if self.app.is_running:
            self.app.update_header(episode, steps, reward)

class VisualTrainApp(App):
    CSS_PATH = "training.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit")
    ]

    def __init__(
        self, 
        task_name: str, 
        episodes: int, 
        output_path: str = None, 
        log_lines: int = 5
    ):
        super().__init__()
        self.task_name = task_name
        self.episodes = episodes
        self.output_path = output_path
        self.log_lines = log_lines
        
        self.rl_task = get_task(task_name)
        self.tui = self.rl_task.render()
        self._worker = None
        self.recent_records = deque(maxlen=20)

    def compose(self) -> ComposeResult:
        yield from self.tui.compose_view()
        log_widget = RichLog(id="log-output", highlight=False, markup=False)
        log_widget.border_title = "Training Logs"
        h = self.log_lines + 2
        log_widget.styles.height = h
        log_widget.styles.min_height = h
        log_widget.styles.max_height = h
        yield log_widget

    def on_mount(self) -> None:
        setup_logger(sink=self.sink_log)
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        self.tui.header.device = device
        self._worker = self.run_worker(self.training_loop, exclusive=True, thread=True)

    def action_quit(self) -> None:
        if self._worker:
            self._worker.cancel()
            logger.warning("Stop signal received. Finishing up...")
        else:
            self.exit()

    def sink_log(self, message):
        record = message.record
        self.recent_records.append(record)
        if self.is_running:
            with contextlib.suppress(RuntimeError):
                self.call_from_thread(self.write_log, record)

    def write_log(self, record):
        with contextlib.suppress(Exception):
            log_widget = self.query_one("#log-output", RichLog)
            time_str = record["time"].strftime("%H:%M:%S")
            level_name = record["level"].name
            msg_text = record["message"]
            
            level_color = "white"
            if level_name == "INFO": 
                level_color = "magenta"
            elif level_name == "WARNING": 
                level_color = "yellow"
            elif level_name == "ERROR": 
                level_color = "red"
            elif level_name == "SUCCESS": 
                level_color = "green"
            elif level_name == "DEBUG": 
                level_color = "cyan"
            
            text = Text()
            text.append(f"{time_str} | ", style="dim")
            text.append(f"{level_name:<7}", style=f"bold {level_color}")
            text.append(f" | {msg_text}")
            log_widget.write(text)

    def update_task_view(self, state, info):
        with contextlib.suppress(RuntimeError):
            self.call_from_thread(self.tui.update_state, state, info)

    def update_header(self, episode, steps, reward):
        with contextlib.suppress(RuntimeError):
            self.call_from_thread(self.tui.update_stats, episode, steps, reward)

    def training_loop(self):
        worker = get_current_worker()
        callbacks = TrainingAppCallback(self)
        trainer = Trainer(
            task_name=self.task_name, 
            episodes=self.episodes, 
            output_path=self.output_path,
            callbacks=callbacks,
            should_stop=lambda: worker.is_cancelled
        )
        trainer.run()
        if worker.is_cancelled:
            logger.info("Training stopped by user.")
        else:
            logger.info("Training finished.")
        self.call_from_thread(self.exit)