from textual.app import App, ComposeResult
from textual.widgets import RichLog
from textual.worker import get_current_worker
from loguru import logger
import torch
from typing import Any, Dict

from ...train import Trainer, TrainingCallbacks
from ...tasks import get_task

class TrainingAppCallback(TrainingCallbacks):
    def __init__(self, app: "VisualTrainApp"):
        self.app = app

    def on_step(self, step: int, state: Any, info: Dict[str, Any]) -> None:
        if self.app.is_running:
            self.app.update_task_view(state, info)

    def on_episode_end(self, episode: int, steps: int, reward: float) -> None:
        if self.app.is_running:
            self.app.update_header(episode, steps, reward)

class VisualTrainApp(App):
    CSS_PATH = "training.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit")
    ]

    def __init__(self, task_name: str, episodes: int, output_path: str = None, log_lines: int = 5):
        super().__init__()
        self.task_name = task_name
        self.episodes = episodes
        self.output_path = output_path
        self.log_lines = log_lines
        
        self.rl_task = get_task(task_name)
        self.tui = self.rl_task.render()
        self._worker = None

    def compose(self) -> ComposeResult:
        yield from self.tui.compose_view()
        
        log_widget = RichLog(id="log-output", highlight=True, markup=True)
        log_widget.border_title = "Training Logs"
        
        h = self.log_lines + 2
        log_widget.styles.height = h
        log_widget.styles.min_height = h
        log_widget.styles.max_height = h
        
        yield log_widget

    def on_mount(self) -> None:
        logger.remove()
        logger.add(self.sink_log, format="{time:HH:mm:ss} | {level} | {message}")
        
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        self.tui.header.device = device

        self._worker = self.run_worker(self.training_loop, exclusive=True, thread=True)

    def action_quit(self) -> None:
        """Handle quit request gracefully."""
        if self._worker:
            self._worker.cancel()
        self.exit()

    def sink_log(self, message):
        """Safely sink logs to the main thread."""
        try:
            if self.is_running:
                self.call_from_thread(self.write_log, message)
        except RuntimeError:
            # App is closing or closed, ignore log
            pass

    def write_log(self, message):
        try:
            log_widget = self.query_one("#log-output", RichLog)
            log_widget.write(message)
        except Exception:
            pass

    def update_task_view(self, state, info):
        try:
            self.call_from_thread(self.tui.update_state, state, info)
        except RuntimeError:
            pass

    def update_header(self, episode, steps, reward):
        try:
            self.call_from_thread(self.tui.update_stats, episode, steps, reward)
        except RuntimeError:
            pass

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
        
        # We don't catch exceptions here, let them bubble or handle inside trainer
        trainer.run()
        
        # Only exit if we finished naturally, not if we are being cancelled
        if not worker.is_cancelled:
            self.call_from_thread(self.exit)
