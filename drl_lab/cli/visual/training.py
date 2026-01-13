from textual.app import App, ComposeResult
from textual.widgets import RichLog
from loguru import logger
from typing import Any, Dict

from ...train import Trainer, TrainingCallbacks
from ...tasks import get_task

class TrainingAppCallback(TrainingCallbacks):
    def __init__(self, app: "VisualTrainApp"):
        self.app = app

    def on_step(self, step: int, state: Any, info: Dict[str, Any]) -> None:
        self.app.update_task_view(state, info)

    def on_episode_end(self, episode: int, steps: int, reward: float) -> None:
        self.app.update_header(episode, steps, reward)

class VisualTrainApp(App):
    CSS_PATH = "training.tcss"

    def __init__(self, task_name: str, episodes: int, output_path: str = None, log_lines: int = 15):
        super().__init__()
        self.task_name = task_name
        self.episodes = episodes
        self.output_path = output_path
        self.log_lines = log_lines
        
        # FIX: Rename task -> rl_task to avoid conflict with textual.App.task
        self.rl_task = get_task(task_name)
        self.tui = self.rl_task.render()

    def compose(self) -> ComposeResult:
        # Mount Task TUI (Header + Content)
        # TUI.compose_view returns Header (dock top) and Content (1fr)
        yield from self.tui.compose_view()
        
        # Log Output at Bottom
        log_widget = RichLog(id="log-output", highlight=True, markup=True)
        # Calculate height: lines + 2 for borders
        log_widget.styles.height = self.log_lines + 2
        yield log_widget

    def on_mount(self) -> None:
        # Redirect loguru
        logger.remove()
        logger.add(self.sink_log, format="{time:HH:mm:ss} | {level} | {message}")
        
        # Start training in background
        self.run_worker(self.training_loop, exclusive=True, thread=True)

    def sink_log(self, message):
        self.call_from_thread(self.write_log, message)

    def write_log(self, message):
        try:
            log_widget = self.query_one("#log-output", RichLog)
            log_widget.write(message)
        except Exception:
            pass

    def update_task_view(self, state, info):
        self.call_from_thread(self.tui.update_state, state, info)

    def update_header(self, episode, steps, reward):
        self.call_from_thread(self.tui.update_stats, episode, steps, reward)

    def training_loop(self):
        callbacks = TrainingAppCallback(self)
        trainer = Trainer(
            task_name=self.task_name, 
            episodes=self.episodes, 
            output_path=self.output_path,
            callbacks=callbacks
        )
        trainer.run()
        self.exit()