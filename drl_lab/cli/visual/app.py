from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Footer, Label
from textual.reactive import reactive
from textual.worker import get_current_worker
from loguru import logger
import torch
import time

from ...tasks import get_task
from ...agent import BaseDQNAgent

class VisualApp(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: auto 1fr auto;
    }
    
    #app-header {
        dock: top;
        height: 3;
        width: 100%;
        background: $primary-background;
        border-bottom: solid $accent;
        layout: horizontal;
    }
    
    #status-label {
        width: 1fr;
        content-align: left middle;
        padding-left: 2;
        text-style: bold;
    }
    
    #device-label {
        width: auto;
        content-align: right middle;
        padding-right: 2;
        color: $accent;
    }

    #game-container {
        width: 100%;
        height: 100%;
        align: center middle;
    }
    
    #log-line {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
    }
    """
    
    BINDINGS = [("q", "quit", "Quit")]

    status_text = reactive("Ready")
    device_text = reactive("CPU")
    last_log = reactive("")

    def __init__(self, task_name: str, weight_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.task_name = task_name
        self.weight_path = weight_path
        # Initialize task early to get TUI
        self.task = get_task(task_name)
        self.tui = self.task.render()

    def compose(self) -> ComposeResult:
        with Horizontal(id="app-header"):
            yield Label(self.status_text, id="status-label")
            yield Label(f"Device: {self.device_text}", id="device-label")
        
        # Mount the Task's TUI
        # The TUI itself returns a list of widgets (Header + Content)
        # We wrap it in a container
        with Container(id="game-container"):
            yield from self.tui.compose_view()
            
        yield Label("", id="log-line")
        yield Footer()

    def on_mount(self) -> None:
        # Redirect loguru
        logger.remove()
        logger.add(self.sink_log, format="{message}")
        
        # Check device
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        self.device_text = device
        self.query_one("#device-label", Label).update(f"Device: {device}")
        
        self.run_worker(self.simulation_loop, exclusive=True, thread=True)

    def sink_log(self, message):
        self.call_from_thread(self.update_log, message)

    def update_log(self, message):
        self.last_log = message.strip()
        self.query_one("#log-line", Label).update(self.last_log)

    def watch_status_text(self, value: str) -> None:
        try:
            self.query_one("#status-label", Label).update(value)
        except Exception:
            pass

    def simulation_loop(self):
        worker = get_current_worker()
        
        try:
            env = self.task.env # Use the task's persistent env
            config = self.task.config
            
            agent = BaseDQNAgent(self.task.state_size, self.task.action_size, config, model_factory=self.task.create_model)
            
            use_random_policy = False
            if self.weight_path:
                logger.info(f"Loading weights from {self.weight_path}")
                agent.load(self.weight_path)
                agent.epsilon = 0.0
            else:
                logger.warning("No weights provided. Using random policy for visualization.")
                agent.epsilon = 1.0 # Full random to show movement
                use_random_policy = True

            logger.info(f"Started {self.task_name}")

            episode = 0
            while not worker.is_cancelled:
                episode += 1
                state, info = env.reset()
                raw_state = state
                state = self.task.preprocess_state(state)
                done = False
                total_reward = 0
                step = 0
                
                while not done and not worker.is_cancelled:
                    step += 1
                    
                    # Update UI
                    self.call_from_thread(self.tui.update_state, raw_state, info)
                    self.call_from_thread(self.tui.update_stats, episode, step, total_reward)
                    
                    # Also update App status for debug
                    status = f"Running {self.task_name}..."
                    self.call_from_thread(lambda: setattr(self, "status_text", status))
                    
                    action = agent.act(state, training=use_random_policy)
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    raw_state = next_state
                    state = self.task.preprocess_state(next_state)
                    total_reward += reward
                    
                    time.sleep(0.05) 
                
                logger.info(f"Episode {episode} finished. Total Reward: {total_reward}")
                time.sleep(0.5)
            
            # Note: We don't close env here as it is managed by the Task
            
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(5)
