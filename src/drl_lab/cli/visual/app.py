from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Static, Label
from textual.reactive import reactive
from textual.worker import get_current_worker
from loguru import logger
import time

from ...tasks import get_task
from ...config import Config
from ...agent import DQNAgent

class VisualApp(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr auto;
    }
    
    #game-container {
        width: 100%;
        height: 100%;
        align: center middle;
    }
    
    #status-bar {
        dock: top;
        height: 3;
        background: $primary-background;
        color: $text;
        border-bottom: solid $accent;
        content-align: center middle;
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
    last_log = reactive("")

    def __init__(self, task_name: str, weight_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.task_name = task_name
        self.weight_path = weight_path

    def compose(self) -> ComposeResult:
        yield Label(self.status_text, id="status-bar")
        
        with Container(id="game-container"):
            yield Static(id="game-view")
            
        yield Label("", id="log-line")
        yield Footer()

    def on_mount(self) -> None:
        # Redirect loguru
        logger.remove()
        logger.add(self.sink_log, format="{message}")
        
        self.run_worker(self.simulation_loop, exclusive=True, thread=True)

    def sink_log(self, message):
        self.call_from_thread(self.update_log, message)

    def update_log(self, message):
        self.last_log = message.strip()
        self.query_one("#log-line", Label).update(self.last_log)

    def watch_status_text(self, value: str) -> None:
        try:
            self.query_one("#status-bar", Label).update(value)
        except:
            pass

    def simulation_loop(self):
        worker = get_current_worker()
        task_name = self.task_name
        
        try:
            task = get_task(task_name)
            env = task.make_env()
            config = Config()
            
            agent = DQNAgent(task.state_size, task.action_size, config)
            
            # Load weights if provided
            if self.weight_path:
                logger.info(f"Loading weights from {self.weight_path}")
                agent.load(self.weight_path)
            else:
                logger.warning("No weight path provided, running with random agent.")
                
            # Disable exploration for inference
            agent.epsilon = 0.0

            logger.info(f"Started visual inference on {task_name}")

            episode = 0
            while not worker.is_cancelled:
                episode += 1
                state, info = env.reset()
                raw_state = state
                state = task.preprocess_state(state)
                done = False
                total_reward = 0
                step = 0
                
                while not done and not worker.is_cancelled:
                    step += 1
                    # Update UI
                    tui_view = task.render_tui(raw_state, info)
                    self.call_from_thread(self.query_one("#game-view", Static).update, tui_view)
                    
                    status = f"Episode: {episode} | Step: {step} | Reward: {total_reward:.2f}"
                    self.call_from_thread(lambda: setattr(self, "status_text", status))
                    
                    # Act greedily (epsilon=0.0)
                    action = agent.act(state, training=False)
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    raw_state = next_state
                    state = task.preprocess_state(next_state)
                    total_reward += reward
                    
                    time.sleep(0.1) # Control speed
                
                logger.info(f"Episode {episode} finished. Total Reward: {total_reward}")
                time.sleep(1) # Pause between episodes
                
            env.close()
            
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(5)

