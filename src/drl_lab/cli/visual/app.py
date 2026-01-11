from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Footer, Static, Label
from textual.reactive import reactive
from textual.worker import get_current_worker
from loguru import logger
import torch
import time

from ...tasks import get_task
from ...utils import Config
from ...agent import BaseDQNAgent

class VisualApp(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: auto 1fr auto;
    }
    
    #header {
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
        # background: $surface; 
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
        self.game_widget = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="header"):
            yield Label(self.status_text, id="status-label")
            yield Label(f"Device: {self.device_text}", id="device-label")
        
        with Container(id="game-container"):
            # Placeholder, will be replaced or content set
            yield Static(id="game-placeholder")
            
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
        except:
            pass

    def simulation_loop(self):
        worker = get_current_worker()
        task_name = self.task_name
        
        try:
            task = get_task(task_name)
            
            # Setup specific widget if task supports it
            custom_widget = task.create_visual_widget()
            if custom_widget:
                self.call_from_thread(self.mount_custom_widget, custom_widget)
                self.game_widget = custom_widget
            else:
                # Fallback to Static
                self.call_from_thread(self.mount_static_widget)
                
            env = task.make_env()
            config = Config()
            
            agent = BaseDQNAgent(task.state_size, task.action_size, config, model_factory=task.create_model)
            
            use_random_policy = False
            if self.weight_path:
                logger.info(f"Loading weights from {self.weight_path}")
                agent.load(self.weight_path)
                agent.epsilon = 0.0
            else:
                logger.warning("No weights provided. Using random policy for visualization.")
                agent.epsilon = 1.0 # Full random to show movement
                use_random_policy = True

            logger.info(f"Started {task_name}")

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
                    if self.game_widget and hasattr(self.game_widget, 'update_state'):
                         # Use custom widget update
                         self.call_from_thread(self.game_widget.update_state, raw_state, info)
                    else:
                         # Fallback to render_tui -> Static.update
                         tui_view = task.render_tui(raw_state, info)
                         self.call_from_thread(self.query_one("#game-view", Static).update, tui_view)
                    
                    status = f"Episode: {episode} | Step: {step} | Reward: {total_reward:.2f}"
                    self.call_from_thread(lambda: setattr(self, "status_text", status))
                    
                    # If we want random behavior (no weights), we must set training=True to use epsilon
                    # If we have weights, training=False uses greedy
                    action = agent.act(state, training=use_random_policy)
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    raw_state = next_state
                    state = task.preprocess_state(next_state)
                    total_reward += reward
                    
                    time.sleep(0.05) 
                
                logger.info(f"Episode {episode} finished. Total Reward: {total_reward}")
                time.sleep(0.5)
                
            env.close()
            
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(5)

    def mount_custom_widget(self, widget):
        container = self.query_one("#game-container", Container)
        container.remove_children()
        container.mount(widget)
        
    def mount_static_widget(self):
        container = self.query_one("#game-container", Container)
        if not container.query("#game-view"):
            container.remove_children()
            container.mount(Static(id="game-view"))