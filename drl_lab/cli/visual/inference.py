from textual.app import App, ComposeResult
from textual.widgets import Footer, Label
from textual.worker import get_current_worker
from loguru import logger
import torch
import time

from ...tasks import get_task
from ...agent import BaseDQNAgent

class VisualInferenceApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #task-content {
        height: 1fr;
        width: 100%;
        align: center middle;
    }
    
    #log-line {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit")
    ]

    def __init__(self, task_name: str, weight_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.task_name = task_name
        self.weight_path = weight_path
        
        self.rl_task = get_task(task_name)
        self.tui = self.rl_task.render()
        self._worker = None

    def compose(self) -> ComposeResult:
        yield from self.tui.compose_view()
        yield Label("", id="log-line")
        yield Footer()

    def on_mount(self) -> None:
        logger.remove()
        logger.add(self.sink_log, format="{message}")
        
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        logger.info(f"Inference Device: {device}")
        
        self._worker = self.run_worker(self.simulation_loop, exclusive=True, thread=True)

    def action_quit(self) -> None:
        if self._worker:
            self._worker.cancel()
        self.exit()

    def sink_log(self, message):
        try:
            if self.is_running:
                self.call_from_thread(self.update_log, message)
        except RuntimeError:
            pass

    def update_log(self, message):
        try:
            self.query_one("#log-line", Label).update(message.strip())
        except Exception:
            pass

    def simulation_loop(self):
        worker = get_current_worker()
        
        try:
            env = self.rl_task.env 
            config = self.rl_task.config
            
            agent = BaseDQNAgent(self.rl_task.state_size, self.rl_task.action_size, config, model_factory=self.rl_task.create_model)
            
            use_random_policy = False
            if self.weight_path:
                logger.info(f"Loading weights from {self.weight_path}")
                agent.load(self.weight_path)
                agent.epsilon = 0.0
            else:
                logger.warning("No weights provided. Using random policy.")
                agent.epsilon = 1.0 
                use_random_policy = True

            logger.info(f"Started {self.task_name}")

            episode = 0
            while not worker.is_cancelled:
                episode += 1
                state, info = env.reset()
                raw_state = state
                state = self.rl_task.preprocess_state(state)
                done = False
                total_reward = 0
                step = 0
                
                while not done and not worker.is_cancelled:
                    step += 1
                    
                    try:
                        self.call_from_thread(self.tui.update_state, raw_state, info)
                        self.call_from_thread(self.tui.update_stats, episode, step, total_reward)
                    except RuntimeError:
                        pass # App closing
                    
                    action = agent.act(state, training=use_random_policy)
                    
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    raw_state = next_state
                    state = self.rl_task.preprocess_state(next_state)
                    total_reward += reward
                    
                    time.sleep(0.05) 
                
                if worker.is_cancelled:
                    break
                    
                logger.info(f"Episode {episode} finished. Total Reward: {total_reward}")
                time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(5)
        
        if not worker.is_cancelled:
            self.call_from_thread(self.exit)
