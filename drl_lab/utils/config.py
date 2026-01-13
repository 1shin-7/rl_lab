from dataclasses import dataclass
from . import paths

@dataclass
class Config:
    env_name: str = "CartPole-v1"
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    learning_rate: float = 0.001
    batch_size: int = 64
    memory_size: int = 2000
    train_start_size: int = 1000
    target_update_freq: int = 10
    episodes: int = 500  # CartPole-v1 is solved at 475 avg reward
    max_steps: int = 200 # Force end episode if taking too long
    # Default paths using centralized utils
    model_path: str = str(paths.get_model_path("dqn_cartpole_model"))
    plot_path: str = str(paths.get_plot_path("training_plot"))
    log_file: str = str(paths.OUTPUTS_DIR / "training.log")
    render_mode: str = None # Set to 'human' for visualization during inference