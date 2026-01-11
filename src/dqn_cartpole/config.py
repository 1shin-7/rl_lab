from dataclasses import dataclass

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
    model_path: str = "outputs/dqn_cartpole_model.pth"
    plot_path: str = "outputs/training_plot.png"
    log_file: str = "outputs/training.log"
    render_mode: str = None # Set to 'human' for visualization during inference
