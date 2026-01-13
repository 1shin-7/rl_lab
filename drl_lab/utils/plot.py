import matplotlib
# Force non-interactive backend 'Agg' before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from loguru import logger
import numpy as np
from . import paths

class PlotRenderer:
    def __init__(self, task_name: str, filepath: Path):
        self.task_name = task_name
        self.filepath = Path(filepath)
        self.rewards = []
        self.moving_avgs = []
        self.window_size = 100

    def update(self, reward: float):
        """Updates the internal data with a new episode reward."""
        self.rewards.append(reward)
        avg = np.mean(self.rewards[-self.window_size:])
        self.moving_avgs.append(avg)

    def render(self):
        """Renders and saves the plot to the configured filepath."""
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.rewards, label='Episode Reward', alpha=0.5)
            plt.plot(self.moving_avgs, label=f'Moving Average ({self.window_size} eps)', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'DQN Training: {self.task_name}')
            plt.legend()
            plt.grid(True)
            
            # Ensure parent directory exists using centralized utils
            paths.ensure_dir(self.filepath)
            
            plt.savefig(self.filepath)
            plt.close()
            logger.info(f"Training plot saved to {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to save plot to {self.filepath}: {e}")
