from pathlib import Path
from typing import Optional, Union, Any, Dict, Protocol, Callable
from loguru import logger

from .utils import PlotRenderer, paths
from .agent import BaseDQNAgent
from .tasks import get_task, BaseTask

class TrainingCallbacks(Protocol):
    def on_step(self, step: int, state: Any, reward: float, info: Dict[str, Any]) -> None: ...
    def on_episode_end(self, episode: int, steps: int, reward: float) -> None: ...

class Trainer:
    """
    Manages the training lifecycle for a Reinforcement Learning agent.
    """

    def __init__(
        self, 
        task_name: str, 
        output_path: Optional[Union[str, Path]] = None, 
        episodes: Optional[int] = None,
        callbacks: Optional[TrainingCallbacks] = None,
        should_stop: Optional[Callable[[], bool]] = None
    ):
        self.task_name = task_name
        self.output_path = Path(output_path) if output_path else None
        self.episodes_override = episodes
        self.callbacks = callbacks
        self.should_stop = should_stop or (lambda: False)
        
        self.task: BaseTask = get_task(task_name)
        self._setup_config()
        self._setup_paths()
        
        # Lazy initialization
        self.agent: Optional[BaseDQNAgent] = None
        self.plotter: Optional[PlotRenderer] = None
        self.best_reward = -float('inf')

    def _setup_config(self) -> None:
        """Applies configuration overrides."""
        if self.episodes_override:
            self.task.config.episodes = self.episodes_override
        self.config = self.task.config

    def _setup_paths(self) -> None:
        """Configures model and plot paths using centralized path utilities."""
        model_path, plot_path = paths.resolve_task_paths(self.task_name, self.output_path)
        self.config.model_path = str(model_path)
        self.config.plot_path = str(plot_path)

    def _initialize(self) -> None:
        """Initializes the agent, environment, and resources."""
        # Ensure environment is ready (handled by Task)
        _ = self.task.env 
        
        self.agent = BaseDQNAgent(
            state_size=self.task.state_size, 
            action_size=self.task.action_size, 
            config=self.config, 
            model_factory=self.task.create_model
        )
        self.plotter = PlotRenderer(self.task.name, Path(self.config.plot_path))
        
        logger.info(f"Initialized training for task: {self.task.name}")
        logger.info(f"   Episodes: {self.config.episodes}")
        logger.info(f"   Device: {self.agent.device}")
        logger.info(f"   Batch Size: {self.config.batch_size} | LR: {self.config.learning_rate}")
        logger.info(f"   Output: {self.config.model_path}")

    def _run_episode(self, episode_idx: int) -> float:
        """
        Runs a single episode.
        
        Returns:
            float: The total reward obtained in this episode.
        """
        self.task.pre_episode(episode_idx)
        
        state, info = self.task.env.reset()
        raw_state = state
        state = self.task.preprocess_state(state)
        
        total_reward = 0.0
        steps = 0
        done = False
        
        # Initial callback
        if self.callbacks:
            self.callbacks.on_step(steps, raw_state, total_reward, info)
        
        while not done and not self.should_stop():
            steps += 1
            action = self.agent.act(state, training=True)
            
            next_state, reward, terminated, truncated, info = self.task.env.step(action)
            total_reward += reward
            
            # Enforce max steps
            if steps >= self.config.max_steps:
                truncated = True
            
            # Callback update with real-time reward
            if self.callbacks:
                self.callbacks.on_step(steps, next_state, total_reward, info)

            next_state_pre = self.task.preprocess_state(next_state)
            done = terminated or truncated
            
            self.agent.remember(state, action, reward, next_state_pre, done)
            state = next_state_pre
            
            self.agent.replay()
        
        self.task.post_episode(episode_idx, total_reward)
        return total_reward, steps

    def _update_agent_state(self, episode_idx: int) -> None:
        """Updates agent internal state (target model, epsilon)."""
        # Update target network
        if episode_idx % self.config.target_update_freq == 0:
            self.agent.update_target_model()

        # Decay epsilon
        if self.agent.epsilon > self.config.epsilon_min:
            self.agent.epsilon *= self.config.epsilon_decay

    def _log_and_save(self, episode_idx: int, steps: int, reward: float) -> None:
        """Handles logging and model saving based on performance."""
        self.plotter.update(reward)
        avg_reward = self.plotter.moving_avgs[-1] if self.plotter.moving_avgs else reward
        
        if self.callbacks:
            self.callbacks.on_episode_end(episode_idx, steps, reward)

        # Log logic: First 20 episodes verbose, then every 10
        should_log = (episode_idx < 20) or ((episode_idx + 1) % 10 == 0)
        
        if should_log:
            logger.info(
                f"Ep {episode_idx + 1:03d}/{self.config.episodes} | "
                f"Steps: {steps:03d} | "
                f"Reward: {reward: >6.2f} | "
                f"Avg: {avg_reward: >6.2f} | "
                f"Eps: {self.agent.epsilon:.3f}"
            )

        if avg_reward > self.best_reward:
            logger.success(f"New Best Avg Reward: {avg_reward:.2f} (prev: {self.best_reward:.2f}). Saving...")
            self.best_reward = avg_reward
            self.agent.save(self.config.model_path)

    def run(self) -> None:
        """Executes the full training loop."""
        self._initialize()
        
        # Hook: Pre-training
        try:
            self.task.pre_training()
        except Exception as e:
            logger.error(f"Error in pre_training hook: {e}")
            return

        try:
            for e in range(self.config.episodes):
                if self.should_stop():
                    logger.warning("Training stop signal received.")
                    break
                    
                reward, steps = self._run_episode(e)
                
                # If stopped mid-episode, don't update state or log
                if self.should_stop():
                    break
                    
                self._update_agent_state(e)
                self._log_and_save(e, steps, reward)
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        except Exception as e:
            logger.exception(f"Unexpected error during training: {e}")
        finally:
            # Hook: Post-training
            try:
                self.task.post_training()
            except Exception as e:
                logger.error(f"Error in post_training hook: {e}")
            
            if self.plotter:
                self.plotter.render()
                
            logger.success(f"Training session ended. Best Avg Reward: {self.best_reward:.2f}")

def train(task_name: str, output_path: str, episodes: int) -> None:
    """Legacy wrapper for backward compatibility."""
    trainer = Trainer(task_name, output_path, episodes)
    trainer.run()
