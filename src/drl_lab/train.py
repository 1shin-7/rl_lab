import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from .config import Config
from .agent import DQNAgent
from .tasks import get_task

def plot_rewards(rewards, moving_avg, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward', alpha=0.5)
    plt.plot(moving_avg, label='Moving Average (100 eps)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training plot saved to {save_path}")

def train(task_name: str, output_path: str, episodes: int):
    # Load default config and override
    config = Config()
    config.episodes = episodes
    if output_path:
        config.model_path = output_path
        # Derive plot path from model path
        config.plot_path = output_path.replace(".pth", ".png")
    
    task = get_task(task_name)
    env = task.make_env()
    
    agent = DQNAgent(task.state_size, task.action_size, config)
    
    logger.info(f"Starting training on {task.name} for {config.episodes} episodes.")
    
    rewards_history = []
    moving_avg_history = []
    best_reward = -float('inf')

    for e in range(config.episodes):
        state, info = env.reset()
        state = task.preprocess_state(state)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = task.preprocess_state(next_state)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay()
        
        # Update target network
        if e % config.target_update_freq == 0:
            agent.update_target_model()

        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:])
        moving_avg_history.append(avg_reward)
        
        if (e+1) % 10 == 0:
            logger.info(f"Episode: {e+1}/{config.episodes} | Score: {total_reward} | Avg: {avg_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(config.model_path)
            
    env.close()
    
    plot_rewards(rewards_history, moving_avg_history, config.plot_path)
    logger.success(f"Training completed. Model saved to {config.model_path}")