import numpy as np
import os
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
    
    # Dynamic output path handling
    if output_path:
        config.model_path = output_path
        if config.model_path.endswith(".pth"):
             config.plot_path = config.model_path.replace(".pth", ".png")
        else:
             config.plot_path = config.model_path + ".png"
    else:
        # Default based on task name
        # Ensure outputs dir exists
        os.makedirs("outputs", exist_ok=True)
        config.model_path = f"outputs/{task_name}.pth"
        config.plot_path = f"outputs/{task_name}.png"
    
    task = get_task(task_name)
    env = task.make_env()
    
    agent = DQNAgent(task.state_size, task.action_size, config)
    
    logger.info(f"Starting training on {task.name} for {config.episodes} episodes.")
    logger.info(f"Output: {config.model_path}")
    logger.info(f"Device: {agent.device} | Batch: {config.batch_size} | LR: {config.learning_rate}")
    
    rewards_history = []
    moving_avg_history = []
    best_reward = -float('inf')

    for e in range(config.episodes):
        state, info = env.reset()
        state = task.preprocess_state(state)
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            steps += 1
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Enforce max steps
            if steps >= config.max_steps:
                truncated = True
            
            next_state = task.preprocess_state(next_state)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay()
        
        # Update target network
        if e % config.target_update_freq == 0:
            agent.update_target_model()

        # Decay epsilon per episode
        if agent.epsilon > config.epsilon_min:
            agent.epsilon *= config.epsilon_decay

        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:])
        moving_avg_history.append(avg_reward)
        
        # Log logic:
        # - First 20 episodes: Log every episode
        # - Thereafter: Log every 10 episodes
        should_log = (e < 20) or ((e+1) % 10 == 0)
        
        if should_log:
            logger.info(f"Episode: {e+1:03d}/{config.episodes} | Steps: {steps:03d} | Score: {total_reward: .2f} | Avg: {avg_reward: .2f} | Epsilon: {agent.epsilon:.3f}")

        if avg_reward > best_reward:
            logger.success(f"New Best Avg Reward: {avg_reward:.2f} (was {best_reward:.2f}). Saving model...")
            best_reward = avg_reward
            agent.save(config.model_path)
            
    env.close()
    
    plot_rewards(rewards_history, moving_avg_history, config.plot_path)
    logger.success(f"Training completed. Best Avg Reward: {best_reward:.2f}. Model saved to {config.model_path}")