import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from .config import Config
from .agent import DQNAgent

def plot_rewards(rewards, moving_avg, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward', alpha=0.5)
    plt.plot(moving_avg, label='Moving Average (100 eps)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training: CartPole-v0')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training plot saved to {save_path}")

def train(config: Config):
    env = gym.make(config.env_name)
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    
    agent = DQNAgent(state_size, action_size, config)
    
    logger.info(f"Starting training on {config.env_name} for {config.episodes} episodes.")
    logger.info(f"Hyperparameters: Gamma={config.gamma}, LR={config.learning_rate}, Batch={config.batch_size}")

    rewards_history = []
    moving_avg_history = []
    best_reward = 0

    for e in range(config.episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Custom reward shaping can go here if needed, but CartPole default is fine for simple pass.
            # Usually reward is +1 per step.
            
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
        
        logger.info(f"Episode: {e+1}/{config.episodes} | Score: {total_reward} | Avg: {avg_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(config.model_path)
            
    env.close()
    
    # Save final plot
    plot_rewards(rewards_history, moving_avg_history, config.plot_path)
    logger.success("Training completed.")
