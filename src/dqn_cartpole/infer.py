import gymnasium as gym
import numpy as np
import time
from loguru import logger
from .config import Config
from .agent import DQNAgent

def infer(config: Config, episodes: int = 5, render: bool = False):
    render_mode = "human" if render else None
    env = gym.make(config.env_name, render_mode=render_mode)
    
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    
    agent = DQNAgent(state_size, action_size, config)
    
    try:
        agent.load(config.model_path)
    except Exception as e:
        logger.error(f"Could not load model from {config.model_path}. Run 'train' first. Error: {e}")
        return

    logger.info(f"Starting inference for {episodes} episodes...")
    
    # Disable exploration
    agent.epsilon = 0.0

    for e in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            
            if render:
                time.sleep(0.01) # Slow down slightly for visualization
        
        logger.info(f"Episode: {e+1}/{episodes} | Score: {total_reward}")

    env.close()
    logger.success("Inference completed.")
