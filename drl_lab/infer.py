import time
from loguru import logger
from .utils import Config
from .agent import BaseDQNAgent
from .tasks import get_task

def infer(task_name: str, weight_path: str, episodes: int = 5, render_mode: str = None):
    config = Config()
    if weight_path:
        config.model_path = weight_path
    
    task = get_task(task_name)
    env = task.make_env(render_mode=render_mode)
    
    agent = BaseDQNAgent(task.state_size, task.action_size, config, model_factory=task.create_model)
    
    try:
        agent.load(config.model_path)
    except Exception as e:
        logger.error(f"Could not load model from {config.model_path}. Error: {e}")
        return

    logger.info(f"Starting inference on {task.name} for {episodes} episodes...")
    
    # Disable exploration
    agent.epsilon = 0.0

    for e in range(episodes):
        state, info = env.reset()
        state = task.preprocess_state(state)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = task.preprocess_state(next_state)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            
            if render_mode == "human":
                time.sleep(0.01) # Slow down slightly for visualization
        
        logger.info(f"Episode: {e+1}/{episodes} | Score: {total_reward}")

    env.close()
    logger.success("Inference completed.")
