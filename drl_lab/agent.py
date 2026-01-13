import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from loguru import logger
from typing import Callable, Union
from pathlib import Path
from .utils import Config

class BaseDQNAgent:
    """
    Base Deep Q-Network Agent.
    
    Attributes:
        state_size (int): Dimension of the state space.
        action_size (int): Dimension of the action space.
        config (Config): Configuration object containing hyperparameters.
        device (torch.device): Computation device (CPU or CUDA).
        memory (deque): Replay memory buffer.
        epsilon (float): Current exploration rate.
        model (nn.Module): The policy network.
        target_model (nn.Module): The target network for stable Q-learning.
        optimizer (optim.Optimizer): The optimizer for training the model.
        loss_fn (nn.Module): The loss function (MSE).
    """

    def __init__(
        self, 
        state_size: int, 
        action_size: int, 
        config: Config, 
        model_factory: Callable[[], nn.Module]
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Logger is handled by the caller/trainer mostly, but we log device here
        logger.debug(f"Agent initialized on device: {self.device}")

        self.memory: deque = deque(maxlen=config.memory_size)
        self.epsilon: float = config.epsilon_start
        
        # Initialize networks
        self.model = model_factory().to(self.device)
        self.target_model = model_factory().to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self) -> None:
        """Transfer weights from the policy model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(
        self, 
        state: Union[np.ndarray, list], 
        action: int, 
        reward: float, 
        next_state: Union[np.ndarray, list], 
        done: bool
    ) -> None:
        """Store a transition tuple in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: Union[np.ndarray, list], training: bool = True) -> int:
        """
        Select an action using an epsilon-greedy policy if training, otherwise greedy.
        
        Args:
            state: The current state.
            training: Whether the agent is in training mode (enables epsilon-greedy).
        
        Returns:
            int: The selected action index.
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Prepare state tensor
        if isinstance(state, list):
            state = np.array(state)
        
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)  # Shape: (1, state_size)
            
        state_t = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            act_values = self.model(state_t)
            
        return int(np.argmax(act_values.cpu().data.numpy()))

    def replay(self) -> float:
        """
        Sample a batch from memory and train the network.
        
        Returns:
            float: The loss value of the training step, or 0.0 if not enough memory.
        """
        if len(self.memory) < self.config.train_start_size:
            return 0.0

        minibatch = random.sample(self.memory, self.config.batch_size)
        
        # Vectorized batch processing
        states = np.array([i[0] for i in minibatch], dtype=np.float32)
        actions = np.array([i[1] for i in minibatch], dtype=np.int64) 
        rewards = np.array([i[2] for i in minibatch], dtype=np.float32)
        next_states = np.array([i[3] for i in minibatch], dtype=np.float32)
        dones = np.array([i[4] for i in minibatch], dtype=np.float32)

        # To device
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device) 
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) 
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device) 

        # 1. Predicted Q values
        current_q_values = self.model(states_t).gather(1, actions_t)

        # 2. Target Q values
        with torch.no_grad():
            # Double DQN could be implemented here (using model for selection, target for evaluation)
            # For now, standard DQN:
            next_q_values = self.target_model(next_states_t).max(1)[0].unsqueeze(1)
            target_q_values = rewards_t + (self.config.gamma * next_q_values * (1.0 - dones_t))

        # 3. Loss & Optimization
        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        return loss.item()

    def load(self, path: Union[str, Path]) -> None:
        """Load model weights from a file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Model file not found: {path}")
            return
            
        logger.info(f"Loading model from {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_model()

    def save(self, path: Union[str, Path]) -> None:
        """Save model weights to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {path}")
        torch.save(self.model.state_dict(), path)