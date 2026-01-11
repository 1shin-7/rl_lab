import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from loguru import logger
from .config import Config

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, config: Config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Device selection: generic torch device selection.
        # This will pick up cuda if available (native Linux/WSL2) or cpu.
        # If user installs torch-directml or others, specific handling might be needed but standard torch is fine.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.memory = deque(maxlen=config.memory_size)
        self.epsilon = config.epsilon_start
        
        # Main model (policy network)
        self.model = DQN(state_size, action_size).to(self.device)
        # Target model (for stable Q-targets)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        """Transfer weights from model to target_model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Select action using epsilon-greedy policy if training, else greedy."""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensor on device
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0) # (1, state_size)
            
        state_t = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            act_values = self.model(state_t)
            
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.config.train_start_size:
            return 0.0

        minibatch = random.sample(self.memory, self.config.batch_size)
        
        # Pre-process batch
        # Convert list of tuples to separate numpy arrays first for speed
        states = np.array([i[0] for i in minibatch], dtype=np.float32)
        actions = np.array([i[1] for i in minibatch], dtype=np.int64) # LongTensor for indexing
        rewards = np.array([i[2] for i in minibatch], dtype=np.float32)
        next_states = np.array([i[3] for i in minibatch], dtype=np.float32)
        dones = np.array([i[4] for i in minibatch], dtype=np.float32)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device) # (batch, 1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) # (batch, 1)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device) # (batch, 1)

        # 1. Get predicted Q values for current states and actions
        # model(states) -> (batch, action_size)
        # gather(1, actions) -> selects the Q-value for the action taken
        current_q_values = self.model(states_t).gather(1, actions_t)

        # 2. Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states_t).max(1)[0].unsqueeze(1)
            target_q_values = rewards_t + (self.config.gamma * next_q_values * (1.0 - dones_t))

        # 3. Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # 4. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay
            
        return loss.item()

    def load(self, name):
        logger.info(f"Loading model from {name}")
        # Map location ensures we can load GPU models on CPU if needed
        self.model.load_state_dict(torch.load(name, map_location=self.device))
        self.update_target_model()

    def save(self, name):
        logger.info(f"Saving model to {name}")
        torch.save(self.model.state_dict(), name)