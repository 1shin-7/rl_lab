import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingMLP(nn.Module):
    """
    Dueling Network Architecture (Wang et al., 2015).
    Splits Q-value estimation into State Value (V) and Advantage (A).
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        
        # Feature extraction layer
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        features = self.feature(x)
        
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        
        # Combine V and A
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q