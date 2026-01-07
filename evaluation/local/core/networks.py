"""
Neural network architectures for SPGG policy and value estimation.
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Policy network for generating LLM parameters.
    
    Takes encoded state as input and outputs generation parameters
    (temperature, top_p, top_k, max_tokens, repetition_penalty, presence_penalty).
    
    Args:
        input_dim: Dimension of input state vector.
        hidden_dim: Dimension of hidden layers.
        output_dim: Dimension of output (number of generation parameters).
    """
    
    def __init__(self, input_dim: int = 896, hidden_dim: int = 256, output_dim: int = 6):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network."""
        return self.network(x)


class ValueNetwork(nn.Module):
    """
    Value network for estimating state values.
    
    Takes encoded state as input and outputs a scalar value estimate.
    
    Args:
        input_dim: Dimension of input state vector.
        hidden_dim: Dimension of hidden layers.
    """
    
    def __init__(self, input_dim: int = 896, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value network."""
        return self.network(x)
