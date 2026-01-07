"""
Neural network architectures for SPGG policy and value estimation.

Implements the actor-critic networks used in PPO training for learning
optimal generation parameters in the sequential public goods game.
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Policy network for SPGG parameter generation.
    
    Maps encoded state representations to normalized generation parameters
    using a multi-layer perceptron with dropout regularization.
    
    Architecture:
        input_dim -> hidden_dim -> 128 -> output_dim (sigmoid activation)
    
    Args:
        input_dim: Dimension of the input state vector. Default: 896.
        hidden_dim: Dimension of the first hidden layer. Default: 256.
        output_dim: Dimension of the output (number of generation params). Default: 6.
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
        """
        Forward pass through the policy network.
        
        Args:
            x: Input state tensor of shape (batch_size, input_dim).
            
        Returns:
            Normalized parameter tensor of shape (batch_size, output_dim),
            with values in [0, 1] that are later scaled to actual parameter ranges.
        """
        return self.network(x)


class ValueNetwork(nn.Module):
    """
    Value network for SPGG state value estimation.
    
    Estimates the expected return from a given state for the PPO baseline.
    Uses the same architecture as the policy network but outputs a scalar value.
    
    Architecture:
        input_dim -> hidden_dim -> 128 -> 1
    
    Args:
        input_dim: Dimension of the input state vector. Default: 896.
        hidden_dim: Dimension of the first hidden layer. Default: 256.
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
        """
        Forward pass through the value network.
        
        Args:
            x: Input state tensor of shape (batch_size, input_dim).
            
        Returns:
            Value estimate tensor of shape (batch_size, 1).
        """
        return self.network(x)
