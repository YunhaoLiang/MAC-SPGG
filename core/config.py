"""
Configuration classes for SPGG framework.

Defines the hyperparameter ranges for dynamic generation parameter optimization.
"""

from dataclasses import dataclass


@dataclass
class SPGGConfig:
    """
    Configuration for SPGG dynamic parameter generation.
    
    Defines the valid ranges for each generation parameter that the policy
    network learns to optimize during the public goods game.
    
    Attributes:
        temp_min: Minimum temperature for sampling.
        temp_max: Maximum temperature for sampling.
        top_p_min: Minimum nucleus sampling probability.
        top_p_max: Maximum nucleus sampling probability.
        top_k_min: Minimum top-k sampling parameter.
        top_k_max: Maximum top-k sampling parameter.
        max_tokens_min: Minimum output token limit.
        max_tokens_max: Maximum output token limit.
        repetition_penalty_min: Minimum repetition penalty.
        repetition_penalty_max: Maximum repetition penalty.
        presence_penalty_min: Minimum presence penalty.
        presence_penalty_max: Maximum presence penalty.
    """
    temp_min: float = 0.3
    temp_max: float = 1.2
    top_p_min: float = 0.6
    top_p_max: float = 0.95
    top_k_min: int = 20
    top_k_max: int = 80
    max_tokens_min: int = 100
    max_tokens_max: int = 1012
    repetition_penalty_min: float = 1.0
    repetition_penalty_max: float = 1.3
    presence_penalty_min: float = 0.0
    presence_penalty_max: float = 0.6
