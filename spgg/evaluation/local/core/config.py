"""
Configuration for SPGG local GSM8K evaluation.
"""

from dataclasses import dataclass


@dataclass
class SPGGConfig:
    """
    Configuration parameters for SPGG generation.
    
    Defines the ranges for dynamic parameter generation by policy networks.
    """
    temp_min: float = 0.3
    temp_max: float = 1.2
    top_p_min: float = 0.6
    top_p_max: float = 0.95
    top_k_min: int = 20
    top_k_max: int = 80
    max_tokens_min: int = 700
    max_tokens_max: int = 2048
    repetition_penalty_min: float = 1.0
    repetition_penalty_max: float = 1.3
    presence_penalty_min: float = 0.0
    presence_penalty_max: float = 0.6
