"""
Core modules for GSM8K evaluation with SPGG framework.

This package provides the neural network architectures, state encoders,
and utility functions for the Sequential Public Goods Game evaluation
on mathematical reasoning tasks.
"""

from .networks import PolicyNetwork, ValueNetwork
from .encoders import MathStateEncoder
from .config import SPGGConfig
from .utils import (
    parse_generation_params,
    load_checkpoint_params,
    load_gsm8k_samples,
    safe_for_json,
    save_results
)
from .agents import APIModelManager, MathProblemPool, MathAgent

__all__ = [
    'PolicyNetwork',
    'ValueNetwork',
    'MathStateEncoder',
    'SPGGConfig',
    'parse_generation_params',
    'load_checkpoint_params',
    'load_gsm8k_samples',
    'safe_for_json',
    'save_results',
    'APIModelManager',
    'MathProblemPool',
    'MathAgent',
]
