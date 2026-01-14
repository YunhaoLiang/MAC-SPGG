"""
SPGG: Sequential Public Goods Game framework.

Main package for multi-agent collaborative reasoning.
"""

# Import from API core (for backward compatibility)
from .api.core import (
    PolicyNetwork,
    ValueNetwork,
    MathStateEncoder,
    SPGGConfig,
    parse_generation_params,
    load_checkpoint_params,
    load_gsm8k_samples,
    safe_for_json,
    save_results,
    APIModelManager,
    MathProblemPool,
    MathAgent,
)

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
