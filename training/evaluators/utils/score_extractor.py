"""
Score extraction utilities for evaluator model outputs.

Extracts four-dimensional reward vectors from model responses:
r = (r_relevance, r_coherence, r_consistency, r_fluency)
"""

import re
from typing import Dict, Optional

DIMENSIONS = ['relevance', 'coherence', 'consistency', 'fluency']
SCORE_MIN = 0.0
SCORE_MAX = 5.0


def extract_scores_from_response(response: str) -> Dict[str, float]:
    """
    Extract four-dimensional scores from model response.
    
    Supports multiple output formats:
    1. Standard format: "Dimension: X.XX"
    2. Sequential numbers format
    3. Mixed format with partial labels
    
    Args:
        response: Raw model output string.
    
    Returns:
        Dictionary mapping dimension names to scores.
        Returns partial results if not all scores can be extracted.
    """
    scores = {}
    response = response.strip()
    
    # Method 1: Pattern matching for labeled scores
    for dim in DIMENSIONS:
        pattern = f"{dim}:\\s*(\\d+\\.?\\d*)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            scores[dim] = _clamp_score(score)
    
    # Method 2: Extract all valid numbers in sequence
    if len(scores) < 4:
        all_numbers = re.findall(r'(\d+\.?\d*)', response)
        valid_numbers = [float(n) for n in all_numbers if _is_valid_score(float(n))]
        
        if len(valid_numbers) >= 4:
            for i, dim in enumerate(DIMENSIONS):
                if dim not in scores:
                    scores[dim] = valid_numbers[i]
    
    # Method 3: Sequential extraction for partial results
    if len(scores) < 4 and len(scores) > 0:
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        for i, dim in enumerate(DIMENSIONS):
            if dim not in scores and i < len(lines):
                number_match = re.search(r'(\d+\.?\d*)', lines[i])
                if number_match:
                    score = float(number_match.group(1))
                    if _is_valid_score(score):
                        scores[dim] = score
    
    return scores


def _clamp_score(score: float) -> float:
    """Clamp score to valid range [0, 5]."""
    return max(SCORE_MIN, min(SCORE_MAX, score))


def _is_valid_score(score: float) -> bool:
    """Check if score is within valid range."""
    return SCORE_MIN <= score <= SCORE_MAX


def format_reward_vector(scores: Dict[str, float]) -> str:
    """
    Format scores as reward vector string.
    
    Args:
        scores: Dictionary of dimension scores.
    
    Returns:
        Formatted string representation of reward vector.
    """
    values = [scores.get(dim, 0.0) for dim in DIMENSIONS]
    return f"r = ({', '.join([f'{v:.2f}' for v in values])})"
