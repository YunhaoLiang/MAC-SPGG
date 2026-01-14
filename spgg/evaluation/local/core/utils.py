"""
Utility functions for local GSM8K evaluation.
"""

import os
import re
import json
import time
import random
import torch
from typing import Dict, List, Any

from .config import SPGGConfig
from .networks import PolicyNetwork
from .encoders import MathStateEncoder


def parse_generation_params(
    raw_params: torch.Tensor,
    config: SPGGConfig,
    agent_id: str = "Agent_Qwen"
) -> Dict[str, Any]:
    """
    Parse raw network output into generation parameters.
    
    Args:
        raw_params: Raw output from policy network (6 dimensions).
        config: SPGG configuration with parameter ranges.
        agent_id: Agent identifier for agent-specific defaults.
    
    Returns:
        Dictionary of generation parameters.
    """
    temperature = torch.clamp(
        torch.sigmoid(raw_params[0]) * (config.temp_max - config.temp_min) + config.temp_min,
        config.temp_min, config.temp_max
    )
    
    top_p = torch.clamp(
        torch.sigmoid(raw_params[1]) * (config.top_p_max - config.top_p_min) + config.top_p_min,
        config.top_p_min, config.top_p_max
    )
    
    top_k = torch.clamp(
        torch.sigmoid(raw_params[2]) * (config.top_k_max - config.top_k_min) + config.top_k_min,
        config.top_k_min, config.top_k_max
    ).int()
    
    # Dynamic max_tokens from raw_params[3]
    max_tokens_raw = torch.sigmoid(raw_params[3]) * (config.max_tokens_max - config.max_tokens_min) + config.max_tokens_min
    max_tokens = int(torch.clamp(max_tokens_raw, config.max_tokens_min, config.max_tokens_max).item())
    
    rep_penalty = torch.clamp(
        torch.sigmoid(raw_params[4]) * (config.repetition_penalty_max - config.repetition_penalty_min) + config.repetition_penalty_min,
        config.repetition_penalty_min, config.repetition_penalty_max
    )
    
    presence_penalty = torch.clamp(
        torch.sigmoid(raw_params[5]) * (config.presence_penalty_max - config.presence_penalty_min) + config.presence_penalty_min,
        config.presence_penalty_min, config.presence_penalty_max
    )
    
    return {
        'temperature': temperature.item(),
        'top_p': top_p.item(),
        'top_k': top_k.item(),
        'max_tokens': max_tokens,
        'repetition_penalty': rep_penalty.item(),
        'presence_penalty': presence_penalty.item(),
        'is_dynamic_params': True
    }


def load_checkpoint_params(checkpoint_path: str) -> Dict[str, Dict]:
    """
    Load trained parameters from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
    
    Returns:
        Dictionary mapping agent IDs to their dynamic generators.
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Checkpoint loaded, rebuilding policy networks...")
    
    # Use current SPGGConfig instead of checkpoint's old config
    config = SPGGConfig()
    checkpoint_agents = checkpoint.get('agents', {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_encoder = MathStateEncoder()
    
    dynamic_generators = {}
    loaded_agents = []
    
    for agent_id in ["Agent_Llama", "Agent_SMOLLM2", "Agent_Qwen"]:
        if agent_id in checkpoint_agents:
            policy_net = PolicyNetwork().to(device)
            checkpoint_policy_data = checkpoint_agents[agent_id]['policy_net']
            
            converted_policy_dict = {}
            for key, value in checkpoint_policy_data.items():
                if not key.startswith('network.'):
                    converted_key = f'network.{key}'
                    converted_policy_dict[converted_key] = value
                else:
                    converted_policy_dict[key] = value
            
            policy_net.load_state_dict(converted_policy_dict)
            policy_net.eval()
            
            dynamic_generators[agent_id] = {
                'policy_network': policy_net,
                'state_encoder': state_encoder,
                'config': config,
                'is_dynamic': True
            }
            loaded_agents.append(agent_id)
    
    print(f"Policy networks rebuilt: {len(loaded_agents)} agents with dynamic parameters")
    return dynamic_generators


def load_gsm8k_samples(json_path: str, max_samples: int = None) -> List[Dict]:
    """
    Load GSM8K samples from JSON file.
    
    Args:
        json_path: Path to the JSON file containing samples.
        max_samples: Maximum number of samples to load (None for all).
    
    Returns:
        List of problem dictionaries with 'question' and 'answer' keys.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        if max_samples and len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
            print(f"Randomly selected {len(samples)} GSM8K problems")
        else:
            print(f"Loaded all {len(samples)} GSM8K problems")
        
        return samples
    except Exception as e:
        print(f"Error loading GSM8K data: {e}")
        return []


def extract_numerical_answer(text: str, show_debug: bool = False) -> str:
    """
    Extract numerical answer from solution text.
    
    Args:
        text: Solution text to extract answer from.
        show_debug: Whether to print debug information.
    
    Returns:
        Extracted answer string.
    """
    text = text.replace(',', '').replace('$', '').replace('\\$', '')
    
    if show_debug:
        print(f"Extracting answer from text (length: {len(text)})")
    
    patterns = [
        r"(?:The answer is|the answer is)\s*(\d+(?:\.\d+)?)",
        r"(?:Answer:|answer:)\s*(\d+(?:\.\d+)?)",
        r"(?:Final answer:|final answer:)\s*(\d+(?:\.\d+)?)",
        r"####\s*(\d+(?:\.\d+)?)",
        r"(?:Therefore|Thus|So),?\s*(?:the answer is|answer is)?\s*(\d+(?:\.\d+)?)",
        r"(?:Total|total|Sum|sum)[:=]?\s*(\d+(?:\.\d+)?)",
        r"(?:equals?|is|=)\s*(\d+(?:\.\d+)?)",
        r"(?:Result|result|Solution|solution)[:=]?\s*(\d+(?:\.\d+)?)",
        r"=\s*(\d+(?:\.\d+)?)\s*(?:$|\n|\.)",
        r"\\boxed\{(\d+(?:\.\d+)?)\}",
        r"\b(\d+(?:\.\d+)?)\b"
    ]
    
    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1]
            if show_debug:
                print(f"Pattern {i+1} matched: {answer}")
            
            try:
                if '.' in answer and float(answer).is_integer():
                    return str(int(float(answer)))
                return answer
            except ValueError:
                continue
    
    all_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    if all_numbers:
        try:
            numbers = [float(n) for n in all_numbers]
            max_number = max(numbers)
            if max_number.is_integer():
                return str(int(max_number))
            return str(max_number)
        except ValueError:
            pass
    
    return "No answer found"


def safe_for_json(obj: Any) -> Any:
    """
    Recursively convert object to JSON-serializable format.
    
    Args:
        obj: Object to convert.
    
    Returns:
        JSON-serializable version of the object.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: safe_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_for_json(v) for v in obj]
    elif hasattr(obj, '__dict__'):
        return f"<{obj.__class__.__name__} object>"
    else:
        return str(obj)


def save_results(
    results: List[Dict],
    result_dir: str,
    batch_id: str = None
) -> str:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of result dictionaries.
        result_dir: Directory to save results.
        batch_id: Optional batch identifier for filename.
    
    Returns:
        Path to the saved file.
    """
    os.makedirs(result_dir, exist_ok=True)
    
    timestamp = int(time.time())
    if batch_id:
        filename = f"gsm8k_local_results_{batch_id}_{timestamp}.json"
    else:
        filename = f"gsm8k_local_results_{timestamp}.json"
    
    filepath = os.path.join(result_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(safe_for_json(results), f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {filepath}")
    return filepath
