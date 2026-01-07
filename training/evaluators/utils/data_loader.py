"""
Data loading utilities for SummEval dataset.
"""

import os
import pandas as pd
from datasets import Dataset
from typing import Optional, List

# Default data paths
DEFAULT_DATA_PATHS = [
    "/content/drive/MyDrive/ML_datasets/summeval/summeval_cleaned.parquet",
    "/content/drive/MyDrive/summeval/summeval_cleaned.parquet",
    "./summeval_cleaned.parquet"
]


def load_summeval_dataset(
    parquet_path: str = None,
    normalize_scores: bool = False
) -> Dataset:
    """
    Load SummEval dataset from parquet file.
    
    Args:
        parquet_path: Path to parquet file. Uses default paths if None.
        normalize_scores: If True, normalize scores from [0,5] to [0,1].
    
    Returns:
        HuggingFace Dataset object.
    
    Raises:
        FileNotFoundError: If data file cannot be found.
    """
    if parquet_path is None:
        parquet_path = _find_data_file(DEFAULT_DATA_PATHS)
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data file not found: {parquet_path}")
    
    print(f"Loading SummEval dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    dataset = Dataset.from_pandas(df)
    print(f"Loaded {len(dataset)} samples")
    
    # Add reward vector fields
    def add_reward_fields(example):
        scale = 1.0 if not normalize_scores else 5.0
        return {
            **example,
            "r_relevance": float(example["relevance"]) / scale,
            "r_coherence": float(example["coherence"]) / scale,
            "r_consistency": float(example["consistency"]) / scale,
            "r_fluency": float(example["fluency"]) / scale,
            "data_type": "summeval"
        }
    
    dataset = dataset.map(add_reward_fields)
    
    # Print sample statistics
    sample = dataset[0]
    print(f"Sample document: {sample['text'][:100]}...")
    print(f"Sample summary: {sample['summary'][:100]}...")
    print(f"Reward vector: ({sample['r_relevance']:.2f}, {sample['r_coherence']:.2f}, "
          f"{sample['r_consistency']:.2f}, {sample['r_fluency']:.2f})")
    
    return dataset


def load_summeval_test_dataset(
    parquet_path: str = None,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dataset:
    """
    Load SummEval test split.
    
    Uses the same split ratio and seed as training for reproducibility.
    
    Args:
        parquet_path: Path to parquet file.
        test_ratio: Fraction of data to use for testing.
        random_seed: Random seed for reproducible splits.
    
    Returns:
        Test split of SummEval dataset.
    """
    dataset = load_summeval_dataset(parquet_path)
    splits = dataset.shuffle(seed=random_seed).train_test_split(
        test_size=test_ratio, seed=random_seed
    )
    test_dataset = splits["test"]
    print(f"Test set: {len(test_dataset)} samples")
    return test_dataset


def _find_data_file(paths: List[str]) -> str:
    """Find the first existing data file from a list of paths."""
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Data file not found in any of: {paths}")
