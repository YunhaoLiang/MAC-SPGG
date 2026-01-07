#!/usr/bin/env python3
"""
Reinforcement learning utilities for SPGG.
Provides state encoding functionality for RL training.
"""

import torch
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel


class StateEncoder:
    """
    Encodes text into fixed-dimensional state representations using BERT.
    Used to convert task descriptions and contributions into vectors for policy/value networks.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize state encoder.
        
        Args:
            model_name: HuggingFace model name for encoding.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text into fixed-dimensional vector.
        
        Args:
            text: Input text to encode.
            
        Returns:
            Encoded state tensor.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        outputs = self.model(**inputs)
        state = outputs.last_hidden_state[:, 0, :]
        
        return state.squeeze(0)  
