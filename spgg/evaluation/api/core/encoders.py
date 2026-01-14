"""
State encoders for SPGG framework.

Converts task descriptions, agent context, and historical information
into fixed-size vector representations for policy and value networks.
"""

import re
import hashlib
import numpy as np
import torch
from typing import List, Optional


class MathStateEncoder:
    """
    State encoder specialized for mathematical reasoning tasks.
    
    Encodes math problems into dense vector representations by combining:
    - Semantic embeddings from BERT (or simplified hash-based features)
    - Agent-specific context features
    - Problem structural features (operators, numbers, etc.)
    - Position embeddings for sequential game dynamics
    - History features from previous agent solutions
    
    The encoder supports partial observation mode where only the immediately
    preceding agent's solution is visible.
    
    Attributes:
        bert_model: SentenceTransformer model for text encoding (if available).
        use_bert: Whether BERT-based encoding is available.
        device: Torch device for tensor operations.
    """
    
    def __init__(self):
        """Initialize the state encoder with optional BERT model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_bert = True
        except ImportError:
            print("Warning: sentence_transformers not installed, using simplified encoding")
            self.use_bert = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def encode_state(
        self,
        math_problem: str,
        agent_id: str,
        position: int = 0,
        previous_solutions: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Encode the current game state into a fixed-size vector.
        
        Args:
            math_problem: The mathematical problem text.
            agent_id: Identifier of the current agent.
            position: Position in the sequential game (0-indexed).
            previous_solutions: List of solutions from previous agents.
                In partial observation mode, only the last solution is used.
        
        Returns:
            State tensor of shape (896,) containing concatenated features.
        """
        # Document embedding (384-dim)
        if self.use_bert:
            doc_embedding = self.bert_model.encode(math_problem[:512])
            if len(doc_embedding) < 384:
                doc_embedding = np.pad(doc_embedding, (0, 384 - len(doc_embedding)))
            else:
                doc_embedding = doc_embedding[:384]
        else:
            doc_embedding = self._simple_text_encoding(math_problem, 384)
        
        # Agent context features (64-dim)
        agent_features = self._get_agent_context_features(agent_id)
        
        # Math problem structural features (64-dim)
        math_features = self._get_math_problem_features(math_problem)
        
        # Position embedding (32-dim)
        position_embedding = self._get_position_embedding(position)
        
        # History features from previous solutions (352-dim)
        history_features = self._get_history_features(previous_solutions, math_problem)
        
        # Concatenate all features: 384 + 64 + 64 + 32 + 352 = 896
        state = np.concatenate([
            doc_embedding,
            agent_features,
            math_features,
            position_embedding,
            history_features
        ])
        
        return torch.FloatTensor(state).to(self.device)
    
    def _simple_text_encoding(self, text: str, target_dim: int) -> np.ndarray:
        """
        Fallback encoding using hash-based features when BERT is unavailable.
        
        Args:
            text: Input text to encode.
            target_dim: Target dimension for the output vector.
        
        Returns:
            Feature vector of shape (target_dim,).
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_numbers = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
        
        features = np.zeros(target_dim)
        for i in range(min(len(hash_numbers), target_dim)):
            features[i] = hash_numbers[i] / 255.0
        
        # Add text statistics as additional features
        if target_dim >= 10:
            features[-10] = min(len(text) / 1000.0, 1.0)
            features[-9] = min(len(text.split()) / 200.0, 1.0)
            features[-8] = text.count('?') / max(len(text), 1)
            features[-7] = text.count('.') / max(len(text), 1)
            features[-6] = text.count('+') / max(len(text), 1)
            features[-5] = text.count('-') / max(len(text), 1)
            features[-4] = text.count('*') / max(len(text), 1)
            features[-3] = text.count('/') / max(len(text), 1)
            features[-2] = text.count('=') / max(len(text), 1)
            features[-1] = len([c for c in text if c.isdigit()]) / max(len(text), 1)
        
        return features
    
    def _get_agent_context_features(self, agent_id: str) -> np.ndarray:
        """
        Generate agent-specific context features.
        
        Args:
            agent_id: Identifier of the agent.
        
        Returns:
            Feature vector of shape (64,) encoding agent identity and capabilities.
        """
        agent_map = {
            "Agent_Llama": [1.0, 0.0, 0.0],
            "Agent_SMOLLM2": [0.0, 1.0, 0.0],
            "Agent_Qwen": [0.0, 0.0, 1.0]
        }
        base_features = agent_map.get(agent_id, [0.5, 0.5, 0.5])
        
        features = np.zeros(64)
        features[:3] = base_features
        features[3:6] = [0.7, 0.8, 0.6]  # Capability indicators
        features[6:] = np.random.normal(0, 0.1, 58)
        
        return features
    
    def _get_math_problem_features(self, math_problem: str) -> np.ndarray:
        """
        Extract structural features from the math problem.
        
        Args:
            math_problem: The mathematical problem text.
        
        Returns:
            Feature vector of shape (64,) encoding problem structure.
        """
        features = np.zeros(64)
        
        # Length features
        features[0] = min(len(math_problem) / 500.0, 1.0)
        features[1] = min(len(math_problem.split()) / 100.0, 1.0)
        
        # Operator frequency features
        math_symbols = ['+', '-', '*', '/', '=', '(', ')', '$', '%']
        for i, symbol in enumerate(math_symbols):
            if i < 50:
                features[2 + i] = math_problem.count(symbol) / max(len(math_problem), 1)
        
        # Number statistics
        numbers = re.findall(r'\d+', math_problem)
        features[55] = len(numbers) / max(len(math_problem.split()), 1)
        features[56] = np.mean([len(n) for n in numbers]) / 10.0 if numbers else 0
        features[57:] = np.random.normal(0, 0.1, 7)
        
        return features
    
    def _get_position_embedding(self, position: int) -> np.ndarray:
        """
        Generate position embedding for sequential game dynamics.
        
        Args:
            position: Position in the sequential game.
        
        Returns:
            Position embedding vector of shape (32,).
        """
        embedding = np.zeros(32)
        
        # One-hot encoding for positions 0-29
        if position < 30:
            embedding[position] = 1.0
        
        # Continuous position features
        embedding[30] = position / 10.0
        embedding[31] = np.sin(position * np.pi / 10.0)
        
        return embedding
    
    def _get_history_features(
        self,
        previous_solutions: Optional[List[str]],
        current_problem: str
    ) -> np.ndarray:
        """
        Extract features from previous agent solutions.
        
        In partial observation mode, only the immediately preceding solution
        is used for feature extraction.
        
        Args:
            previous_solutions: List of solutions from previous agents.
            current_problem: The current math problem (for computing overlap).
        
        Returns:
            History feature vector of shape (352,).
        """
        features = np.zeros(352)
        
        if not previous_solutions:
            return features
        
        # Partial observation: only process the last solution
        if previous_solutions:
            solution = previous_solutions[-1]
            if solution:
                features[0] = min(len(solution) / 300.0, 1.0)
                features[1] = min(len(solution.split()) / 100.0, 1.0)
                
                # Word overlap with current problem
                common_words = set(solution.lower().split()) & set(current_problem.lower().split())
                features[2] = len(common_words) / max(len(solution.split()), 1)
                features[3:117] = np.random.normal(0, 0.1, 114)
        
        return features
