"""
State encoders for SPGG mathematical reasoning tasks.
"""

import re
import hashlib
import numpy as np
import torch
from typing import List, Optional


class MathStateEncoder:
    """
    Encodes mathematical problem states for policy network input.
    
    Combines BERT embeddings (or simple text features), agent context,
    mathematical features, position embeddings, and history features.
    """
    
    def __init__(self):
        """Initialize the state encoder."""
        try:
            from sentence_transformers import SentenceTransformer
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_bert = True
        except ImportError:
            print("Warning: sentence_transformers not installed, using simple encoding")
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
        Encode the current state for policy network input.
        
        Args:
            math_problem: The mathematical problem text.
            agent_id: Identifier of the current agent.
            position: Position in the sequential game.
            previous_solutions: Previous solutions from other agents.
        
        Returns:
            Encoded state tensor of dimension 896.
        """
        # Document embedding (384 dim)
        if self.use_bert:
            doc_embedding = self.bert_model.encode(math_problem[:512])
            if len(doc_embedding) < 384:
                doc_embedding = np.pad(doc_embedding, (0, 384 - len(doc_embedding)))
            else:
                doc_embedding = doc_embedding[:384]
        else:
            doc_embedding = self._simple_text_encoding(math_problem, 384)
        
        # Agent context features (64 dim)
        agent_features = self._get_agent_context_features(agent_id)
        
        # Math problem features (64 dim)
        math_features = self._get_math_problem_features(math_problem)
        
        # Position embedding (32 dim)
        position_embedding = self._get_position_embedding(position)
        
        # History features (352 dim)
        history_features = self._get_history_features(previous_solutions, math_problem)
        
        # Concatenate all features (384 + 64 + 64 + 32 + 352 = 896)
        state = np.concatenate([
            doc_embedding,
            agent_features,
            math_features,
            position_embedding,
            history_features
        ])
        
        return torch.FloatTensor(state).to(self.device)
    
    def _simple_text_encoding(self, text: str, target_dim: int) -> np.ndarray:
        """Simple hash-based text encoding when BERT is unavailable."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_numbers = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
        
        features = np.zeros(target_dim)
        for i in range(min(len(hash_numbers), target_dim)):
            features[i] = hash_numbers[i] / 255.0
        
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
        """Get agent-specific context features."""
        agent_map = {
            "Agent_Llama": [1.0, 0.0, 0.0],
            "Agent_SMOLLM2": [0.0, 1.0, 0.0],
            "Agent_Qwen": [0.0, 0.0, 1.0]
        }
        
        base_features = agent_map.get(agent_id, [0.5, 0.5, 0.5])
        features = np.zeros(64)
        features[:3] = base_features
        features[3:6] = [0.7, 0.8, 0.6]
        features[6:] = np.random.normal(0, 0.1, 58)
        
        return features
    
    def _get_math_problem_features(self, math_problem: str) -> np.ndarray:
        """Extract mathematical features from the problem text."""
        features = np.zeros(64)
        
        features[0] = min(len(math_problem) / 500.0, 1.0)
        features[1] = min(len(math_problem.split()) / 100.0, 1.0)
        
        math_symbols = ['+', '-', '*', '/', '=', '(', ')', '$', '%']
        for i, symbol in enumerate(math_symbols):
            if i < 50:
                features[2 + i] = math_problem.count(symbol) / max(len(math_problem), 1)
        
        numbers = re.findall(r'\d+', math_problem)
        features[55] = len(numbers) / max(len(math_problem.split()), 1)
        features[56] = np.mean([len(n) for n in numbers]) / 10.0 if numbers else 0
        features[57:] = np.random.normal(0, 0.1, 7)
        
        return features
    
    def _get_position_embedding(self, position: int) -> np.ndarray:
        """Get position embedding for the agent's position in sequence."""
        embedding = np.zeros(32)
        
        if position < 30:
            embedding[position] = 1.0
        embedding[30] = position / 10.0
        embedding[31] = np.sin(position * np.pi / 10.0)
        
        return embedding
    
    def _get_history_features(
        self,
        previous_solutions: Optional[List[str]],
        current_problem: str
    ) -> np.ndarray:
        """Extract features from previous solutions."""
        features = np.zeros(352)
        
        if not previous_solutions:
            return features
        
        for i, solution in enumerate(previous_solutions[:3]):
            if solution:
                start_idx = i * 117
                features[start_idx] = min(len(solution) / 300.0, 1.0)
                features[start_idx + 1] = min(len(solution.split()) / 100.0, 1.0)
                common_words = set(solution.lower().split()) & set(current_problem.lower().split())
                features[start_idx + 2] = len(common_words) / max(len(solution.split()), 1)
                features[start_idx + 3:start_idx + 117] = np.random.normal(0, 0.1, 114)
        
        return features
