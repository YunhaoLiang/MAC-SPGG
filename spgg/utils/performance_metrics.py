#!/usr/bin/env python3
"""
Performance metrics calculation utilities.
Provides unified metric computation to avoid code duplication.
"""

import numpy as np
from typing import Dict, List, Optional


class PerformanceMetrics:
    """Unified performance metrics calculator."""
    
    @staticmethod
    def calculate_composite_score(
        avg_reward: float, 
        accuracy: float, 
        avg_score: float, 
        reward_weight: float = 1.0, 
        accuracy_weight: float = 2.0, 
        score_weight: float = 1.0
    ) -> float:
        """
        Calculate composite score from multiple metrics.
        
        Args:
            avg_reward: Average reward value.
            accuracy: Accuracy metric.
            avg_score: Average score.
            reward_weight: Weight for reward component.
            accuracy_weight: Weight for accuracy component.
            score_weight: Weight for score component.
            
        Returns:
            Weighted composite score.
        """
        return (avg_reward * reward_weight + 
                accuracy * accuracy_weight + 
                avg_score * score_weight)
    
    @staticmethod
    def find_best_epoch(performance_data: List[Dict]) -> Optional[Dict]:
        """
        Find the epoch with best performance.
        
        Args:
            performance_data: List of epoch performance data.
            
        Returns:
            Best epoch data dictionary.
        """
        if not performance_data:
            return None
        
        best_epoch = None
        best_score = float('-inf')
        
        for epoch_data in performance_data:
            score = PerformanceMetrics.calculate_composite_score(
                epoch_data.get('avg_reward', 0),
                epoch_data.get('accuracy', 0),
                epoch_data.get('avg_score', 0)
            )
            
            if score > best_score:
                best_score = score
                best_epoch = epoch_data.copy()
                best_epoch['composite_score'] = score
        
        return best_epoch
    
    @staticmethod
    def calculate_trends(performance_data: List[Dict]) -> Dict:
        """
        Calculate performance trends over epochs.
        
        Args:
            performance_data: List of epoch performance data.
            
        Returns:
            Dictionary containing trend analysis results.
        """
        if not performance_data:
            return {}
        
        epochs = [d['epoch'] for d in performance_data]
        rewards = [d.get('avg_reward', 0) for d in performance_data]
        accuracies = [d.get('accuracy', 0) for d in performance_data]
        scores = [d.get('avg_score', 0) for d in performance_data]
        
        def calculate_linear_trend(values):
            """Calculate linear trend (slope) of values."""
            if len(values) < 2:
                return 0
            return np.polyfit(range(len(values)), values, 1)[0]
        
        return {
            "reward_trend": calculate_linear_trend(rewards),
            "accuracy_trend": calculate_linear_trend(accuracies),
            "score_trend": calculate_linear_trend(scores),
            "total_epochs": len(epochs),
            "final_metrics": {
                "reward": rewards[-1] if rewards else 0,
                "accuracy": accuracies[-1] if accuracies else 0,
                "score": scores[-1] if scores else 0
            },
            "best_metrics": {
                "reward": max(rewards) if rewards else 0,
                "accuracy": max(accuracies) if accuracies else 0,
                "score": max(scores) if scores else 0
            }
        }
    
    @staticmethod
    def format_performance_summary(epoch_data: Dict) -> str:
        """
        Format performance data into readable summary.
        
        Args:
            epoch_data: Epoch performance data dictionary.
            
        Returns:
            Formatted summary string.
        """
        if not epoch_data:
            return "No performance data available"
        
        composite_score = PerformanceMetrics.calculate_composite_score(
            epoch_data.get('avg_reward', 0),
            epoch_data.get('accuracy', 0),
            epoch_data.get('avg_score', 0)
        )
        
        summary = f"Epoch {epoch_data.get('epoch', 'N/A')} Performance:\n"
        summary += f"  Average Reward: {epoch_data.get('avg_reward', 0):.4f}\n"
        summary += f"  Accuracy: {epoch_data.get('accuracy', 0):.2%}\n"
        summary += f"  Average Score: {epoch_data.get('avg_score', 0):.4f}\n"
        summary += f"  Composite Score: {composite_score:.4f}\n"
        summary += f"  Episodes: {epoch_data.get('num_episodes', 0)}"
        
        return summary
    
    @staticmethod
    def get_trend_description(trend_value: float, threshold: float = 0.001) -> str:
        """
        Get human-readable trend description.
        
        Args:
            trend_value: Trend slope value.
            threshold: Threshold for determining trend direction.
            
        Returns:
            Trend description string.
        """
        if trend_value > threshold:
            return "Increasing"
        elif trend_value < -threshold:
            return "Decreasing"
        else:
            return "Stable"
