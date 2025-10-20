"""Confidence scoring for detections."""

import numpy as np
from typing import Dict, List


class ConfidenceScorer:
    """Calculate confidence scores for detections."""
    
    def __init__(self):
        self.weights = {
            'reprojection_error': 0.3,
            'feature_count': 0.2,
            'geometric_consistency': 0.25,
            'coverage': 0.25
        }
    
    def calculate_detection_confidence(self, n_features: int, 
                                      reprojection_error: float,
                                      max_error: float = 10.0) -> float:
        """Calculate confidence for feature detection."""
        feature_score = min(n_features / 50.0, 1.0)
        error_score = max(0, 1.0 - (reprojection_error / max_error))
        return (feature_score + error_score) / 2
    
    def calculate_landmark_confidence(self, geometric_valid: bool, 
                                     position_error: float,
                                     max_error: float = 5.0) -> float:
        """Calculate confidence for landmark detection."""
        geometry_score = 1.0 if geometric_valid else 0.5
        position_score = max(0, 1.0 - (position_error / max_error))
        return (geometry_score + position_score) / 2
    
    def calculate_overall_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall confidence."""
        weighted_sum = sum(scores.get(key, 0) * weight 
                          for key, weight in self.weights.items())
        return min(weighted_sum, 1.0)
