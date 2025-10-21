"""Confidence scoring for detections."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class ConfidenceScorer:
    """Calculate confidence scores for detections."""
    
    def __init__(self):
        self.weights = {
            'reprojection_error': 0.3,
            'feature_count': 0.2,
            'geometric_consistency': 0.25,
            'coverage': 0.25
        }
    
    def calculate_scores(self, field_mask: np.ndarray = None,
                        lines: List[Tuple[int, int, int, int]] = None,
                        circles: List[Tuple[int, int, int]] = None,
                        homography: Optional[np.ndarray] = None,
                        landmarks: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Calculate comprehensive confidence scores for all detections.
        
        Args:
            field_mask: Binary field segmentation mask
            lines: Detected line segments
            circles: Detected circles
            homography: Homography matrix
            landmarks: Classified landmarks
            
        Returns:
            Dictionary of confidence scores
        """
        scores = {}
        
        # Feature count score
        total_features = (len(lines) if lines else 0) + \
                        (len(circles) if circles else 0)
        scores['feature_count'] = min(total_features / 50.0, 1.0)
        
        # Coverage score (from field mask)
        if field_mask is not None:
            coverage = np.sum(field_mask > 0) / field_mask.size
            scores['coverage'] = min(coverage * 2, 1.0)  # Scale to 0-1
        else:
            scores['coverage'] = 0.0
        
        # Homography quality score
        if homography is not None:
            # Simple validation: check if matrix is reasonable
            try:
                det = np.linalg.det(homography[:2, :2])
                scores['homography'] = min(abs(det) / 10.0, 1.0)
            except:
                scores['homography'] = 0.0
        else:
            scores['homography'] = 0.0
        
        # Geometric consistency score
        if lines and len(lines) >= 4:
            scores['geometric_consistency'] = 0.8  # Simplified
        else:
            scores['geometric_consistency'] = 0.3
        
        # Overall weighted confidence
        scores['overall'] = self.calculate_overall_confidence(scores)
        
        return scores
    
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
