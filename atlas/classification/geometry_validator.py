"""Geometry validation for pitch features."""

import numpy as np
from typing import List, Tuple


class GeometryValidator:
    """Validate geometric relationships of pitch features."""
    
    def __init__(self, tolerance: float = 5.0):
        self.tolerance = tolerance
    
    def validate_aspect_ratio(self, length: float, width: float, 
                             expected_ratio: float = 1.54) -> bool:
        """Validate pitch aspect ratio."""
        actual_ratio = length / width
        return abs(actual_ratio - expected_ratio) < 0.1
    
    def validate_parallel_lines(self, line1: Tuple[np.ndarray, np.ndarray], 
                               line2: Tuple[np.ndarray, np.ndarray]) -> bool:
        """Check if two lines are parallel."""
        v1 = line1[1] - line1[0]
        v2 = line2[1] - line2[0]
        
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        dot_product = abs(np.dot(v1_norm, v2_norm))
        return dot_product > 0.95
    
    def validate_perpendicular_lines(self, line1: Tuple[np.ndarray, np.ndarray], 
                                    line2: Tuple[np.ndarray, np.ndarray]) -> bool:
        """Check if two lines are perpendicular."""
        v1 = line1[1] - line1[0]
        v2 = line2[1] - line2[0]
        
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        dot_product = abs(np.dot(v1_norm, v2_norm))
        return dot_product < 0.1
    
    def validate_symmetry(self, points_left: np.ndarray, 
                         points_right: np.ndarray, 
                         center: np.ndarray) -> bool:
        """Validate symmetric features."""
        dist_left = np.mean(np.linalg.norm(points_left - center, axis=1))
        dist_right = np.mean(np.linalg.norm(points_right - center, axis=1))
        return abs(dist_left - dist_right) < self.tolerance
