"""Coordinate validation utilities."""

import numpy as np
from typing import Tuple


class CoordinateValidator:
    """Validate coordinate transformations and measurements."""
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
    
    def is_within_pitch(self, world_coords: np.ndarray) -> bool:
        """Check if coordinates are within pitch boundaries."""
        x, y = world_coords[0], world_coords[1]
        return (abs(x) <= self.pitch_length / 2 and 
                abs(y) <= self.pitch_width / 2)
    
    def validate_distance(self, distance: float, 
                         min_dist: float = 0.0, 
                         max_dist: float = 150.0) -> bool:
        """Validate if distance is reasonable."""
        return min_dist <= distance <= max_dist
    
    def validate_transformation(self, H: np.ndarray) -> Tuple[bool, str]:
        """Validate homography matrix properties."""
        if H is None:
            return False, "Matrix is None"
        
        if H.shape != (3, 3):
            return False, "Invalid matrix shape"
        
        det = np.linalg.det(H)
        if abs(det) < 1e-6:
            return False, "Matrix is singular"
        
        if H[2, 2] == 0:
            return False, "Invalid normalization"
        
        return True, "Valid"
