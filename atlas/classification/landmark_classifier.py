"""Landmark classification for pitch features."""

import numpy as np
from typing import Dict, List, Tuple, Optional


class LandmarkClassifier:
    """Classify detected features as pitch landmarks."""
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
    
    def classify_rectangle(self, corners: np.ndarray) -> Optional[str]:
        """Classify rectangle by size ratios."""
        if len(corners) != 4:
            return None
        
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])
        area = width * height
        
        if 16.0 < width < 17.0 and 40.0 < height < 41.0:
            return 'penalty_area'
        elif 5.0 < width < 6.5 and 18.0 < height < 19.0:
            return 'goal_area'
        elif abs(width - self.pitch_length) < 5 and abs(height - self.pitch_width) < 5:
            return 'pitch_boundary'
        
        return 'unknown'
    
    def classify_circle(self, center: Tuple[float, float], radius: float) -> str:
        """Classify circle by radius."""
        if 8.0 < radius < 10.0:
            return 'center_circle'
        elif 0.5 < radius < 2.0:
            return 'corner_arc'
        return 'unknown'
    
    def identify_penalty_spot(self, point: np.ndarray, 
                             goal_line: np.ndarray) -> bool:
        """Check if point is likely a penalty spot."""
        distance = np.abs(np.linalg.norm(point - goal_line))
        return 10.5 < distance < 12.0
