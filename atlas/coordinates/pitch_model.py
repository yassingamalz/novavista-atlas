"""Pitch model definition with standard dimensions."""

import numpy as np
from typing import Dict


class PitchModel:
    """Standard football pitch model with FIFA dimensions."""
    
    def __init__(self, length: float = 105.0, width: float = 68.0):
        self.length = length
        self.width = width
        self.half_length = length / 2
        self.half_width = width / 2
        
        self.dimensions = self._create_dimensions()
        self.landmarks = self._create_landmarks()
    
    def _create_dimensions(self) -> Dict[str, float]:
        """Define standard pitch dimensions."""
        return {
            'length': self.length,
            'width': self.width,
            'penalty_area_length': 16.5,
            'penalty_area_width': 40.3,
            'goal_area_length': 5.5,
            'goal_area_width': 18.32,
            'center_circle_radius': 9.15,
            'corner_arc_radius': 1.0,
            'penalty_spot_distance': 11.0,
            'goal_width': 7.32,
            'goal_height': 2.44
        }
    
    def _create_landmarks(self) -> Dict[str, np.ndarray]:
        """Define key landmark positions."""
        return {
            'center': np.array([0, 0]),
            'left_goal': np.array([-self.half_length, 0]),
            'right_goal': np.array([self.half_length, 0]),
            'corners': np.array([
                [-self.half_length, -self.half_width],
                [self.half_length, -self.half_width],
                [self.half_length, self.half_width],
                [-self.half_length, self.half_width]
            ]),
            'penalty_spots': np.array([
                [-self.half_length + 11.0, 0],
                [self.half_length - 11.0, 0]
            ])
        }
    
    def get_pitch_boundaries(self) -> np.ndarray:
        """Get pitch boundary corners."""
        return self.landmarks['corners']
