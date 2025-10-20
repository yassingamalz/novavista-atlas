"""Template matching for pitch model."""

import numpy as np
from typing import Dict, List, Tuple


class TemplateMatcher:
    """Match detected features to standard pitch template."""
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.template = self._create_template()
    
    def _create_template(self) -> Dict[str, np.ndarray]:
        """Create standard pitch template with key points."""
        half_length = self.pitch_length / 2
        half_width = self.pitch_width / 2
        
        template = {
            'corners': np.array([
                [-half_length, -half_width],
                [half_length, -half_width],
                [half_length, half_width],
                [-half_length, half_width]
            ]),
            'center': np.array([[0, 0]]),
            'penalty_spots': np.array([
                [-40.5, 0],
                [40.5, 0]
            ]),
            'center_circle_points': self._generate_circle_points(0, 0, 9.15, 36)
        }
        return template
    
    def _generate_circle_points(self, cx: float, cy: float, 
                               radius: float, n_points: int) -> np.ndarray:
        """Generate points on a circle."""
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        x = cx + radius * np.cos(angles)
        y = cy + radius * np.sin(angles)
        return np.column_stack([x, y])
    
    def get_template_points(self) -> np.ndarray:
        """Get all template points as flat array."""
        all_points = []
        for key, points in self.template.items():
            all_points.append(points)
        return np.vstack(all_points)
