"""Template matching for pitch model."""

import numpy as np
from typing import Dict, List, Tuple, Optional


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
    
    def match_features(self, lines: List[Tuple[int, int, int, int]] = None,
                      circles: List[Tuple[int, int, int]] = None,
                      corners: List[Tuple[int, int]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Match detected features to template model.
        
        Args:
            lines: Detected line segments
            circles: Detected circles (x, y, radius)
            corners: Detected corner points
            
        Returns:
            List of (detected_point, template_point) pairs for homography estimation
        """
        matched_pairs = []
        
        # Match corners to template corners
        if corners and len(corners) >= 4:
            template_corners = self.template['corners']
            # Simple matching: sort by position and pair with template
            corners_array = np.array(corners[:4])
            for i, corner in enumerate(corners_array):
                if i < len(template_corners):
                    matched_pairs.append((corner, template_corners[i]))
        
        # Match center circle if detected
        if circles:
            # Find largest circle (likely center circle)
            largest_circle = max(circles, key=lambda c: c[2])
            if 50 < largest_circle[2] < 150:  # Reasonable radius range
                center_point = np.array([largest_circle[0], largest_circle[1]])
                template_center = self.template['center'][0]
                matched_pairs.append((center_point, template_center))
        
        # Match lines to template lines (simplified)
        if lines and len(lines) >= 4:
            # Find longest lines (likely touchlines/goal lines)
            sorted_lines = sorted(lines, key=lambda l: 
                                np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2), 
                                reverse=True)
            
            # Match line midpoints to template
            for i, line in enumerate(sorted_lines[:4]):
                midpoint = np.array([(line[0] + line[2])/2, (line[1] + line[3])/2])
                # Use corners as reference
                if i < len(self.template['corners']):
                    matched_pairs.append((midpoint, self.template['corners'][i]))
        
        return matched_pairs
