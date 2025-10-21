"""Landmark classification for pitch features."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class LandmarkClassifier:
    """Classify detected features as pitch landmarks."""
    
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
    
    def classify(self, lines: List[Tuple[int, int, int, int]] = None,
                circles: List[Tuple[int, int, int]] = None,
                corners: List[Tuple[int, int]] = None,
                homography: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Classify all detected features into pitch landmarks.
        
        Args:
            lines: Detected line segments
            circles: Detected circles (x, y, radius)
            corners: Detected corner points
            homography: Optional homography matrix for validation
            
        Returns:
            Dictionary of classified landmarks
        """
        landmarks = {
            'center_circle': None,
            'center_line': None,
            'penalty_areas': {'left': None, 'right': None},
            'goal_areas': {'left': None, 'right': None},
            'corner_arcs': [],
            'touchlines': [],
            'goal_lines': []
        }
        
        # Classify circles
        if circles:
            for circle in circles:
                x, y, radius = circle
                circle_type = self.classify_circle((x, y), radius)
                
                if circle_type == 'center_circle' and landmarks['center_circle'] is None:
                    landmarks['center_circle'] = {
                        'center': {'x': int(x), 'y': int(y)},
                        'radius': int(radius),
                        'confidence': 0.90
                    }
                elif circle_type == 'corner_arc':
                    landmarks['corner_arcs'].append({
                        'center': {'x': int(x), 'y': int(y)},
                        'radius': int(radius),
                        'confidence': 0.80
                    })
        
        # Classify lines
        if lines:
            # Separate horizontal and vertical lines
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 30 or angle > 150:  # Horizontal-ish
                    horizontal_lines.append(line)
                elif 60 < angle < 120:  # Vertical-ish
                    vertical_lines.append(line)
            
            # Find center line (longest vertical line near center)
            if vertical_lines:
                center_line = max(vertical_lines, 
                                 key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2))
                landmarks['center_line'] = {
                    'start': {'x': int(center_line[0]), 'y': int(center_line[1])},
                    'end': {'x': int(center_line[2]), 'y': int(center_line[3])},
                    'confidence': 0.85
                }
            
            # Store touchlines and goal lines
            landmarks['touchlines'] = [
                {
                    'start': {'x': int(line[0]), 'y': int(line[1])},
                    'end': {'x': int(line[2]), 'y': int(line[3])},
                    'confidence': 0.85
                }
                for line in horizontal_lines[:2]  # Top 2 horizontal lines
            ]
            
            landmarks['goal_lines'] = [
                {
                    'start': {'x': int(line[0]), 'y': int(line[1])},
                    'end': {'x': int(line[2]), 'y': int(line[3])},
                    'confidence': 0.85
                }
                for line in vertical_lines[:2]  # Top 2 vertical lines
            ]
        
        # Add corners if detected
        if corners and len(corners) >= 4:
            # Assume first 4 corners are pitch corners
            pitch_corners = corners[:4]
            landmarks['pitch_corners'] = [
                {'x': int(x), 'y': int(y), 'confidence': 0.85}
                for x, y in pitch_corners
            ]
        
        return landmarks
    
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
        """
        Classify circle by radius.
        
        Args:
            center: Circle center coordinates
            radius: Circle radius (can be in pixels or meters depending on context)
            
        Returns:
            Circle type classification
        """
        # Check for meter-based classification first (FIFA standard)
        if 8.0 < radius < 10.0:  # Center circle: 9.15m ± tolerance
            return 'center_circle'
        elif 0.5 < radius < 1.5:  # Corner arc: 1.0m ± tolerance
            return 'corner_arc'
        
        # Check for pixel-based classification (for image space)
        elif 50 < radius < 150:  # Center circle in pixels
            return 'center_circle'
        elif 5 < radius < 18:  # Corner arc in pixels (tight range)
            return 'corner_arc'
        
        return 'unknown'
    
    def identify_penalty_spot(self, point: np.ndarray, 
                             goal_line: np.ndarray) -> bool:
        """Check if point is likely a penalty spot."""
        distance = np.abs(np.linalg.norm(point - goal_line))
        return 10.5 < distance < 12.0
