"""Line detection using Hough Transform."""

import cv2
import numpy as np
from typing import List, Tuple


class LineDetector:
    """Detects pitch lines using Hough Transform."""
    
    def __init__(self, rho: float = 1, theta: float = np.pi/180, threshold: int = 100):
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
    
    def detect(self, image: np.ndarray, mask: np.ndarray = None) -> List[Tuple[int, int, int, int]]:
        """Detect lines in image."""
        edges = cv2.Canny(image, 50, 150)
        if mask is not None:
            edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        lines = cv2.HoughLinesP(edges, self.rho, self.theta, self.threshold,
                                minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return []
        
        return [(int(x1), int(y1), int(x2), int(y2)) for line in lines for x1, y1, x2, y2 in line]
    
    def filter_by_angle(self, lines: List[Tuple[int, int, int, int]], 
                       angle_range: Tuple[float, float]) -> List[Tuple[int, int, int, int]]:
        """Filter lines by angle range."""
        filtered = []
        for x1, y1, x2, y2 in lines:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if angle_range[0] <= angle <= angle_range[1]:
                filtered.append((x1, y1, x2, y2))
        return filtered
