"""Circle detection using Hough Circle Transform."""

import cv2
import numpy as np
from typing import List, Tuple


class CircleDetector:
    """Detects circular features like center circle."""
    
    def __init__(self, dp: float = 1.2, min_dist: int = 100, 
                 param1: int = 50, param2: int = 30):
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
    
    def detect(self, image: np.ndarray, min_radius: int = 10, 
              max_radius: int = 200) -> List[Tuple[int, int, int]]:
        """Detect circles in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.dp, self.min_dist,
                                   param1=self.param1, param2=self.param2,
                                   minRadius=min_radius, maxRadius=max_radius)
        
        if circles is None:
            return []
        
        circles = np.uint16(np.around(circles))
        return [(int(x), int(y), int(r)) for x, y, r in circles[0, :]]
