"""Circle detection using Hough Circle Transform."""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class CircleDetector:
    """Detects circular features like center circle."""
    
    def __init__(self, dp: float = 1.2, min_dist: int = 100, 
                 param1: int = 50, param2: int = 30):
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
    
    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
              min_radius: int = 10, max_radius: int = 200) -> List[Tuple[int, int, int]]:
        """
        Detect circles in image.
        
        Args:
            image: Input BGR image
            mask: Optional binary mask to limit detection area
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            
        Returns:
            List of circles as (x, y, radius) tuples
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = cv2.medianBlur(gray, 5)
        
        # Apply mask if provided
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.dp, self.min_dist,
                                   param1=self.param1, param2=self.param2,
                                   minRadius=min_radius, maxRadius=max_radius)
        
        if circles is None:
            return []
        
        circles = np.uint16(np.around(circles))
        return [(int(x), int(y), int(r)) for x, y, r in circles[0, :]]
