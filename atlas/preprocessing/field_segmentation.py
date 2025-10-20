"""
Field Segmentation Module
Isolates playing surface from surroundings using color-based segmentation
"""

import cv2
import numpy as np
from typing import Tuple


def segment_field(
    frame: np.ndarray,
    hsv_lower: Tuple[int, int, int] = (35, 40, 40),
    hsv_upper: Tuple[int, int, int] = (85, 255, 255),
    morph_kernel: int = 5
) -> np.ndarray:
    """
    Segment field from frame using HSV color thresholding
    
    Args:
        frame: Input BGR frame
        hsv_lower: Lower HSV threshold for green
        hsv_upper: Upper HSV threshold for green
        morph_kernel: Kernel size for morphological operations
        
    Returns:
        Binary mask (1=field, 0=background)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    
    # Morphological operations
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    
    return mask
