"""Corner detection utilities."""

import cv2
import numpy as np
from typing import List, Tuple


def detect_corners_harris(image: np.ndarray, block_size: int = 2, 
                         ksize: int = 3, k: float = 0.04) -> List[Tuple[int, int]]:
    """Detect corners using Harris corner detector."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    
    threshold = 0.01 * dst.max()
    corners = np.argwhere(dst > threshold)
    return [(int(x), int(y)) for y, x in corners]


def detect_corners_shi_tomasi(image: np.ndarray, max_corners: int = 100, 
                               quality_level: float = 0.01, 
                               min_distance: int = 10) -> List[Tuple[int, int]]:
    """Detect corners using Shi-Tomasi method."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    
    if corners is None:
        return []
    
    return [(int(x), int(y)) for corner in corners for x, y in corner]
