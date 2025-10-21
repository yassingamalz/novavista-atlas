"""Corner detection utilities."""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class CornerDetector:
    """Corner detection for pitch landmarks."""
    
    def __init__(self, method: str = "shi_tomasi", max_corners: int = 100, 
                 quality_level: float = 0.01, min_distance: int = 10):
        """
        Initialize corner detector.
        
        Args:
            method: Detection method ("harris" or "shi_tomasi")
            max_corners: Maximum number of corners to detect
            quality_level: Quality level for corner detection
            min_distance: Minimum distance between corners
        """
        self.method = method
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
    
    def detect(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Detect corners in image.
        
        Args:
            image: Input BGR image
            mask: Optional binary mask to limit detection area
            
        Returns:
            List of corner coordinates as (x, y) tuples
        """
        if self.method == "harris":
            return self.detect_corners_harris(image, mask)
        else:
            return self.detect_corners_shi_tomasi(image, mask)
    
    def detect_corners_harris(self, image: np.ndarray, mask: Optional[np.ndarray] = None, 
                             block_size: int = 2, ksize: int = 3, 
                             k: float = 0.04) -> List[Tuple[int, int]]:
        """Detect corners using Harris corner detector."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, block_size, ksize, k)
        dst = cv2.dilate(dst, None)
        
        # Apply mask if provided
        if mask is not None:
            dst = dst * (mask > 0)
        
        threshold = 0.01 * dst.max()
        corners = np.argwhere(dst > threshold)
        corners = [(int(x), int(y)) for y, x in corners]
        
        # Limit number of corners
        return corners[:self.max_corners]
    
    def detect_corners_shi_tomasi(self, image: np.ndarray, 
                                  mask: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """Detect corners using Shi-Tomasi method."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        corners = cv2.goodFeaturesToTrack(
            gray, 
            self.max_corners, 
            self.quality_level, 
            self.min_distance,
            mask=mask
        )
        
        if corners is None:
            return []
        
        return [(int(x), int(y)) for corner in corners for x, y in corner]


# Utility functions for backward compatibility
def detect_corners_harris(image: np.ndarray, block_size: int = 2, 
                         ksize: int = 3, k: float = 0.04) -> List[Tuple[int, int]]:
    """Detect corners using Harris corner detector."""
    detector = CornerDetector(method="harris")
    return detector.detect_corners_harris(image, None, block_size, ksize, k)


def detect_corners_shi_tomasi(image: np.ndarray, max_corners: int = 100, 
                               quality_level: float = 0.01, 
                               min_distance: int = 10) -> List[Tuple[int, int]]:
    """Detect corners using Shi-Tomasi method."""
    detector = CornerDetector(method="shi_tomasi", max_corners=max_corners,
                             quality_level=quality_level, min_distance=min_distance)
    return detector.detect_corners_shi_tomasi(image)
