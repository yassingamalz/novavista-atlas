"""Homography calculation and transformation."""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class HomographyCalculator:
    """Calculate homography transformation matrix."""
    
    def __init__(self, ransac_threshold: float = 5.0, max_iters: int = 2000):
        self.ransac_threshold = ransac_threshold
        self.max_iters = max_iters
    
    def calculate(self, src_points: np.ndarray, 
                 dst_points: np.ndarray) -> Optional[np.ndarray]:
        """Calculate homography matrix using RANSAC."""
        if len(src_points) < 4 or len(dst_points) < 4:
            return None
        
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 
                                     self.ransac_threshold, maxIters=self.max_iters)
        return H
    
    def transform_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Transform points using homography matrix."""
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (H @ points_homogeneous.T).T
        transformed = transformed[:, :2] / transformed[:, 2:]
        return transformed
    
    def calculate_reprojection_error(self, src_points: np.ndarray, 
                                    dst_points: np.ndarray, 
                                    H: np.ndarray) -> float:
        """Calculate average reprojection error."""
        transformed = self.transform_points(src_points, H)
        errors = np.linalg.norm(transformed - dst_points, axis=1)
        return float(np.mean(errors))
