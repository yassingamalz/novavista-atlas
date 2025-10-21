"""Homography calculation and transformation."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union


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
    
    def estimate(self, matched_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Optional[np.ndarray], int]:
        """
        Estimate homography from matched point pairs.
        
        Args:
            matched_pairs: List of (source_point, template_point) tuples
            
        Returns:
            Tuple of (homography matrix, number of inliers)
        """
        if len(matched_pairs) < 4:
            return None, 0
        
        src_points = np.array([pair[0] for pair in matched_pairs])
        dst_points = np.array([pair[1] for pair in matched_pairs])
        
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC,
                                     self.ransac_threshold, maxIters=self.max_iters)
        
        inliers = np.sum(mask) if mask is not None else 0
        return H, int(inliers)
    
    def transform_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Transform points using homography matrix."""
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (H @ points_homogeneous.T).T
        transformed = transformed[:, :2] / transformed[:, 2:]
        return transformed
    
    def calculate_reprojection_error(self, 
                                    src_points: Union[List[Tuple[np.ndarray, np.ndarray]], np.ndarray],
                                    dst_points: Optional[np.ndarray] = None,
                                    H: Optional[np.ndarray] = None) -> float:
        """
        Calculate average reprojection error.
        
        Args:
            src_points: Either matched pairs list or source points array
            dst_points: Destination points array (if src_points is array)
            H: Homography matrix (if dst_points provided) or second argument (if matched_pairs)
            
        Returns:
            Average reprojection error in pixels
        """
        # Handle both signatures:
        # 1. calculate_reprojection_error(matched_pairs, H)
        # 2. calculate_reprojection_error(src_points, dst_points, H)
        
        if dst_points is None:
            # Signature 1: matched_pairs format
            matched_pairs = src_points
            homography = H
            
            if homography is None or len(matched_pairs) == 0:
                return float('inf')
            
            src_pts = np.array([pair[0] for pair in matched_pairs])
            dst_pts = np.array([pair[1] for pair in matched_pairs])
        else:
            # Signature 2: separate arrays format
            src_pts = src_points
            dst_pts = dst_points
            homography = H
            
            if homography is None:
                return float('inf')
        
        transformed = self.transform_points(src_pts, homography)
        errors = np.linalg.norm(transformed - dst_pts, axis=1)
        return float(np.mean(errors))


# Alias for core.py compatibility
HomographyEstimator = HomographyCalculator
