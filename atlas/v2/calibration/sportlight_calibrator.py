"""
Atlas v2 - Camera Calibrator for Sportlight Integration
Converts Sportlight keypoints + lines to camera parameters and homography
"""

import numpy as np
import cv2
from typing import Dict, Optional


class Calibrator:
    """
    Atlas v2 camera calibration using Sportlight detections
    
    Converts keypoints + lines to camera parameters and homography
    Uses Direct Linear Transform (DLT) for automatic calibration
    """
    
    # FIFA standard field dimensions (meters)
    FIELD_LENGTH = 105.0  
    FIELD_WIDTH = 68.0    
    
    # Standard field keypoint positions in 3D world coordinates (meters)
    # Origin at top-left corner, X = length direction, Y = width direction, Z = height
    FIELD_3D_MODEL = {
        # Corner flags
        'corner_top_left': [0, 0, 0],
        'corner_top_right': [FIELD_LENGTH, 0, 0],
        'corner_bottom_left': [0, FIELD_WIDTH, 0],
        'corner_bottom_right': [FIELD_LENGTH, FIELD_WIDTH, 0],
        
        # Goal posts
        'goal_top_left_post': [0, FIELD_WIDTH/2 - 3.66, 0],
        'goal_top_right_post': [0, FIELD_WIDTH/2 + 3.66, 0],
        'goal_bottom_left_post': [FIELD_LENGTH, FIELD_WIDTH/2 - 3.66, 0],
        'goal_bottom_right_post': [FIELD_LENGTH, FIELD_WIDTH/2 + 3.66, 0],
        
        # Penalty box corners
        'penalty_top_left': [16.5, FIELD_WIDTH/2 - 20.15, 0],
        'penalty_top_right': [16.5, FIELD_WIDTH/2 + 20.15, 0],
        'penalty_bottom_left': [FIELD_LENGTH - 16.5, FIELD_WIDTH/2 - 20.15, 0],
        'penalty_bottom_right': [FIELD_LENGTH - 16.5, FIELD_WIDTH/2 + 20.15, 0],
        
        # Center
        'center_spot': [FIELD_LENGTH/2, FIELD_WIDTH/2, 0],
        'midfield_left': [FIELD_LENGTH/2, 0, 0],
        'midfield_right': [FIELD_LENGTH/2, FIELD_WIDTH, 0],
    }
    
    def __init__(self):
        """Initialize calibrator with standard field model"""
        self.homography = None
        self.camera_matrix = None
        
    def calibrate(self, keypoints: np.ndarray, 
                  lines: np.ndarray,
                  image_shape: tuple) -> Optional[Dict]:
        """
        Calibrate camera from detected features
        
        Args:
            keypoints: (N, 2) detected 2D keypoints in image coordinates
            lines: (M, 4) detected lines [x1, y1, x2, y2] in image coordinates
            image_shape: (height, width) of image
            
        Returns:
            Dict with calibration parameters:
                - homography: 3x3 homography matrix for ground plane mapping
                - valid: bool indicating if calibration succeeded
            Or None if calibration failed
        """
        
        # TODO: Use Sportlight's calibration method when available
        # Sportlight likely provides its own calibration as part of the solution
        # Check their code for exact method
        
        try:
            # Match keypoints to known 3D field positions
            points_2d, points_3d = self._match_keypoints_to_field_model(keypoints)
            
            if len(points_2d) < 4:
                print(f"[Atlas v2] Insufficient point correspondences: {len(points_2d)} < 4")
                return None
            
            # Compute homography for ground plane (Z=0)
            self.homography, mask = cv2.findHomography(
                points_3d[:, :2],  # X, Y only (Z=0)
                points_2d,
                cv2.RANSAC,
                5.0
            )
            
            if self.homography is None:
                print("[Atlas v2] Homography computation failed")
                return None
            
            # Estimate camera matrix from homography
            self._estimate_camera_matrix(image_shape)
            
            return {
                'homography': self.homography,
                'camera_matrix': self.camera_matrix,
                'valid': True
            }
            
        except Exception as e:
            print(f"[Atlas v2] Calibration failed: {e}")
            return None
    
    def _match_keypoints_to_field_model(self, keypoints: np.ndarray) -> tuple:
        """
        Match detected keypoints to 3D field model
        
        Args:
            keypoints: (N, 2) detected 2D keypoints
            
        Returns:
            Tuple of (points_2d, points_3d) - matched correspondence pairs
        """
        # TODO: Implement actual keypoint matching based on Sportlight's keypoint labels
        # For now, use placeholder matching assuming first 4 points are corners
        
        if len(keypoints) < 4:
            return np.array([]), np.array([])
        
        # Placeholder: Assume first 4 keypoints are field corners
        points_2d = keypoints[:4]
        points_3d = np.array([
            self.FIELD_3D_MODEL['corner_top_left'],
            self.FIELD_3D_MODEL['corner_top_right'],
            self.FIELD_3D_MODEL['corner_bottom_left'],
            self.FIELD_3D_MODEL['corner_bottom_right'],
        ])
        
        return points_2d, points_3d
    
    def _estimate_camera_matrix(self, image_shape: tuple):
        """
        Estimate camera intrinsic matrix from image shape
        
        Args:
            image_shape: (height, width) of image
        """
        h, w = image_shape
        
        # Rough estimate: focal length ≈ image width
        focal_length = w
        
        # Principal point at image center
        cx = w / 2
        cy = h / 2
        
        self.camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
