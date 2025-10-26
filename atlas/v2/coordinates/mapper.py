"""
Atlas v2 - Coordinate Mapper
Maps between image coordinates and field coordinates using homography
"""

import numpy as np
import cv2
from typing import Tuple


class Mapper:
    """Map between image and field coordinates using homography"""
    
    def __init__(self, homography: np.ndarray):
        """
        Initialize coordinate mapper
        
        Args:
            homography: 3x3 homography matrix from calibration
        """
        self.H = homography
        self.H_inv = np.linalg.inv(homography)
    
    def image_to_field(self, u: float, v: float) -> Tuple[float, float]:
        """
        Convert image pixel coordinates to field coordinates
        
        Args:
            u: Image x-coordinate (pixels, horizontal)
            v: Image y-coordinate (pixels, vertical)
            
        Returns:
            (x, y): Field coordinates in meters
                   x: Along field length (0 to 105m)
                   y: Along field width (0 to 68m)
        """
        # Create homogeneous coordinate
        point_2d = np.array([[u, v]], dtype=np.float32)
        point_2d_hom = np.hstack([point_2d, np.ones((1, 1))])
        
        # Apply inverse homography
        point_3d_hom = (self.H_inv @ point_2d_hom.T).T
        
        # Normalize by homogeneous coordinate
        point_3d = point_3d_hom[:, :2] / point_3d_hom[:, 2:]
        
        return float(point_3d[0, 0]), float(point_3d[0, 1])
    
    def field_to_image(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert field coordinates to image pixel coordinates
        
        Args:
            x: Field x-coordinate (meters, 0-105)
            y: Field y-coordinate (meters, 0-68)
            
        Returns:
            (u, v): Image coordinates in pixels
        """
        # Create point in field coordinates
        point_3d = np.array([[x, y]], dtype=np.float32)
        
        # Apply homography using OpenCV
        point_2d = cv2.perspectiveTransform(
            point_3d.reshape(-1, 1, 2),
            self.H
        )
        
        return int(point_2d[0, 0, 0]), int(point_2d[0, 0, 1])
    
    def batch_image_to_field(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Convert multiple image points to field coordinates
        
        Args:
            points_2d: (N, 2) array of image coordinates [u, v]
            
        Returns:
            (N, 2) array of field coordinates [x, y] in meters
        """
        # Add homogeneous coordinate
        points_2d_hom = np.hstack([points_2d, np.ones((len(points_2d), 1))])
        
        # Apply inverse homography
        points_3d_hom = (self.H_inv @ points_2d_hom.T).T
        
        # Normalize
        points_3d = points_3d_hom[:, :2] / points_3d_hom[:, 2:]
        
        return points_3d
    
    def batch_field_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Convert multiple field points to image coordinates
        
        Args:
            points_3d: (N, 2) array of field coordinates [x, y] in meters
            
        Returns:
            (N, 2) array of image coordinates [u, v] in pixels
        """
        points_2d = cv2.perspectiveTransform(
            points_3d.reshape(-1, 1, 2).astype(np.float32),
            self.H
        )
        
        return points_2d.reshape(-1, 2)
    
    def compute_distance(self, u1: float, v1: float, 
                         u2: float, v2: float) -> float:
        """
        Compute real-world distance between two image points
        
        Args:
            u1, v1: First point in image coordinates
            u2, v2: Second point in image coordinates
            
        Returns:
            Distance in meters
        """
        # Convert both points to field coordinates
        x1, y1 = self.image_to_field(u1, v1)
        x2, y2 = self.image_to_field(u2, v2)
        
        # Euclidean distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return distance
    
    def get_field_zone(self, x: float, y: float) -> str:
        """
        Determine which zone of the field a point is in
        
        Args:
            x: Field x-coordinate (0-105m)
            y: Field y-coordinate (0-68m)
            
        Returns:
            Zone name: 'defensive_third', 'middle_third', or 'attacking_third'
        """
        if x < 35:
            return 'defensive_third'
        elif x < 70:
            return 'middle_third'
        else:
            return 'attacking_third'
    
    def is_in_penalty_area(self, x: float, y: float, 
                           attacking: bool = True) -> bool:
        """
        Check if a point is inside penalty area
        
        Args:
            x: Field x-coordinate (0-105m)
            y: Field y-coordinate (0-68m)
            attacking: If True, check attacking penalty area (far end)
                      If False, check defensive penalty area (near end)
            
        Returns:
            True if point is inside penalty area
        """
        # Penalty area: 16.5m from goal line, 40.3m wide (centered)
        penalty_depth = 16.5
        penalty_width = 40.3
        center_y = 34  # Half of 68m
        
        if attacking:
            # Far penalty area (x > 105 - 16.5 = 88.5)
            in_depth = x >= (105 - penalty_depth)
        else:
            # Near penalty area (x < 16.5)
            in_depth = x <= penalty_depth
        
        # Check width
        in_width = abs(y - center_y) <= (penalty_width / 2)
        
        return in_depth and in_width
