"""Coordinate transformation utilities."""

import numpy as np
from typing import Tuple, Union, Optional


class CoordinateTransformer:
    """Transform between pixel and real-world coordinates."""
    
    def __init__(self, homography_matrix: Optional[np.ndarray] = None):
        """
        Initialize coordinate transformer.
        
        Args:
            homography_matrix: Optional 3x3 homography matrix
        """
        self.H = None
        self.H_inv = None
        if homography_matrix is not None:
            self.set_homography(homography_matrix)
    
    def set_homography(self, homography_matrix: np.ndarray):
        """
        Set or update the homography matrix.
        
        Args:
            homography_matrix: 3x3 homography matrix
        """
        self.H = homography_matrix
        try:
            self.H_inv = np.linalg.inv(homography_matrix)
        except np.linalg.LinAlgError:
            self.H_inv = None
    
    def pixel_to_world(self, pixel_coords: Union[np.ndarray, Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Transform pixel coordinates to real-world meters.
        
        Args:
            pixel_coords: Pixel coordinates as (x, y) tuple or array
            
        Returns:
            World coordinates in meters, or None if no homography set
        """
        if self.H is None:
            return None
            
        if isinstance(pixel_coords, tuple):
            pixel_coords = np.array([pixel_coords])
        
        points_h = np.hstack([pixel_coords, np.ones((pixel_coords.shape[0], 1))])
        world_h = (self.H @ points_h.T).T
        world_coords = world_h[:, :2] / world_h[:, 2:]
        return world_coords
    
    def world_to_pixel(self, world_coords: Union[np.ndarray, Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Transform real-world coordinates to pixel coordinates.
        
        Args:
            world_coords: World coordinates in meters
            
        Returns:
            Pixel coordinates, or None if no homography set
        """
        if self.H_inv is None:
            return None
            
        if isinstance(world_coords, tuple):
            world_coords = np.array([world_coords])
        
        points_h = np.hstack([world_coords, np.ones((world_coords.shape[0], 1))])
        pixel_h = (self.H_inv @ points_h.T).T
        pixel_coords = pixel_h[:, :2] / pixel_h[:, 2:]
        return pixel_coords
    
    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> Optional[float]:
        """
        Calculate Euclidean distance in world coordinates.
        
        Args:
            point1: First point in pixel coordinates
            point2: Second point in pixel coordinates
            
        Returns:
            Distance in meters, or None if no homography set
        """
        if self.H is None:
            return None
            
        world1 = self.pixel_to_world(point1)
        world2 = self.pixel_to_world(point2)
        
        if world1 is None or world2 is None:
            return None
            
        return float(np.linalg.norm(world2 - world1))
