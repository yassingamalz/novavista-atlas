"""Coordinate transformation utilities."""

import numpy as np
from typing import Tuple, Union


class CoordinateTransformer:
    """Transform between pixel and real-world coordinates."""
    
    def __init__(self, homography_matrix: np.ndarray):
        self.H = homography_matrix
        self.H_inv = np.linalg.inv(homography_matrix)
    
    def pixel_to_world(self, pixel_coords: Union[np.ndarray, Tuple[float, float]]) -> np.ndarray:
        """Transform pixel coordinates to real-world meters."""
        if isinstance(pixel_coords, tuple):
            pixel_coords = np.array([pixel_coords])
        
        points_h = np.hstack([pixel_coords, np.ones((pixel_coords.shape[0], 1))])
        world_h = (self.H @ points_h.T).T
        world_coords = world_h[:, :2] / world_h[:, 2:]
        return world_coords
    
    def world_to_pixel(self, world_coords: Union[np.ndarray, Tuple[float, float]]) -> np.ndarray:
        """Transform real-world coordinates to pixel coordinates."""
        if isinstance(world_coords, tuple):
            world_coords = np.array([world_coords])
        
        points_h = np.hstack([world_coords, np.ones((world_coords.shape[0], 1))])
        pixel_h = (self.H_inv @ points_h.T).T
        pixel_coords = pixel_h[:, :2] / pixel_h[:, 2:]
        return pixel_coords
    
    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance in world coordinates."""
        world1 = self.pixel_to_world(point1)
        world2 = self.pixel_to_world(point2)
        return float(np.linalg.norm(world2 - world1))
