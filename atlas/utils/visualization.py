"""Visualization utilities for debugging and display."""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def draw_field_mask(image: np.ndarray, mask: np.ndarray, 
                   alpha: float = 0.4) -> np.ndarray:
    """Overlay field mask on image."""
    overlay = image.copy()
    overlay[mask > 0] = [0, 255, 0]
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def draw_lines(image: np.ndarray, lines: List[Tuple[int, int, int, int]], 
              color: Tuple[int, int, int] = (0, 255, 0), 
              thickness: int = 2) -> np.ndarray:
    """Draw detected lines on image."""
    output = image.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(output, (x1, y1), (x2, y2), color, thickness)
    return output


def draw_circles(image: np.ndarray, circles: List[Tuple[int, int, int]], 
                color: Tuple[int, int, int] = (255, 0, 0), 
                thickness: int = 2) -> np.ndarray:
    """Draw detected circles on image."""
    output = image.copy()
    for x, y, r in circles:
        cv2.circle(output, (x, y), r, color, thickness)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
    return output


def draw_keypoints(image: np.ndarray, keypoints: List, 
                  color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Draw keypoints on image."""
    output = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(output, (x, y), 3, color, -1)
    return output


def draw_homography_grid(image: np.ndarray, H: np.ndarray, 
                        grid_size: int = 10) -> np.ndarray:
    """Draw transformed grid overlay."""
    output = image.copy()
    h, w = image.shape[:2]
    
    for i in range(0, w, grid_size):
        pts = np.array([[i, 0], [i, h]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H)
        cv2.line(output, tuple(transformed[0][0].astype(int)), 
                tuple(transformed[1][0].astype(int)), (255, 255, 0), 1)
    
    for i in range(0, h, grid_size):
        pts = np.array([[0, i], [w, i]], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H)
        cv2.line(output, tuple(transformed[0][0].astype(int)), 
                tuple(transformed[1][0].astype(int)), (255, 255, 0), 1)
    
    return output
