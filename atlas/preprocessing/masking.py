"""Masking utilities for field isolation."""

import cv2
import numpy as np


def apply_morphological_ops(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological operations to clean mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    """Extract largest connected component."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255


def apply_convex_hull(mask: np.ndarray) -> np.ndarray:
    """Apply convex hull to smooth boundaries."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    result = np.zeros_like(mask)
    cv2.drawContours(result, [hull], 0, 255, -1)
    return result
