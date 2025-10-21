"""Image enhancement module for field detection."""

import cv2
import numpy as np
from typing import Tuple


class ImageEnhancer:
    """Image enhancement for better field detection."""
    
    def __init__(self, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)):
        """
        Initialize image enhancer.
        
        Args:
            clip_limit: CLAHE clip limit
            tile_size: CLAHE tile grid size
        """
        self.clip_limit = clip_limit
        self.tile_size = tile_size
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full enhancement pipeline.
        
        Args:
            image: Input BGR image
            
        Returns:
            Enhanced image
        """
        # Apply CLAHE for better contrast
        enhanced = self.apply_clahe(image)
        
        # Reduce noise
        enhanced = self.reduce_noise(enhanced)
        
        return enhanced
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def enhance_lines(self, image: np.ndarray) -> np.ndarray:
        """Enhance white lines on the field."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        return enhanced
    
    def reduce_noise(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur to reduce noise."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# Utility functions for backward compatibility
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    enhancer = ImageEnhancer(clip_limit, tile_size)
    return enhancer.apply_clahe(image)


def enhance_lines(image: np.ndarray) -> np.ndarray:
    """Enhance white lines on the field."""
    enhancer = ImageEnhancer()
    return enhancer.enhance_lines(image)


def reduce_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to reduce noise."""
    enhancer = ImageEnhancer()
    return enhancer.reduce_noise(image, kernel_size)
