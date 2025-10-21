"""Tests for preprocessing module."""

import pytest
import numpy as np
import cv2
from atlas.preprocessing.field_segmentation import FieldSegmenter
from atlas.preprocessing.enhancement import apply_clahe, enhance_lines, reduce_noise
from atlas.preprocessing.masking import apply_morphological_ops, get_largest_component


class TestFieldSegmentation:
    """Test field segmentation functionality."""
    
    def test_segmenter_initialization(self):
        """Test FieldSegmenter can be initialized."""
        segmenter = FieldSegmenter()
        assert segmenter is not None
        assert segmenter.lower_green is not None
        assert segmenter.upper_green is not None
    
    def test_segment_field_returns_mask(self):
        """Test that segment_field returns a valid mask."""
        segmenter = FieldSegmenter()
        test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_image[100:600, 200:1000] = [50, 200, 50]  # Green region
        
        mask = segmenter.segment_field(test_image)
        assert mask is not None
        assert mask.shape == (720, 1280)
        assert mask.dtype == np.uint8


class TestEnhancement:
    """Test image enhancement functions."""
    
    def test_apply_clahe(self):
        """Test CLAHE enhancement."""
        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        enhanced = apply_clahe(test_image)
        assert enhanced.shape == test_image.shape
    
    def test_enhance_lines(self):
        """Test line enhancement."""
        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        enhanced = enhance_lines(test_image)
        assert enhanced.shape == (480, 640)
    
    def test_reduce_noise(self):
        """Test noise reduction."""
        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        smoothed = reduce_noise(test_image)
        assert smoothed.shape == test_image.shape


class TestMasking:
    """Test masking utilities."""
    
    def test_morphological_ops(self):
        """Test morphological operations."""
        test_mask = np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255
        cleaned = apply_morphological_ops(test_mask)
        assert cleaned.shape == test_mask.shape
    
    def test_largest_component(self):
        """Test largest component extraction."""
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        test_mask[20:40, 20:40] = 255
        test_mask[60:80, 60:80] = 255
        largest = get_largest_component(test_mask)
        assert largest.shape == test_mask.shape
    
    def test_apply_convex_hull(self):
        """Test convex hull application."""
        from atlas.preprocessing.masking import apply_convex_hull
        
        # Create a mask with an irregular shape
        test_mask = np.zeros((200, 200), dtype=np.uint8)
        # Create a C-shaped region (not convex)
        cv2.rectangle(test_mask, (50, 50), (150, 150), 255, -1)
        cv2.rectangle(test_mask, (75, 75), (125, 125), 0, -1)  # Remove center
        
        hull_mask = apply_convex_hull(test_mask)
        
        assert hull_mask.shape == test_mask.shape
        assert hull_mask.dtype == np.uint8
        # Convex hull should fill in the hole
        # The area of convex hull should be >= original area
        assert np.sum(hull_mask) >= np.sum(test_mask)
    
    def test_apply_convex_hull_empty_mask(self):
        """Test convex hull on empty mask."""
        from atlas.preprocessing.masking import apply_convex_hull
        
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        hull_mask = apply_convex_hull(test_mask)
        
        assert hull_mask.shape == test_mask.shape
        # Empty mask should remain empty
        assert np.sum(hull_mask) == 0
    
    def test_apply_convex_hull_single_component(self):
        """Test convex hull on single convex component."""
        from atlas.preprocessing.masking import apply_convex_hull
        
        # Create a simple rectangle (already convex)
        test_mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(test_mask, (50, 50), (150, 150), 255, -1)
        
        hull_mask = apply_convex_hull(test_mask)
        
        assert hull_mask.shape == test_mask.shape
        # For a rectangle, convex hull should be very similar
        # (may differ slightly at edges due to contour approximation)
        assert np.abs(np.sum(hull_mask) - np.sum(test_mask)) < 1000
    
    def test_apply_convex_hull_complex_shape(self):
        """Test convex hull on complex irregular shape."""
        from atlas.preprocessing.masking import apply_convex_hull
        
        # Create a star-like shape (very non-convex)
        test_mask = np.zeros((300, 300), dtype=np.uint8)
        center = (150, 150)
        # Draw a pentagon
        points = []
        for i in range(5):
            angle = i * 2 * np.pi / 5
            x = int(center[0] + 80 * np.cos(angle))
            y = int(center[1] + 80 * np.sin(angle))
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(test_mask, [points], 255)
        
        hull_mask = apply_convex_hull(test_mask)
        
        assert hull_mask.shape == test_mask.shape
        # Convex hull of pentagon should have larger area
        assert np.sum(hull_mask) > np.sum(test_mask)
    
    def test_apply_convex_hull_preserves_dimensions(self):
        """Test that convex hull preserves image dimensions."""
        from atlas.preprocessing.masking import apply_convex_hull
        
        # Test different image sizes
        for height, width in [(100, 100), (200, 300), (480, 640)]:
            test_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(test_mask, (width // 2, height // 2), 30, 255, -1)
            
            hull_mask = apply_convex_hull(test_mask)
            
            assert hull_mask.shape == (height, width)
            assert hull_mask.dtype == np.uint8
