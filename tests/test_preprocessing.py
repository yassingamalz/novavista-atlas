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
