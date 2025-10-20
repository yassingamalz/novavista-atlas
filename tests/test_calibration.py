"""Tests for calibration module."""

import pytest
import numpy as np
from atlas.calibration.homography import HomographyCalculator
from atlas.calibration.template_matcher import TemplateMatcher


class TestHomography:
    """Test homography calculation."""
    
    def test_homography_calculator_initialization(self):
        """Test HomographyCalculator initialization."""
        calc = HomographyCalculator()
        assert calc is not None
    
    def test_calculate_homography(self):
        """Test homography calculation."""
        calc = HomographyCalculator()
        src_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst_points = np.array([[10, 10], [110, 5], [105, 115], [5, 110]], dtype=np.float32)
        
        H = calc.calculate(src_points, dst_points)
        assert H is not None
        assert H.shape == (3, 3)
    
    def test_transform_points(self):
        """Test point transformation."""
        calc = HomographyCalculator()
        H = np.eye(3)
        points = np.array([[100, 100], [200, 200]], dtype=np.float32)
        
        transformed = calc.transform_points(points, H)
        assert transformed.shape == points.shape


class TestTemplateMatcher:
    """Test template matching."""
    
    def test_template_matcher_initialization(self):
        """Test TemplateMatcher initialization."""
        matcher = TemplateMatcher()
        assert matcher is not None
        assert matcher.pitch_length == 105.0
        assert matcher.pitch_width == 68.0
    
    def test_get_template_points(self):
        """Test getting template points."""
        matcher = TemplateMatcher()
        points = matcher.get_template_points()
        assert points is not None
        assert len(points) > 0
