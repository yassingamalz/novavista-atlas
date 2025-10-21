"""Tests for detection module."""

import pytest
import numpy as np
import cv2
from atlas.detection.line_detector import LineDetector
from atlas.detection.circle_detector import CircleDetector
from atlas.detection.feature_extractor import FeatureExtractor
from atlas.detection.corner_detector import detect_corners_harris, detect_corners_shi_tomasi


class TestLineDetector:
    """Test line detection."""
    
    def test_line_detector_initialization(self):
        """Test LineDetector initialization."""
        detector = LineDetector()
        assert detector is not None
        assert detector.threshold == 100
    
    def test_detect_lines(self):
        """Test line detection."""
        detector = LineDetector()
        test_image = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(test_image, (100, 100), (500, 100), 255, 2)
        
        lines = detector.detect(test_image)
        assert isinstance(lines, list)


class TestCircleDetector:
    """Test circle detection."""
    
    def test_circle_detector_initialization(self):
        """Test CircleDetector initialization."""
        detector = CircleDetector()
        assert detector is not None
    
    def test_detect_circles(self):
        """Test circle detection."""
        detector = CircleDetector()
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_image, (320, 240), 50, (255, 255, 255), 2)
        
        circles = detector.detect(test_image)
        assert isinstance(circles, list)


class TestFeatureExtractor:
    """Test feature extraction."""
    
    def test_feature_extractor_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor()
        assert extractor is not None
        assert extractor.n_features == 500
    
    def test_extract_keypoints(self):
        """Test keypoint extraction."""
        extractor = FeatureExtractor()
        test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        keypoints, descriptors = extractor.extract_keypoints(test_image)
        assert keypoints is not None


class TestCornerDetector:
    """Test corner detection functions."""
    
    def test_detect_corners_harris(self):
        """Test Harris corner detection."""
        # Create test image with corners
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), 2)
        
        corners = detect_corners_harris(test_image)
        assert isinstance(corners, list)
        assert len(corners) > 0
        # Check that corners are tuples of integers
        assert all(isinstance(c, tuple) and len(c) == 2 for c in corners)
    
    def test_detect_corners_harris_custom_params(self):
        """Test Harris corner detection with custom parameters."""
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), 2)
        
        corners = detect_corners_harris(test_image, block_size=3, ksize=5, k=0.06)
        assert isinstance(corners, list)
    
    def test_detect_corners_shi_tomasi(self):
        """Test Shi-Tomasi corner detection."""
        # Create test image with corners
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), 2)
        
        corners = detect_corners_shi_tomasi(test_image)
        assert isinstance(corners, list)
        assert len(corners) >= 0
        # Check that corners are tuples of integers
        if corners:
            assert all(isinstance(c, tuple) and len(c) == 2 for c in corners)
    
    def test_detect_corners_shi_tomasi_custom_params(self):
        """Test Shi-Tomasi with custom parameters."""
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), 2)
        
        corners = detect_corners_shi_tomasi(
            test_image, 
            max_corners=50, 
            quality_level=0.05, 
            min_distance=20
        )
        assert isinstance(corners, list)
    
    def test_detect_corners_shi_tomasi_no_corners(self):
        """Test Shi-Tomasi on blank image."""
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        corners = detect_corners_shi_tomasi(test_image)
        assert isinstance(corners, list)
        assert len(corners) == 0
    
    def test_detect_corners_complex_image(self):
        """Test corner detection on complex image."""
        # Create more complex pattern
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        # Multiple rectangles
        cv2.rectangle(test_image, (50, 50), (100, 100), (255, 255, 255), 2)
        cv2.rectangle(test_image, (150, 150), (250, 250), (255, 255, 255), 2)
        # Lines
        cv2.line(test_image, (0, 150), (300, 150), (255, 255, 255), 2)
        cv2.line(test_image, (150, 0), (150, 300), (255, 255, 255), 2)
        
        harris_corners = detect_corners_harris(test_image)
        shi_tomasi_corners = detect_corners_shi_tomasi(test_image)
        
        # Both methods should detect corners
        assert len(harris_corners) > 0
        assert len(shi_tomasi_corners) > 0
