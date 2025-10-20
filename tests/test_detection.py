"""Tests for detection module."""

import pytest
import numpy as np
from atlas.detection.line_detector import LineDetector
from atlas.detection.circle_detector import CircleDetector
from atlas.detection.feature_extractor import FeatureExtractor


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
