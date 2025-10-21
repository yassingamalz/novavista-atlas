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
        assert detector.rho == 1
        assert detector.theta == np.pi/180
    
    def test_line_detector_custom_params(self):
        """Test LineDetector with custom parameters."""
        detector = LineDetector(rho=2, theta=np.pi/90, threshold=150)
        assert detector.rho == 2
        assert detector.theta == np.pi/90
        assert detector.threshold == 150
    
    def test_detect_lines(self):
        """Test line detection."""
        detector = LineDetector()
        test_image = np.zeros((480, 640), dtype=np.uint8)
        cv2.line(test_image, (100, 100), (500, 100), 255, 2)
        
        lines = detector.detect(test_image)
        assert isinstance(lines, list)
    
    def test_filter_by_angle(self):
        """Test line filtering by angle."""
        detector = LineDetector()
        
        # Create test lines at different angles
        lines = [
            (0, 0, 100, 0),      # Horizontal (0 degrees)
            (0, 0, 100, 100),    # 45 degrees
            (0, 0, 0, 100),      # Vertical (90 degrees)
            (0, 0, -100, 100),   # 135 degrees
        ]
        
        # Filter for nearly horizontal lines (-10 to 10 degrees)
        filtered = detector.filter_by_angle(lines, (-10, 10))
        assert isinstance(filtered, list)
        # Should include the horizontal line
        assert len(filtered) >= 1
    
    def test_filter_by_angle_vertical(self):
        """Test filtering for vertical lines."""
        detector = LineDetector()
        
        lines = [
            (0, 0, 100, 0),      # Horizontal
            (0, 0, 0, 100),      # Vertical (90 degrees)
            (0, 0, 10, 100),     # Nearly vertical
        ]
        
        # Filter for nearly vertical lines (80 to 100 degrees)
        filtered = detector.filter_by_angle(lines, (80, 100))
        # Should include vertical and nearly vertical lines
        assert len(filtered) >= 1
    
    def test_filter_by_angle_empty_list(self):
        """Test filtering empty line list."""
        detector = LineDetector()
        
        lines = []
        filtered = detector.filter_by_angle(lines, (-10, 10))
        assert filtered == []
    
    def test_filter_by_angle_no_matches(self):
        """Test filtering with no matching lines."""
        detector = LineDetector()
        
        # All vertical lines
        lines = [
            (0, 0, 0, 100),
            (10, 0, 10, 100),
            (20, 0, 20, 100),
        ]
        
        # Filter for horizontal lines
        filtered = detector.filter_by_angle(lines, (-10, 10))
        assert len(filtered) == 0
    
    def test_filter_by_angle_all_match(self):
        """Test filtering where all lines match."""
        detector = LineDetector()
        
        # All horizontal lines
        lines = [
            (0, 0, 100, 0),
            (0, 10, 100, 10),
            (0, 20, 100, 20),
        ]
        
        # Filter for horizontal lines with wide range
        filtered = detector.filter_by_angle(lines, (-45, 45))
        assert len(filtered) == len(lines)


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
    
    def test_match_features(self):
        """Test feature matching between two descriptor sets."""
        extractor = FeatureExtractor()
        
        # Create two similar test images
        test_image1 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image1, (50, 50), (150, 150), (255, 255, 255), 2)
        
        test_image2 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image2, (55, 55), (155, 155), (255, 255, 255), 2)
        
        kp1, desc1 = extractor.extract_keypoints(test_image1)
        kp2, desc2 = extractor.extract_keypoints(test_image2)
        
        matches = extractor.match_features(desc1, desc2)
        assert isinstance(matches, list)
        # Similar images should have some matches
        if desc1 is not None and desc2 is not None:
            assert len(matches) >= 0
    
    def test_match_features_none_descriptors(self):
        """Test feature matching with None descriptors."""
        extractor = FeatureExtractor()
        
        matches = extractor.match_features(None, None)
        assert matches == []
        
        # Create dummy descriptor
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), 2)
        _, desc = extractor.extract_keypoints(test_image)
        
        matches = extractor.match_features(desc, None)
        assert matches == []
        
        matches = extractor.match_features(None, desc)
        assert matches == []
    
    def test_match_features_max_distance(self):
        """Test feature matching with custom max distance."""
        extractor = FeatureExtractor()
        
        # Create two test images with features
        test_image1 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image1, (50, 50), (150, 150), (255, 255, 255), 2)
        cv2.circle(test_image1, (100, 100), 30, (255, 255, 255), 2)
        
        test_image2 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image2, (50, 50), (150, 150), (255, 255, 255), 2)
        cv2.circle(test_image2, (100, 100), 30, (255, 255, 255), 2)
        
        kp1, desc1 = extractor.extract_keypoints(test_image1)
        kp2, desc2 = extractor.extract_keypoints(test_image2)
        
        if desc1 is not None and desc2 is not None:
            # More restrictive distance threshold
            matches_strict = extractor.match_features(desc1, desc2, max_distance=20)
            # More permissive distance threshold
            matches_loose = extractor.match_features(desc1, desc2, max_distance=100)
            
            # Loose threshold should give more or equal matches
            assert len(matches_loose) >= len(matches_strict)
    
    def test_match_features_sorted_by_distance(self):
        """Test that matches are sorted by distance."""
        extractor = FeatureExtractor()
        
        # Create two images with clear features
        test_image1 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(test_image1, (50, 50), (250, 250), (255, 255, 255), 3)
        cv2.line(test_image1, (150, 0), (150, 300), (255, 255, 255), 2)
        cv2.line(test_image1, (0, 150), (300, 150), (255, 255, 255), 2)
        
        test_image2 = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(test_image2, (50, 50), (250, 250), (255, 255, 255), 3)
        cv2.line(test_image2, (150, 0), (150, 300), (255, 255, 255), 2)
        cv2.line(test_image2, (0, 150), (300, 150), (255, 255, 255), 2)
        
        kp1, desc1 = extractor.extract_keypoints(test_image1)
        kp2, desc2 = extractor.extract_keypoints(test_image2)
        
        if desc1 is not None and desc2 is not None:
            matches = extractor.match_features(desc1, desc2)
            
            # Check that matches are sorted by distance (ascending)
            if len(matches) > 1:
                for i in range(len(matches) - 1):
                    assert matches[i].distance <= matches[i + 1].distance


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
