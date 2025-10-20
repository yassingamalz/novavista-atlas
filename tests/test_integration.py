"""Integration tests for complete pipeline."""

import pytest
import numpy as np
import cv2
from atlas.preprocessing.field_segmentation import FieldSegmenter
from atlas.detection.line_detector import LineDetector
from atlas.calibration.homography import HomographyCalculator


class TestIntegration:
    """Test complete processing pipeline."""
    
    def test_basic_pipeline(self):
        """Test basic field detection pipeline."""
        # Create test image
        test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_image[100:620, 140:1140] = [50, 200, 50]  # Green field
        
        # Add white lines
        cv2.rectangle(test_image, (140, 100), (1140, 620), (255, 255, 255), 2)
        cv2.line(test_image, (640, 100), (640, 620), (255, 255, 255), 2)
        
        # Segment field
        segmenter = FieldSegmenter()
        mask = segmenter.segment_field(test_image)
        assert mask is not None
        
        # Detect lines
        line_detector = LineDetector()
        lines = line_detector.detect(test_image, mask)
        assert len(lines) > 0
    
    def test_homography_pipeline(self):
        """Test calibration pipeline."""
        # Create corresponding points
        src_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst_points = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32)
        
        # Calculate homography
        calc = HomographyCalculator()
        H = calc.calculate(src_points, dst_points)
        assert H is not None
        
        # Calculate error
        error = calc.calculate_reprojection_error(src_points, dst_points, H)
        assert error < 5.0
