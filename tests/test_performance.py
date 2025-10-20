"""Performance tests."""

import pytest
import numpy as np
import time
from atlas.preprocessing.field_segmentation import FieldSegmenter
from atlas.detection.line_detector import LineDetector
from atlas.utils.metrics import PerformanceMetrics


class TestPerformance:
    """Test performance benchmarks."""
    
    def test_field_segmentation_speed(self):
        """Test field segmentation performance."""
        segmenter = FieldSegmenter()
        test_image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        
        start = time.time()
        mask = segmenter.segment_field(test_image)
        duration = (time.time() - start) * 1000
        
        assert duration < 500  # Should complete in under 500ms
    
    def test_line_detection_speed(self):
        """Test line detection performance."""
        detector = LineDetector()
        test_image = np.random.randint(0, 256, (1080, 1920), dtype=np.uint8)
        
        start = time.time()
        lines = detector.detect(test_image)
        duration = (time.time() - start) * 1000
        
        assert duration < 1000  # Should complete in under 1 second
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        metrics = PerformanceMetrics()
        
        metrics.start_timer('test_operation')
        time.sleep(0.1)
        duration = metrics.stop_timer('test_operation')
        
        assert 90 < duration < 150  # Should be around 100ms
        
        summary = metrics.get_summary()
        assert 'test_operation' in summary
