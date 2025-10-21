"""Tests for config module."""

import pytest
from atlas.config import DEFAULT_CONFIG


class TestConfig:
    """Test configuration module."""
    
    def test_default_config_exists(self):
        """Test that default config exists."""
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, dict)
    
    def test_preprocessing_config(self):
        """Test preprocessing configuration."""
        assert 'preprocessing' in DEFAULT_CONFIG
        preproc = DEFAULT_CONFIG['preprocessing']
        
        assert 'hsv_lower' in preproc
        assert 'hsv_upper' in preproc
        assert len(preproc['hsv_lower']) == 3
        assert len(preproc['hsv_upper']) == 3
        
        assert 'blur_kernel' in preproc
        assert 'morph_kernel' in preproc
    
    def test_detection_config(self):
        """Test detection configuration."""
        assert 'detection' in DEFAULT_CONFIG
        detect = DEFAULT_CONFIG['detection']
        
        assert 'canny_low' in detect
        assert 'canny_high' in detect
        assert 'hough_threshold' in detect
        assert 'min_line_length' in detect
        assert 'max_line_gap' in detect
    
    def test_calibration_config(self):
        """Test calibration configuration."""
        assert 'calibration' in DEFAULT_CONFIG
        calib = DEFAULT_CONFIG['calibration']
        
        assert 'ransac_threshold' in calib
        assert 'ransac_iterations' in calib
        assert 'min_matches' in calib
    
    def test_pitch_config(self):
        """Test pitch configuration."""
        assert 'pitch' in DEFAULT_CONFIG
        pitch = DEFAULT_CONFIG['pitch']
        
        assert 'length_meters' in pitch
        assert 'width_meters' in pitch
        assert pitch['length_meters'] == 105.0
        assert pitch['width_meters'] == 68.0
    
    def test_config_values_valid(self):
        """Test that config values are sensible."""
        # HSV values should be in valid range
        hsv_lower = DEFAULT_CONFIG['preprocessing']['hsv_lower']
        hsv_upper = DEFAULT_CONFIG['preprocessing']['hsv_upper']
        
        assert all(0 <= v <= 255 for v in hsv_lower)
        assert all(0 <= v <= 255 for v in hsv_upper)
        
        # Thresholds should be positive
        assert DEFAULT_CONFIG['calibration']['ransac_threshold'] > 0
        assert DEFAULT_CONFIG['calibration']['ransac_iterations'] > 0
        
        # Pitch dimensions should be reasonable
        assert 90 <= DEFAULT_CONFIG['pitch']['length_meters'] <= 120
        assert 45 <= DEFAULT_CONFIG['pitch']['width_meters'] <= 90
