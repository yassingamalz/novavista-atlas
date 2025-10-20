"""Tests for coordinates module."""

import pytest
import numpy as np
from atlas.coordinates.transformer import CoordinateTransformer
from atlas.coordinates.pitch_model import PitchModel
from atlas.coordinates.validator import CoordinateValidator


class TestCoordinateTransformer:
    """Test coordinate transformation."""
    
    def test_transformer_initialization(self):
        """Test CoordinateTransformer initialization."""
        H = np.eye(3)
        transformer = CoordinateTransformer(H)
        assert transformer is not None
    
    def test_pixel_to_world(self):
        """Test pixel to world transformation."""
        H = np.eye(3)
        transformer = CoordinateTransformer(H)
        result = transformer.pixel_to_world((100, 100))
        assert result is not None


class TestPitchModel:
    """Test pitch model."""
    
    def test_pitch_model_initialization(self):
        """Test PitchModel initialization."""
        model = PitchModel()
        assert model is not None
        assert model.length == 105.0
        assert model.width == 68.0
    
    def test_get_pitch_boundaries(self):
        """Test getting pitch boundaries."""
        model = PitchModel()
        boundaries = model.get_pitch_boundaries()
        assert boundaries is not None
        assert len(boundaries) == 4


class TestCoordinateValidator:
    """Test coordinate validation."""
    
    def test_validator_initialization(self):
        """Test CoordinateValidator initialization."""
        validator = CoordinateValidator()
        assert validator is not None
    
    def test_is_within_pitch(self):
        """Test pitch boundary check."""
        validator = CoordinateValidator()
        result = validator.is_within_pitch(np.array([0, 0]))
        assert result == True
