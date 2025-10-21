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
        assert result.shape[1] == 2  # Should return 2D coordinates
    
    def test_pixel_to_world_identity_matrix(self):
        """Test pixel to world with identity transformation."""
        H = np.eye(3)
        transformer = CoordinateTransformer(H)
        result = transformer.pixel_to_world((100, 100))
        # With identity matrix, coordinates should remain the same
        assert np.allclose(result[0], [100, 100])
    
    def test_world_to_pixel(self):
        """Test world to pixel transformation."""
        H = np.eye(3)
        transformer = CoordinateTransformer(H)
        result = transformer.world_to_pixel((50.0, 30.0))
        assert result is not None
        assert result.shape[1] == 2  # Should return 2D coordinates
    
    def test_world_to_pixel_identity_matrix(self):
        """Test world to pixel with identity transformation."""
        H = np.eye(3)
        transformer = CoordinateTransformer(H)
        result = transformer.world_to_pixel((50.0, 30.0))
        # With identity matrix, coordinates should remain the same
        assert np.allclose(result[0], [50.0, 30.0])
    
    def test_pixel_to_world_array_input(self):
        """Test pixel to world with numpy array input."""
        H = np.eye(3)
        transformer = CoordinateTransformer(H)
        points = np.array([[100, 100], [200, 200]])
        result = transformer.pixel_to_world(points)
        assert result.shape == (2, 2)
        assert np.allclose(result, points)
    
    def test_world_to_pixel_array_input(self):
        """Test world to pixel with numpy array input."""
        H = np.eye(3)
        transformer = CoordinateTransformer(H)
        points = np.array([[50.0, 30.0], [25.0, 15.0]])
        result = transformer.world_to_pixel(points)
        assert result.shape == (2, 2)
        assert np.allclose(result, points)
    
    def test_roundtrip_transformation(self):
        """Test that pixel->world->pixel gives original coordinates."""
        H = np.array([[1.2, 0.1, 100],
                      [0.1, 1.3, 50],
                      [0.001, 0.001, 1]])
        transformer = CoordinateTransformer(H)
        
        original_pixel = np.array([[320, 240]])
        world = transformer.pixel_to_world(original_pixel)
        back_to_pixel = transformer.world_to_pixel(world)
        
        # Should get back very close to original
        assert np.allclose(original_pixel, back_to_pixel, atol=1e-5)
    
    def test_calculate_distance(self):
        """Test distance calculation between points."""
        H = np.eye(3)
        transformer = CoordinateTransformer(H)
        
        point1 = np.array([[0, 0]])
        point2 = np.array([[3, 4]])
        distance = transformer.calculate_distance(point1, point2)
        
        # Distance should be 5 (3-4-5 triangle)
        assert np.isclose(distance, 5.0)
        assert isinstance(distance, float)
    
    def test_calculate_distance_same_point(self):
        """Test distance calculation for same point."""
        H = np.eye(3)
        transformer = CoordinateTransformer(H)
        
        point = np.array([[100, 100]])
        distance = transformer.calculate_distance(point, point)
        
        # Distance should be 0
        assert np.isclose(distance, 0.0)
    
    def test_calculate_distance_with_transformation(self):
        """Test distance calculation with non-identity transformation."""
        # Scale transformation (2x in both dimensions)
        H = np.array([[2, 0, 0],
                      [0, 2, 0],
                      [0, 0, 1]])
        transformer = CoordinateTransformer(H)
        
        # In pixel space, distance is 5
        point1 = np.array([[0, 0]])
        point2 = np.array([[3, 4]])
        
        # In world space (scaled by 2), distance should be 10
        distance = transformer.calculate_distance(point1, point2)
        assert np.isclose(distance, 10.0, atol=0.1)


class TestPitchModel:
    """Test pitch model."""
    
    def test_pitch_model_initialization(self):
        """Test PitchModel initialization."""
        model = PitchModel()
        assert model is not None
        assert model.length == 105.0
        assert model.width == 68.0
    
    def test_pitch_model_custom_dimensions(self):
        """Test PitchModel with custom dimensions."""
        model = PitchModel(length=100.0, width=64.0)
        assert model.length == 100.0
        assert model.width == 64.0
    
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
        assert validator.pitch_length == 105.0
        assert validator.pitch_width == 68.0
    
    def test_validator_custom_pitch(self):
        """Test validator with custom pitch dimensions."""
        validator = CoordinateValidator(pitch_length=100.0, pitch_width=64.0)
        assert validator.pitch_length == 100.0
        assert validator.pitch_width == 64.0
    
    def test_is_within_pitch(self):
        """Test pitch boundary check."""
        validator = CoordinateValidator()
        
        # Center point (should be inside)
        result = validator.is_within_pitch(np.array([0, 0]))
        assert result == True
        
        # Point inside pitch
        result = validator.is_within_pitch(np.array([20, 20]))
        assert result == True
        
        # Point at boundary
        result = validator.is_within_pitch(np.array([52.5, 34]))
        assert result == True
        
        # Point outside pitch
        result = validator.is_within_pitch(np.array([100, 50]))
        assert result == False
        
        # Point outside pitch (negative)
        result = validator.is_within_pitch(np.array([-100, -50]))
        assert result == False
    
    def test_validate_distance(self):
        """Test distance validation."""
        validator = CoordinateValidator()
        
        # Valid distance
        result = validator.validate_distance(50.0)
        assert result == True
        
        # Zero distance (valid)
        result = validator.validate_distance(0.0)
        assert result == True
        
        # Maximum valid distance
        result = validator.validate_distance(150.0)
        assert result == True
        
        # Too large
        result = validator.validate_distance(200.0)
        assert result == False
        
        # Negative (invalid)
        result = validator.validate_distance(-10.0)
        assert result == False
    
    def test_validate_distance_custom_range(self):
        """Test distance validation with custom range."""
        validator = CoordinateValidator()
        
        result = validator.validate_distance(5.0, min_dist=1.0, max_dist=10.0)
        assert result == True
        
        result = validator.validate_distance(0.5, min_dist=1.0, max_dist=10.0)
        assert result == False
        
        result = validator.validate_distance(15.0, min_dist=1.0, max_dist=10.0)
        assert result == False
    
    def test_validate_transformation_valid(self):
        """Test transformation validation with valid matrix."""
        validator = CoordinateValidator()
        
        # Valid identity matrix
        H = np.eye(3)
        is_valid, message = validator.validate_transformation(H)
        assert is_valid == True
        assert message == "Valid"
        
        # Valid transformation matrix
        H = np.array([[1.2, 0.1, 100],
                      [0.1, 1.3, 50],
                      [0.001, 0.001, 1]])
        is_valid, message = validator.validate_transformation(H)
        assert is_valid == True
    
    def test_validate_transformation_none(self):
        """Test transformation validation with None."""
        validator = CoordinateValidator()
        
        is_valid, message = validator.validate_transformation(None)
        assert is_valid == False
        assert "None" in message
    
    def test_validate_transformation_wrong_shape(self):
        """Test transformation validation with wrong shape."""
        validator = CoordinateValidator()
        
        H = np.eye(2)  # Wrong shape
        is_valid, message = validator.validate_transformation(H)
        assert is_valid == False
        assert "shape" in message
    
    def test_validate_transformation_singular(self):
        """Test transformation validation with singular matrix."""
        validator = CoordinateValidator()
        
        # Singular matrix (det = 0)
        H = np.zeros((3, 3))
        is_valid, message = validator.validate_transformation(H)
        assert is_valid == False
        assert "singular" in message.lower()
    
    def test_validate_transformation_invalid_normalization(self):
        """Test transformation validation with invalid normalization."""
        validator = CoordinateValidator()
        
        # Matrix with H[2,2] = 0 (also singular, caught first)
        H = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]])
        is_valid, message = validator.validate_transformation(H)
        assert is_valid == False
        # This matrix is singular, so it's caught by that check first
        assert "singular" in message.lower()
