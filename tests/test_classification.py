"""Tests for classification module."""

import pytest
import numpy as np
from atlas.classification.landmark_classifier import LandmarkClassifier
from atlas.classification.geometry_validator import GeometryValidator
from atlas.classification.confidence_scorer import ConfidenceScorer


class TestLandmarkClassifier:
    """Test landmark classification."""
    
    def test_classifier_initialization(self):
        """Test LandmarkClassifier initialization."""
        classifier = LandmarkClassifier()
        assert classifier is not None
        assert classifier.pitch_length == 105.0
        assert classifier.pitch_width == 68.0
    
    def test_classifier_custom_pitch(self):
        """Test classifier with custom pitch dimensions."""
        classifier = LandmarkClassifier(pitch_length=100.0, pitch_width=65.0)
        assert classifier.pitch_length == 100.0
        assert classifier.pitch_width == 65.0
    
    def test_classify_circle(self):
        """Test circle classification."""
        classifier = LandmarkClassifier()
        
        # Center circle
        result = classifier.classify_circle((50, 50), 9.15)
        assert result == 'center_circle'
        
        # Corner arc
        result = classifier.classify_circle((0, 0), 1.0)
        assert result == 'corner_arc'
        
        # Unknown
        result = classifier.classify_circle((10, 10), 20.0)
        assert result == 'unknown'
    
    def test_classify_rectangle_penalty_area(self):
        """Test penalty area classification."""
        classifier = LandmarkClassifier()
        
        # Penalty area dimensions (16.5m x 40.32m in real life)
        corners = np.array([[0, 0], [16.5, 0], [16.5, 40.3], [0, 40.3]])
        result = classifier.classify_rectangle(corners)
        assert result == 'penalty_area'
    
    def test_classify_rectangle_goal_area(self):
        """Test goal area classification."""
        classifier = LandmarkClassifier()
        
        # Goal area dimensions (5.5m x 18.32m)
        corners = np.array([[0, 0], [5.5, 0], [5.5, 18.3], [0, 18.3]])
        result = classifier.classify_rectangle(corners)
        assert result == 'goal_area'
    
    def test_classify_rectangle_pitch_boundary(self):
        """Test pitch boundary classification."""
        classifier = LandmarkClassifier()
        
        # Full pitch dimensions
        corners = np.array([[0, 0], [105, 0], [105, 68], [0, 68]])
        result = classifier.classify_rectangle(corners)
        assert result == 'pitch_boundary'
    
    def test_classify_rectangle_unknown(self):
        """Test unknown rectangle."""
        classifier = LandmarkClassifier()
        
        corners = np.array([[0, 0], [30, 0], [30, 30], [0, 30]])
        result = classifier.classify_rectangle(corners)
        assert result == 'unknown'
    
    def test_classify_rectangle_invalid(self):
        """Test invalid rectangle."""
        classifier = LandmarkClassifier()
        
        # Not 4 corners
        corners = np.array([[0, 0], [10, 0], [10, 10]])
        result = classifier.classify_rectangle(corners)
        assert result is None
    
    def test_identify_penalty_spot(self):
        """Test penalty spot identification."""
        classifier = LandmarkClassifier()
        
        # 11m from goal line (penalty spot distance)
        point = np.array([11.0, 34.0])
        goal_line = np.array([0.0, 34.0])
        result = classifier.identify_penalty_spot(point, goal_line)
        assert result == True
        
        # Too far
        point = np.array([20.0, 34.0])
        result = classifier.identify_penalty_spot(point, goal_line)
        assert result == False
        
        # Too close
        point = np.array([5.0, 34.0])
        result = classifier.identify_penalty_spot(point, goal_line)
        assert result == False


class TestGeometryValidator:
    """Test geometry validation."""
    
    def test_validator_initialization(self):
        """Test GeometryValidator initialization."""
        validator = GeometryValidator()
        assert validator is not None
        assert validator.tolerance == 5.0
    
    def test_validator_custom_tolerance(self):
        """Test validator with custom tolerance."""
        validator = GeometryValidator(tolerance=10.0)
        assert validator.tolerance == 10.0
    
    def test_validate_aspect_ratio(self):
        """Test aspect ratio validation."""
        validator = GeometryValidator()
        
        # Valid ratio (105/68 ≈ 1.54)
        result = validator.validate_aspect_ratio(105.0, 68.0)
        assert result == True
        
        # Invalid ratio
        result = validator.validate_aspect_ratio(100.0, 50.0)
        assert result == False
    
    def test_validate_parallel_lines(self):
        """Test parallel line validation."""
        validator = GeometryValidator()
        
        # Parallel horizontal lines
        line1 = (np.array([0, 0]), np.array([10, 0]))
        line2 = (np.array([0, 5]), np.array([10, 5]))
        result = validator.validate_parallel_lines(line1, line2)
        assert result == True
        
        # Non-parallel lines
        line1 = (np.array([0, 0]), np.array([10, 0]))
        line2 = (np.array([0, 0]), np.array([0, 10]))
        result = validator.validate_parallel_lines(line1, line2)
        assert result == False
    
    def test_validate_perpendicular_lines(self):
        """Test perpendicular line validation."""
        validator = GeometryValidator()
        
        # Perpendicular lines
        line1 = (np.array([0, 0]), np.array([10, 0]))
        line2 = (np.array([0, 0]), np.array([0, 10]))
        result = validator.validate_perpendicular_lines(line1, line2)
        assert result == True
        
        # Non-perpendicular lines
        line1 = (np.array([0, 0]), np.array([10, 0]))
        line2 = (np.array([0, 0]), np.array([10, 5]))
        result = validator.validate_perpendicular_lines(line1, line2)
        assert result == False
    
    def test_validate_symmetry(self):
        """Test symmetry validation."""
        validator = GeometryValidator()
        
        center = np.array([50, 50])
        
        # Symmetric points
        points_left = np.array([[40, 50], [40, 45], [40, 55]])
        points_right = np.array([[60, 50], [60, 45], [60, 55]])
        result = validator.validate_symmetry(points_left, points_right, center)
        assert result == True
        
        # Asymmetric points
        points_left = np.array([[40, 50], [40, 45], [40, 55]])
        points_right = np.array([[70, 50], [70, 45], [70, 55]])
        result = validator.validate_symmetry(points_left, points_right, center)
        assert result == False
    
    def test_validate_diagonal_lines(self):
        """Test diagonal line validation."""
        validator = GeometryValidator()
        
        # Parallel diagonal lines
        line1 = (np.array([0, 0]), np.array([10, 10]))
        line2 = (np.array([5, 0]), np.array([15, 10]))
        result = validator.validate_parallel_lines(line1, line2)
        assert result == True


class TestConfidenceScorer:
    """Test confidence scoring."""
    
    def test_scorer_initialization(self):
        """Test ConfidenceScorer initialization."""
        scorer = ConfidenceScorer()
        assert scorer is not None
    
    def test_calculate_detection_confidence(self):
        """Test detection confidence calculation."""
        scorer = ConfidenceScorer()
        confidence = scorer.calculate_detection_confidence(20, 100)
        assert 0 <= confidence <= 1
        assert confidence > 0
