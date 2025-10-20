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
    
    def test_classify_circle(self):
        """Test circle classification."""
        classifier = LandmarkClassifier()
        result = classifier.classify_circle((0, 0), 9.15)
        assert result == 'center_circle'


class TestGeometryValidator:
    """Test geometry validation."""
    
    def test_validator_initialization(self):
        """Test GeometryValidator initialization."""
        validator = GeometryValidator()
        assert validator is not None
    
    def test_validate_aspect_ratio(self):
        """Test aspect ratio validation."""
        validator = GeometryValidator()
        result = validator.validate_aspect_ratio(105.0, 68.0)
        assert result is True


class TestConfidenceScorer:
    """Test confidence scoring."""
    
    def test_scorer_initialization(self):
        """Test ConfidenceScorer initialization."""
        scorer = ConfidenceScorer()
        assert scorer is not None
    
    def test_calculate_detection_confidence(self):
        """Test detection confidence calculation."""
        scorer = ConfidenceScorer()
        confidence = scorer.calculate_detection_confidence(50, 2.0)
        assert 0.0 <= confidence <= 1.0
