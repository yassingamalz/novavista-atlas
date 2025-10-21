"""Tests for calibration RANSAC module."""

import pytest
import numpy as np
from atlas.calibration.ransac import RANSAC


class TestRANSAC:
    """Test RANSAC algorithm."""
    
    def test_ransac_initialization(self):
        """Test RANSAC initialization."""
        ransac = RANSAC()
        assert ransac is not None
        assert ransac.threshold == 3.0
        assert ransac.max_iters == 1000
        assert ransac.min_samples == 4
    
    def test_ransac_custom_params(self):
        """Test RANSAC with custom parameters."""
        ransac = RANSAC(threshold=5.0, max_iters=500, min_samples=3)
        assert ransac.threshold == 5.0
        assert ransac.max_iters == 500
        assert ransac.min_samples == 3
    
    def test_ransac_line_fitting(self):
        """Test RANSAC with line fitting."""
        # Generate line data with outliers
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1 + np.random.normal(0, 0.5, 50)
        
        # Add outliers
        y[0] = 100
        y[10] = -50
        
        data = np.column_stack([x, y])
        
        def line_model(points):
            if len(points) < 2:
                return None
            p1, p2 = points[:2]
            return (p1, p2)
        
        def line_score(data, model):
            p1, p2 = model
            distances = []
            for point in data:
                # Calculate distance from point to line
                # Using formula: |ax + by + c| / sqrt(a^2 + b^2)
                v = p2 - p1
                w = point - p1
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c2 == 0:
                    d = np.linalg.norm(w)
                else:
                    b = c1 / c2
                    pb = p1 + b * v
                    d = np.linalg.norm(point - pb)
                distances.append(d)
            return np.array(distances)
        
        ransac = RANSAC(threshold=2.0, max_iters=100)
        model, inliers = ransac.fit(data, line_model, line_score)
        
        assert model is not None
        assert len(inliers) > 0
        # Should identify most points as inliers (excluding outliers)
        assert np.sum(inliers) >= 45
    
    def test_ransac_insufficient_data(self):
        """Test RANSAC with insufficient data."""
        ransac = RANSAC(min_samples=4)
        data = np.array([[1, 2], [3, 4]])  # Only 2 points
        
        def dummy_model(points):
            return points
        
        def dummy_score(data, model):
            return np.zeros(len(data))
        
        model, inliers = ransac.fit(data, dummy_model, dummy_score)
        assert model is None
        assert len(inliers) == 0
    
    def test_ransac_circle_fitting(self):
        """Test RANSAC with circle fitting."""
        # Generate circle data
        np.random.seed(42)
        theta = np.linspace(0, 2*np.pi, 100)
        cx, cy, r = 5, 5, 3
        x = cx + r * np.cos(theta) + np.random.normal(0, 0.1, 100)
        y = cy + r * np.sin(theta) + np.random.normal(0, 0.1, 100)
        
        # Add outliers
        x[0] = 50
        y[0] = 50
        
        data = np.column_stack([x, y])
        
        def circle_model(points):
            if len(points) < 3:
                return None
            # Simple circle center estimate
            center = np.mean(points, axis=0)
            radius = np.mean(np.linalg.norm(points - center, axis=1))
            return (center, radius)
        
        def circle_score(data, model):
            center, radius = model
            distances = np.abs(np.linalg.norm(data - center, axis=1) - radius)
            return distances
        
        ransac = RANSAC(threshold=0.5, max_iters=100, min_samples=3)
        model, inliers = ransac.fit(data, circle_model, circle_score)
        
        assert model is not None
        center, radius = model
        assert len(inliers) > 0
        # Should identify most points as inliers
        assert np.sum(inliers) >= 95
        # Center should be close to actual center
        assert np.linalg.norm(center - np.array([cx, cy])) < 0.5
        # Radius should be close to actual radius
        assert abs(radius - r) < 0.5
