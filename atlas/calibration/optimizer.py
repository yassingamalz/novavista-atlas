"""Homography optimization using Levenberg-Marquardt."""

import numpy as np
from scipy.optimize import least_squares
from typing import Optional


class HomographyOptimizer:
    """Refine homography matrix using non-linear optimization."""
    
    def __init__(self, max_iters: int = 100):
        self.max_iters = max_iters
    
    def optimize(self, H: np.ndarray, src_points: np.ndarray, 
                dst_points: np.ndarray) -> np.ndarray:
        """Optimize homography matrix."""
        h_params = H.flatten()[:8]
        
        def residuals(params):
            H_opt = self._params_to_matrix(params)
            transformed = self._transform_points(src_points, H_opt)
            return (transformed - dst_points).flatten()
        
        result = least_squares(residuals, h_params, method='lm', max_nfev=self.max_iters)
        return self._params_to_matrix(result.x)
    
    def _params_to_matrix(self, params: np.ndarray) -> np.ndarray:
        """Convert parameter vector to 3x3 matrix."""
        H = np.append(params, 1).reshape(3, 3)
        return H
    
    def _transform_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Transform points using homography."""
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (H @ points_h.T).T
        return transformed[:, :2] / transformed[:, 2:]
