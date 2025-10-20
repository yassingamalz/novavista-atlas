"""RANSAC implementation for robust estimation."""

import numpy as np
from typing import Tuple, Optional


class RANSAC:
    """RANSAC algorithm for outlier rejection."""
    
    def __init__(self, threshold: float = 3.0, max_iters: int = 1000, 
                 min_samples: int = 4):
        self.threshold = threshold
        self.max_iters = max_iters
        self.min_samples = min_samples
    
    def fit(self, data: np.ndarray, model_func, score_func) -> Tuple[Optional[object], np.ndarray]:
        """Fit model using RANSAC."""
        best_model = None
        best_inliers = np.array([])
        best_score = -np.inf
        
        n_samples = len(data)
        if n_samples < self.min_samples:
            return None, np.array([])
        
        for _ in range(self.max_iters):
            indices = np.random.choice(n_samples, self.min_samples, replace=False)
            sample = data[indices]
            
            model = model_func(sample)
            if model is None:
                continue
            
            scores = score_func(data, model)
            inliers = scores < self.threshold
            score = np.sum(inliers)
            
            if score > best_score:
                best_score = score
                best_model = model
                best_inliers = inliers
        
        return best_model, best_inliers
