"""Performance metrics and evaluation."""

import numpy as np
from typing import List, Dict
from time import time


class PerformanceMetrics:
    """Track performance metrics."""
    
    def __init__(self):
        self.start_times = {}
        self.durations = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        self.start_times[name] = time()
    
    def stop_timer(self, name: str) -> float:
        """Stop timing and return duration in milliseconds."""
        if name not in self.start_times:
            return 0.0
        duration = (time() - self.start_times[name]) * 1000
        self.durations[name] = duration
        return duration
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all timings."""
        return self.durations.copy()


class AccuracyMetrics:
    """Calculate accuracy metrics."""
    
    @staticmethod
    def calculate_precision_recall(true_positives: int, false_positives: int, 
                                   false_negatives: int) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score."""
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    @staticmethod
    def calculate_reprojection_error(predicted: np.ndarray, 
                                    ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate reprojection error statistics."""
        errors = np.linalg.norm(predicted - ground_truth, axis=1)
        return {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'max_error': float(np.max(errors)),
            'std_error': float(np.std(errors))
        }
