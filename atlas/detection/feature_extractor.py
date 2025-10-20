"""Feature extraction and matching."""

import cv2
import numpy as np
from typing import List, Tuple


class FeatureExtractor:
    """Extract and match keypoints for homography estimation."""
    
    def __init__(self, n_features: int = 500):
        self.n_features = n_features
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def extract_keypoints(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Extract ORB keypoints and descriptors."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      max_distance: float = 50) -> List[cv2.DMatch]:
        """Match features between two descriptor sets."""
        if desc1 is None or desc2 is None:
            return []
        
        matches = self.bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < max_distance]
        return good_matches
