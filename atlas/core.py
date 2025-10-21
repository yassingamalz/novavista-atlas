"""
Atlas Core Processor
Main entry point for field detection and calibration
"""

from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import time

from atlas.preprocessing.field_segmentation import FieldSegmenter
from atlas.preprocessing.enhancement import ImageEnhancer
from atlas.detection.line_detector import LineDetector
from atlas.detection.circle_detector import CircleDetector
from atlas.detection.corner_detector import CornerDetector
from atlas.calibration.homography import HomographyEstimator
from atlas.calibration.template_matcher import TemplateMatcher
from atlas.classification.landmark_classifier import LandmarkClassifier
from atlas.classification.confidence_scorer import ConfidenceScorer
from atlas.coordinates.transformer import CoordinateTransformer


class AtlasProcessor:
    """Main processor for field detection and camera calibration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Atlas processor
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.version = "1.0.0"
        
        # Initialize all modules
        self.enhancer = ImageEnhancer()
        self.segmenter = FieldSegmenter()
        self.line_detector = LineDetector()
        self.circle_detector = CircleDetector()
        self.corner_detector = CornerDetector()
        self.homography_estimator = HomographyEstimator()
        self.template_matcher = TemplateMatcher()
        self.landmark_classifier = LandmarkClassifier()
        self.confidence_scorer = ConfidenceScorer()
        self.coordinate_transformer = CoordinateTransformer()
        
    def process_frame(self, frame_input: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Process a single frame for field detection
        
        Args:
            frame_input: Path to image file or numpy array
            
        Returns:
            Dictionary containing detection results in NovaVista Atlas format
        """
        start_time = time.time()
        
        # Load frame
        if isinstance(frame_input, str):
            frame = cv2.imread(frame_input)
            frame_id = Path(frame_input).stem
        else:
            frame = frame_input
            frame_id = f"frame_{int(time.time())}"
            
        if frame is None:
            raise ValueError(f"Failed to load frame from {frame_input}")
        
        try:
            # Step 1: Enhance image
            enhanced = self.enhancer.enhance(frame)
            
            # Step 2: Segment field
            field_mask = self.segmenter.segment_field(enhanced)
            field_pixels = np.sum(field_mask > 0)
            total_pixels = field_mask.size
            field_percentage = (field_pixels / total_pixels) * 100
            
            # Step 3: Detect features
            lines = self.line_detector.detect(enhanced, mask=field_mask)
            circles = self.circle_detector.detect(enhanced, mask=field_mask)
            corners = self.corner_detector.detect(enhanced, mask=field_mask)
            
            # Step 4: Calculate homography
            H = None
            reprojection_error = None
            inliers = 0
            
            if len(lines) >= 4:
                try:
                    matched_pairs = self.template_matcher.match_features(
                        lines=lines,
                        circles=circles,
                        corners=corners
                    )
                    
                    if len(matched_pairs) >= 4:
                        H, inliers = self.homography_estimator.estimate(matched_pairs)
                        reprojection_error = self.homography_estimator.calculate_reprojection_error(
                            matched_pairs, H
                        )
                        self.coordinate_transformer.set_homography(H)
                except Exception as e:
                    # Homography calculation failed, continue without it
                    pass
            
            # Step 5: Classify landmarks
            landmarks = self.landmark_classifier.classify(
                lines=lines,
                circles=circles,
                corners=corners,
                homography=H
            )
            
            # Step 6: Calculate confidence scores
            confidence_scores = self.confidence_scorer.calculate_scores(
                field_mask=field_mask,
                lines=lines,
                circles=circles,
                homography=H,
                landmarks=landmarks
            )
            
            # Determine status
            if field_percentage < 30:
                status = "failed"
            elif H is None or len(lines) < 5:
                status = "partial"
            else:
                status = "success"
            
            # Build output JSON
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            result = {
                "system": "NovaVista Atlas",
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "frame_id": frame_id,
                "status": status,  # Top-level for convenience
                
                "field_detection": {
                    "status": status,
                    "confidence": confidence_scores.get("overall", 0.0),
                    "field_percentage": round(field_percentage, 2),
                    "lines_detected": len(lines),
                    "circles_detected": len(circles),
                    "corners_detected": len(corners),
                },
                
                "calibration": {
                    "homography_matrix": H.tolist() if H is not None else None,
                    "pitch_dimensions": {
                        "length_meters": 105.0,
                        "width_meters": 68.0,
                        "source": "standard"
                    },
                    "transformation_quality": {
                        "reprojection_error": round(reprojection_error, 2) if reprojection_error else None,
                        "confidence": confidence_scores.get("homography", 0.0),
                        "inliers": inliers
                    }
                },
                
                "landmarks": landmarks,
                
                "camera_analysis": {
                    "view_type": self._estimate_view_type(field_percentage, len(lines)),
                    "visible_area_percentage": round(field_percentage, 2)
                },
                
                "processing_metadata": {
                    "processing_time_ms": round(processing_time, 2),
                    "image_size": {
                        "width": frame.shape[1],
                        "height": frame.shape[0]
                    },
                    "algorithm_version": "1.0.0",
                    "gpu_used": False,
                    "warnings": [],
                    "errors": []
                }
            }
            
            return result
            
        except Exception as e:
            # Return error result
            return {
                "system": "NovaVista Atlas",
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "frame_id": frame_id,
                "status": "failed",  # Top-level for convenience
                "field_detection": {
                    "status": "failed",
                    "confidence": 0.0,
                },
                "processing_metadata": {
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "errors": [str(e)]
                }
            }
    
    def _estimate_view_type(self, field_percentage: float, line_count: int) -> str:
        """Estimate camera view type based on field visibility."""
        if field_percentage > 70 and line_count > 10:
            return "broadcast"
        elif field_percentage > 50:
            return "tactical"
        elif field_percentage > 30:
            return "close_up"
        else:
            return "unknown"
