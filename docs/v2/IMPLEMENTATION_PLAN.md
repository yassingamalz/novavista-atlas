# 🛠️ NovaVista Atlas v2 - Implementation Plan

## 🎯 Overview

This document provides **step-by-step implementation instructions** for integrating YOLOv8-seg into Atlas v2. Follow this guide after completing YOLOv8 training and having your production model ready.

---

## 📋 Prerequisites

Before starting implementation:
- [x] Trained YOLOv8-seg model (`final_egyptian.pt`)
- [x] Python 3.8+ installed
- [x] Existing Atlas v1 codebase
- [x] YOLOv8 installed (`pip install ultralytics`)

---

## 🏗️ Implementation Phases

1. **Project Structure Setup** - Create v2 directory structure
2. **YOLO Integration** - Wrap YOLOv8 segmentation
3. **Post-Processing** - Clean YOLO masks
4. **Homography Enhancement** - Add temporal smoothing
5. **Validation Module** - Quality checks
6. **v2 Processor** - Main pipeline
7. **Testing & Deployment** - Validate and deploy

---

## Phase 1: Project Structure Setup

### Step 1.1: Create v2 Directory Structure

```bash
cd novavista-atlas

# Create v2 module structure
mkdir -p atlas/v2/segmentation/weights
mkdir -p atlas/v2/postprocessing
mkdir -p atlas/v2/homography
mkdir -p atlas/v2/validation

# Create __init__ files
touch atlas/v2/__init__.py
touch atlas/v2/segmentation/__init__.py
touch atlas/v2/postprocessing/__init__.py
touch atlas/v2/homography/__init__.py
touch atlas/v2/validation/__init__.py
```

### Step 1.2: Copy Trained Model

```bash
# Copy your trained YOLOv8 model to weights directory
cp /path/to/runs/egyptian/final_model/weights/best.pt atlas/v2/segmentation/weights/final_egyptian.pt

# Optional: Copy ONNX for faster inference
cp /path/to/runs/egyptian/final_model/weights/best.onnx atlas/v2/segmentation/weights/final_egyptian.onnx
```

### Step 1.3: Update Requirements

Add to `requirements.txt`:
```txt
# Existing requirements
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0

# New for v2
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Phase 2: YOLO Integration Module

### Step 2.1: Create YOLO Wrapper

Create `atlas/v2/segmentation/yolo_field_detector.py`:

```python
"""
YOLOv8-seg Field Detector
Production wrapper for soccer field segmentation
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Optional
import time


class FieldDetector:
    """
    Soccer field detector using YOLOv8-seg
    """
    
    def __init__(
        self,
        model_path: str = "atlas/v2/segmentation/weights/final_egyptian.pt",
        confidence_threshold: float = 0.7,
        device: str = None,
        verbose: bool = False
    ):
        """
        Initialize field detector
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence_threshold: Minimum confidence for detection (0.7 recommended)
            device: Device ('cuda' or 'cpu'), auto-detect if None
            verbose: Print inference details
        """
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
        print(f"🔧 Loading YOLOv8-seg model from {model_path}")
        self.model = YOLO(model_path)
        
        # Set device
        if device is None:
            device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        self.device = device
        
        if verbose:
            print(f"   Device: {self.device}")
            print(f"   Confidence threshold: {self.confidence_threshold}")
    
    def detect_field(
        self,
        frame: np.ndarray,
        return_confidence: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Detect soccer field in frame
        
        Args:
            frame: Input BGR frame
            return_confidence: Whether to return confidence score
            
        Returns:
            Binary mask (255=field, 0=background) or None if not detected
            Confidence score (if return_confidence=True)
        """
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=[0],  # class 0 = soccer field
            verbose=False,
            device=self.device
        )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Extract mask
        if len(results) == 0 or len(results[0].masks) == 0:
            if self.verbose:
                print(f"⚠️  No field detected (inference: {inference_time:.1f}ms)")
            return (None, 0.0) if return_confidence else None
        
        # Get first mask (highest confidence)
        mask_data = results[0].masks.data[0].cpu().numpy()
        confidence = float(results[0].boxes.conf[0])
        
        # Resize mask to original frame size
        mask_resized = cv2.resize(
            mask_data,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convert to binary (0 or 255)
        mask_binary = (mask_resized * 255).astype(np.uint8)
        
        if self.verbose:
            print(f"✅ Field detected: conf={confidence:.3f}, time={inference_time:.1f}ms")
        
        if return_confidence:
            return mask_binary, confidence
        return mask_binary
    
    def detect_batch(
        self,
        frames: list,
        return_confidences: bool = False
    ) -> list:
        """
        Detect fields in multiple frames (batched inference)
        
        Args:
            frames: List of BGR frames
            return_confidences: Return confidence scores
            
        Returns:
            List of masks (and confidences if requested)
        """
        # Batch inference
        results = self.model.predict(
            frames,
            conf=self.confidence_threshold,
            classes=[0],
            verbose=False,
            device=self.device,
            stream=True  # Memory-efficient
        )
        
        outputs = []
        for frame, result in zip(frames, results):
            if len(result.masks) == 0:
                mask = None
                conf = 0.0
            else:
                mask_data = result.masks.data[0].cpu().numpy()
                mask_resized = cv2.resize(
                    mask_data,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                mask = (mask_resized * 255).astype(np.uint8)
                conf = float(result.boxes.conf[0])
            
            if return_confidences:
                outputs.append((mask, conf))
            else:
                outputs.append(mask)
        
        return outputs


# Convenience function
def detect_soccer_field(
    frame: np.ndarray,
    model_path: str = "atlas/v2/segmentation/weights/final_egyptian.pt",
    confidence_threshold: float = 0.7
) -> Optional[np.ndarray]:
    """
    Convenience function for field detection
    
    Args:
        frame: Input BGR frame
        model_path: Path to YOLOv8 model
        confidence_threshold: Minimum confidence
        
    Returns:
        Binary mask or None
    """
    detector = FieldDetector(model_path, confidence_threshold)
    return detector.detect_field(frame)


if __name__ == "__main__":
    # Test inference
    print("Testing YOLOv8 field detector...")
    
    # Create dummy frame
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    detector = FieldDetector(
        model_path="atlas/v2/segmentation/weights/final_egyptian.pt",
        verbose=True
    )
    
    mask, confidence = detector.detect_field(test_frame, return_confidence=True)
    
    if mask is not None:
        print(f"✅ Detection successful")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Field pixels: {np.sum(mask == 255):,}")
        print(f"   Confidence: {confidence:.3f}")
    else:
        print("❌ No field detected")
```

---

## Phase 3: Post-Processing Module

### Step 3.1: Create Mask Cleaner

Create `atlas/v2/postprocessing/mask_cleaner.py`:

```python
"""
Post-processing for YOLO field masks
Minimal cleanup since YOLO outputs are already clean
"""

import cv2
import numpy as np
from typing import Dict


class MaskCleaner:
    """
    Clean up YOLO segmentation masks
    """
    
    def __init__(self, morph_kernel: int = 5):
        """
        Initialize cleaner
        
        Args:
            morph_kernel: Kernel size for morphological operations
        """
        self.morph_kernel = morph_kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel, morph_kernel)
        )
    
    def clean(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean binary mask
        
        Args:
            mask: Binary mask (255=field, 0=background)
            
        Returns:
            Cleaned binary mask
        """
        if mask is None:
            return None
        
        # Keep only largest connected component
        mask_cleaned = self._keep_largest_component(mask)
        
        # Small morphological closing to fill tiny gaps
        mask_cleaned = cv2.morphologyEx(
            mask_cleaned,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=1
        )
        
        return mask_cleaned
    
    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
        
        if num_labels <= 1:
            return mask
        
        # Find largest component (excluding background=0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Create mask with only largest component
        mask_cleaned = np.zeros_like(mask)
        mask_cleaned[labels == largest_label] = 255
        
        return mask_cleaned
    
    def calculate_metrics(self, mask: np.ndarray) -> Dict:
        """
        Calculate quality metrics for mask
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary of metrics
        """
        if mask is None:
            return {
                "valid": False,
                "field_percentage": 0.0,
                "field_pixels": 0,
                "num_components": 0
            }
        
        total_pixels = mask.size
        field_pixels = np.sum(mask == 255)
        field_percentage = (field_pixels / total_pixels) * 100
        
        # Count components
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask)
        num_components = num_labels - 1  # Exclude background
        
        return {
            "valid": True,
            "field_percentage": field_percentage,
            "field_pixels": field_pixels,
            "total_pixels": total_pixels,
            "num_components": num_components
        }


def clean_field_mask(mask: np.ndarray, morph_kernel: int = 5) -> np.ndarray:
    """
    Convenience function for mask cleaning
    
    Args:
        mask: Binary mask
        morph_kernel: Kernel size
        
    Returns:
        Cleaned mask
    """
    cleaner = MaskCleaner(morph_kernel)
    return cleaner.clean(mask)
```

---

## Phase 4: Homography with Temporal Smoothing

### Step 4.1: Create Temporal Smoother

Create `atlas/v2/homography/smoother.py`:

```python
"""
Temporal Homography Smoother
Stabilize homography estimates across frames
"""

import numpy as np
from typing import Optional, Tuple, Dict


class HomographySmoother:
    """
    Smooth homography matrices using exponential moving average
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize smoother
        
        Args:
            alpha: Smoothing factor [0,1]
                   0.3 = good balance (recommended)
                   Lower = smoother but slower adaptation
                   Higher = more responsive but less stable
        """
        self.alpha = alpha
        self.H_prev = None
        self.frame_count = 0
    
    def smooth(
        self,
        H_current: Optional[np.ndarray],
        quality_score: float
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Smooth homography estimate
        
        Args:
            H_current: Current homography (3×3) or None if failed
            quality_score: Quality score [0,1] of current estimate
            
        Returns:
            Smoothed homography, state dictionary
        """
        self.frame_count += 1
        
        state = {
            "frame": self.frame_count,
            "smoothed": False,
            "fallback": False,
            "quality": quality_score
        }
        
        # First frame
        if self.H_prev is None:
            if H_current is not None and quality_score > 0.5:
                self.H_prev = H_current.copy()
                return H_current, state
            else:
                state["fallback"] = True
                return None, state
        
        # Current estimation failed
        if H_current is None or quality_score < 0.4:
            # Use previous homography (fallback)
            state["fallback"] = True
            state["smoothed"] = True
            return self.H_prev, state
        
        # Apply EMA smoothing
        H_smooth = self.alpha * H_current + (1 - self.alpha) * self.H_prev
        
        # Normalize (H[2,2] should be 1)
        H_smooth = H_smooth / H_smooth[2, 2]
        
        # Update history
        self.H_prev = H_smooth
        
        state["smoothed"] = True
        return H_smooth, state
    
    def reset(self):
        """Reset smoother state"""
        self.H_prev = None
        self.frame_count = 0
```

### Step 4.2: Create Enhanced Estimator

Create `atlas/v2/homography/estimator_v2.py`:

```python
"""
Enhanced Homography Estimator with Temporal Smoothing
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict

from atlas.v2.homography.smoother import HomographySmoother


class HomographyEstimatorV2:
    """
    Homography estimator with temporal smoothing and validation
    """
    
    def __init__(
        self,
        smoothing_alpha: float = 0.3,
        ransac_threshold: float = 5.0,
        min_inliers: int = 8
    ):
        """
        Initialize estimator
        
        Args:
            smoothing_alpha: Temporal smoothing factor
            ransac_threshold: RANSAC reprojection threshold (pixels)
            min_inliers: Minimum number of inliers required
        """
        self.smoother = HomographySmoother(alpha=smoothing_alpha)
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
    
    def estimate(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Estimate homography with validation and smoothing
        
        Args:
            src_points: Source points (Nx2)
            dst_points: Destination points (Nx2)
            
        Returns:
            Homography matrix (3x3) or None
            Metrics dictionary
        """
        metrics = {
            "num_points": len(src_points),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "quality_score": 0.0
        }
        
        # Need at least 4 points
        if len(src_points) < 4:
            H_smooth, smooth_state = self.smoother.smooth(None, 0.0)
            metrics.update(smooth_state)
            return H_smooth, metrics
        
        # Estimate homography using RANSAC
        H, mask = cv2.findHomography(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold
        )
        
        if H is None:
            H_smooth, smooth_state = self.smoother.smooth(None, 0.0)
            metrics.update(smooth_state)
            return H_smooth, metrics
        
        # Calculate metrics
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(src_points)
        
        metrics["inliers"] = int(inliers)
        metrics["inlier_ratio"] = float(inlier_ratio)
        
        # Quality score based on inlier ratio and count
        quality_score = inlier_ratio * min(1.0, inliers / 12.0)
        metrics["quality_score"] = float(quality_score)
        
        # Temporal smoothing
        H_smooth, smooth_state = self.smoother.smooth(H, quality_score)
        metrics.update(smooth_state)
        
        return H_smooth, metrics
    
    def reset(self):
        """Reset estimator state"""
        self.smoother.reset()
```

---

## Phase 5: Validation Module

### Step 5.1: Create Validator

Create `atlas/v2/validation/validator.py`:

```python
"""
Quality Validation Module
Validate homography and mask quality
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class QualityValidator:
    """
    Validate field detection and homography quality
    """
    
    def __init__(
        self,
        min_field_percentage: float = 30.0,
        max_field_percentage: float = 70.0,
        min_inlier_ratio: float = 0.6
    ):
        """
        Initialize validator
        
        Args:
            min_field_percentage: Minimum field area percentage
            max_field_percentage: Maximum field area percentage
            min_inlier_ratio: Minimum RANSAC inlier ratio
        """
        self.min_field_percentage = min_field_percentage
        self.max_field_percentage = max_field_percentage
        self.min_inlier_ratio = min_inlier_ratio
    
    def validate_mask(self, mask: np.ndarray) -> Dict:
        """
        Validate field mask quality
        
        Args:
            mask: Binary mask
            
        Returns:
            Validation results
        """
        if mask is None:
            return {
                "valid": False,
                "reason": "No mask detected"
            }
        
        total_pixels = mask.size
        field_pixels = np.sum(mask == 255)
        field_percentage = (field_pixels / total_pixels) * 100
        
        # Check field area percentage
        if field_percentage < self.min_field_percentage:
            return {
                "valid": False,
                "reason": f"Field too small ({field_percentage:.1f}%)",
                "field_percentage": field_percentage
            }
        
        if field_percentage > self.max_field_percentage:
            return {
                "valid": False,
                "reason": f"Field too large ({field_percentage:.1f}%)",
                "field_percentage": field_percentage
            }
        
        return {
            "valid": True,
            "field_percentage": field_percentage
        }
    
    def validate_homography(
        self,
        H: np.ndarray,
        metrics: Dict,
        frame_shape: Tuple[int, int]
    ) -> Dict:
        """
        Validate homography quality
        
        Args:
            H: Homography matrix (3x3)
            metrics: Homography metrics from estimator
            frame_shape: Frame shape (height, width)
            
        Returns:
            Validation results
        """
        if H is None:
            return {
                "valid": False,
                "reason": "No homography estimated"
            }
        
        # Check inlier ratio
        inlier_ratio = metrics.get("inlier_ratio", 0.0)
        if inlier_ratio < self.min_inlier_ratio:
            return {
                "valid": False,
                "reason": f"Low inlier ratio ({inlier_ratio:.2f})",
                "inlier_ratio": inlier_ratio
            }
        
        # Check determinant (should be positive)
        det = np.linalg.det(H[:2, :2])
        if det <= 0:
            return {
                "valid": False,
                "reason": "Degenerate homography (det <= 0)",
                "determinant": det
            }
        
        # Check corner projections are within bounds
        h, w = frame_shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
        
        for i, point in enumerate(projected.squeeze()):
            if not (0 <= point[0] < w and 0 <= point[1] < h):
                return {
                    "valid": False,
                    "reason": f"Corner {i} projects outside frame",
                    "projected_corners": projected.squeeze().tolist()
                }
        
        return {
            "valid": True,
            "inlier_ratio": inlier_ratio,
            "determinant": float(det),
            "projected_corners": projected.squeeze().tolist()
        }
```

---

## Phase 6: v2 Processor

### Step 6.1: Create Main Processor

Create `atlas/v2/processor_v2.py`:

```python
"""
Atlas v2 Main Processor
Orchestrates YOLO segmentation + homography pipeline
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import time

from atlas.v2.segmentation.yolo_field_detector import FieldDetector
from atlas.v2.postprocessing.mask_cleaner import MaskCleaner
from atlas.v2.homography.estimator_v2 import HomographyEstimatorV2
from atlas.v2.validation.validator import QualityValidator
from atlas.detection.line_detector import LineDetector  # Reuse from v1
from atlas.detection.corner_detector import CornerDetector  # Reuse from v1


class AtlasProcessorV2:
    """
    NovaVista Atlas v2 - Production field analysis processor
    """
    
    def __init__(
        self,
        model_path: str = "atlas/v2/segmentation/weights/final_egyptian.pt",
        confidence_threshold: float = 0.7,
        smoothing_alpha: float = 0.3,
        verbose: bool = False
    ):
        """
        Initialize Atlas v2 processor
        
        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Field detection threshold
            smoothing_alpha: Temporal smoothing factor
            verbose: Print detailed processing info
        """
        self.verbose = verbose
        
        if verbose:
            print("🚀 Initializing NovaVista Atlas v2...")
        
        # Initialize modules
        self.field_detector = FieldDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            verbose=verbose
        )
        
        self.mask_cleaner = MaskCleaner()
        self.homography_estimator = HomographyEstimatorV2(
            smoothing_alpha=smoothing_alpha
        )
        self.validator = QualityValidator()
        
        # Reuse v1 components
        self.line_detector = LineDetector()
        self.corner_detector = CornerDetector()
        
        if verbose:
            print("✅ Atlas v2 initialized successfully")
    
    def process_frame(
        self,
        frame: np.ndarray,
        template_points: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Process single frame
        
        Args:
            frame: Input BGR frame
            template_points: Field template points for homography (Nx2)
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        result = {
            "success": False,
            "mask": None,
            "homography": None,
            "keypoints": None,
            "confidence": 0.0,
            "processing_time_ms": 0.0,
            "error": None
        }
        
        try:
            # Step 1: Detect field
            mask, confidence = self.field_detector.detect_field(
                frame,
                return_confidence=True
            )
            
            if mask is None:
                result["error"] = "Field not detected"
                result["processing_time_ms"] = (time.time() - start_time) * 1000
                return result
            
            # Step 2: Clean mask
            mask_cleaned = self.mask_cleaner.clean(mask)
            
            # Step 3: Validate mask
            mask_validation = self.validator.validate_mask(mask_cleaned)
            if not mask_validation["valid"]:
                result["error"] = mask_validation["reason"]
                result["mask"] = mask_cleaned
                result["confidence"] = confidence
                result["processing_time_ms"] = (time.time() - start_time) * 1000
                return result
            
            # Step 4: Detect lines (reuse v1)
            lines = self.line_detector.detect(frame, mask_cleaned)
            
            # Step 5: Extract keypoints (reuse v1)
            keypoints = self.corner_detector.extract_corners(lines, mask_cleaned)
            
            # Step 6: Estimate homography (if template provided)
            H = None
            H_metrics = {}
            if template_points is not None and keypoints is not None and len(keypoints) >= 4:
                H, H_metrics = self.homography_estimator.estimate(
                    keypoints,
                    template_points
                )
                
                # Validate homography
                H_validation = self.validator.validate_homography(
                    H,
                    H_metrics,
                    frame.shape[:2]
                )
                
                if not H_validation["valid"]:
                    if self.verbose:
                        print(f"⚠️  Homography validation failed: {H_validation['reason']}")
            
            # Success
            result["success"] = True
            result["mask"] = mask_cleaned
            result["homography"] = H
            result["keypoints"] = keypoints
            result["confidence"] = confidence
            result["homography_metrics"] = H_metrics
            result["mask_validation"] = mask_validation
            
        except Exception as e:
            result["error"] = str(e)
            if self.verbose:
                print(f"❌ Processing error: {e}")
        
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        return result
    
    def process_video(
        self,
        video_path: str,
        template_points: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        skip_frames: int = 1
    ) -> list:
        """
        Process video file
        
        Args:
            video_path: Path to input video
            template_points: Field template points
            output_path: Optional path for output video with overlays
            skip_frames: Process every Nth frame (1=all frames)
            
        Returns:
            List of results per processed frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        results = []
        frame_idx = 0
        
        # Video writer if output requested
        writer = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if requested
            if frame_idx % skip_frames == 0:
                result = self.process_frame(frame, template_points)
                result["frame_index"] = frame_idx
                results.append(result)
                
                # Draw overlay if writing output
                if writer and result["success"]:
                    frame_overlay = self._draw_overlay(frame, result)
                    writer.write(frame_overlay)
                elif writer:
                    writer.write(frame)
                
                if self.verbose and frame_idx % 30 == 0:
                    print(f"Processed frame {frame_idx}, Success: {result['success']}")
            
            frame_idx += 1
        
        cap.release()
        if writer:
            writer.release()
        
        if self.verbose:
            success_count = sum(1 for r in results if r["success"])
            print(f"✅ Video processing complete:")
            print(f"   Total frames: {len(results)}")
            print(f"   Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
        
        return results
    
    def _draw_overlay(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw field mask and keypoints overlay"""
        overlay = frame.copy()
        
        # Draw mask
        if result["mask"] is not None:
            mask_color = np.zeros_like(frame)
            mask_color[result["mask"] == 255] = [0, 255, 0]  # Green
            overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
        
        # Draw keypoints
        if result["keypoints"] is not None:
            for point in result["keypoints"]:
                cv2.circle(overlay, tuple(point.astype(int)), 5, (0, 0, 255), -1)
        
        # Draw confidence
        text = f"Conf: {result['confidence']:.3f}"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return overlay
    
    def reset(self):
        """Reset processor state (temporal smoothing, etc.)"""
        self.homography_estimator.reset()


if __name__ == "__main__":
    # Test processor
    print("Testing Atlas v2 processor...")
    
    processor = AtlasProcessorV2(verbose=True)
    
    # Test on dummy frame
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    result = processor.process_frame(test_frame)
    
    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Processing time: {result['processing_time_ms']:.1f}ms")
    if result['error']:
        print(f"  Error: {result['error']}")
```

---

## Phase 7: Testing & Deployment

### Step 7.1: Create Test Script

Create `tests/test_atlas_v2.py`:

```python
"""
Test Atlas v2 on Egyptian league footage
"""

import cv2
import numpy as np
from atlas.v2.processor_v2 import AtlasProcessorV2


def test_single_frame(image_path: str):
    """Test on single image"""
    print(f"Testing on {image_path}")
    
    processor = AtlasProcessorV2(verbose=True)
    frame = cv2.imread(image_path)
    
    result = processor.process_frame(frame)
    
    print(f"\n✅ Results:")
    print(f"   Success: {result['success']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
    
    # Save visualization
    if result['mask'] is not None:
        output_path = image_path.replace('.jpg', '_result.jpg')
        overlay = processor._draw_overlay(frame, result)
        cv2.imwrite(output_path, overlay)
        print(f"   Saved result: {output_path}")


def test_video(video_path: str, output_path: str = None):
    """Test on video"""
    print(f"Testing on video: {video_path}")
    
    processor = AtlasProcessorV2(verbose=True)
    
    results = processor.process_video(
        video_path,
        output_path=output_path,
        skip_frames=1  # Process every frame
    )
    
    # Calculate statistics
    success_count = sum(1 for r in results if r["success"])
    avg_time = np.mean([r["processing_time_ms"] for r in results])
    avg_conf = np.mean([r["confidence"] for r in results if r["success"]])
    
    print(f"\n✅ Video Processing Results:")
    print(f"   Total frames: {len(results)}")
    print(f"   Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    print(f"   Avg processing time: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
    print(f"   Avg confidence: {avg_conf:.3f}")


if __name__ == "__main__":
    # Test on sample images
    test_single_frame("data/test_images/cairo_stadium_day.jpg")
    test_single_frame("data/test_images/alexandria_night.jpg")
    test_single_frame("data/test_images/broadcast_angle.jpg")
    
    # Test on video
    test_video(
        "data/test_videos/egyptian_match.mp4",
        output_path="data/test_videos/egyptian_match_result.mp4"
    )
```

### Step 7.2: Run Tests

```bash
# Test on images
python tests/test_atlas_v2.py

# Test on full Egyptian league match
python -c "
from atlas.v2.processor_v2 import AtlasProcessorV2
processor = AtlasProcessorV2(verbose=True)
results = processor.process_video('path/to/egyptian_match.mp4', output_path='output.mp4')
"
```

### Step 7.3: Production Deployment

**Create API endpoint** (optional):
```python
from flask import Flask, request, jsonify
from atlas.v2.processor_v2 import AtlasProcessorV2
import base64
import cv2
import numpy as np

app = Flask(__name__)
processor = AtlasProcessorV2()

@app.route('/api/v2/detect_field', methods=['POST'])
def detect_field():
    """API endpoint for field detection"""
    # Get image from request
    image_data = request.json['image']  # Base64 encoded
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process
    result = processor.process_frame(frame)
    
    # Return JSON response
    return jsonify({
        "success": result["success"],
        "confidence": result["confidence"],
        "processing_time_ms": result["processing_time_ms"],
        "error": result["error"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## ✅ Success Criteria

**Atlas v2 is production-ready when:**

### Performance Metrics:
- [ ] Field IoU ≥ 0.90 on Egyptian validation set
- [ ] Inference time < 100ms per frame (GPU)
- [ ] 95%+ detection rate on full matches
- [ ] Temporal stability ±2px frame-to-frame

### Functionality:
- [ ] Works on all Egyptian league stadiums
- [ ] Handles day/night/shadow conditions
- [ ] Robust to broadcast/aerial/ground angles
- [ ] No per-stadium tuning required

### Integration:
- [ ] All modules integrated and tested
- [ ] API endpoints working (if applicable)
- [ ] Documentation complete
- [ ] Ready for player tracking integration (Stage 2)

---

## 📚 Next Steps

1. ✅ Complete implementation of all modules
2. → Test on Egyptian league validation set
3. → Run full match processing tests
4. → Optimize for production (TensorRT/ONNX if needed)
5. → Build player detection system (separate module)
6. → Launch commercial Egyptian League Data Center

---

**Document Version:** 2.0  
**Last Updated:** October 2024  
**Implementation Approach:** YOLOv8-seg + Modular Pipeline  
**Estimated Implementation Time:** 1-2 weeks  
**Status:** Ready for Implementation ✅
