# 🛠️ NovaVista Atlas v2 - Implementation Plan

## 🎯 Overview

This document provides **step-by-step implementation instructions** with complete code examples for building Atlas v2. Follow this guide after completing the training guide and having a trained UNet model ready.

---

## 📋 Prerequisites

Before starting implementation:
- [x] Trained UNet model (`best_model.pth`)
- [x] Python 3.8+ installed
- [x] Existing Atlas v1 codebase
- [x] Basic understanding of PyTorch

---

## 🏗️ Implementation Phases

### Phase 1: Project Structure Setup
### Phase 2: Segmentation Module
### Phase 3: Enhanced Homography with Temporal Smoothing
### Phase 4: Validation Module
### Phase 5: v2 Processor Integration
### Phase 6: Testing & Validation

---

## Phase 1: Project Structure Setup

### Step 1.1: Create v2 Directory Structure

```bash
cd novavista-atlas

# Create v2 module structure
mkdir -p atlas/v2/segmentation/weights
mkdir -p atlas/v2/homography
mkdir -p atlas/v2/validation
mkdir -p atlas/v2/refinement

# Create __init__ files
touch atlas/v2/__init__.py
touch atlas/v2/segmentation/__init__.py
touch atlas/v2/homography/__init__.py
touch atlas/v2/validation/__init__.py
touch atlas/v2/refinement/__init__.py
```

### Step 1.2: Update Requirements

Add to `requirements.txt`:
```txt
# Existing requirements
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0

# New for v2
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Phase 2: Segmentation Module

### Step 2.1: UNet Architecture

Create `atlas/v2/segmentation/model_unet.py`:

```python
"""
UNet Model for Pitch Segmentation
Lightweight encoder-decoder architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet for binary segmentation (pitch detection)
    
    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes (1 for binary)
        base_channels: Base number of channels (default 64)
    """
    
    def __init__(self, n_channels=3, n_classes=1, base_channels=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        # Output
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


def create_model(pretrained_path=None, device='cuda'):
    """
    Create UNet model
    
    Args:
        pretrained_path: Path to pretrained weights
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = UNet(n_channels=3, n_classes=1, base_channels=64)
    
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded weights from {pretrained_path}")
    
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Test model
    model = UNet()
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 2.2: Inference Wrapper

Create `atlas/v2/segmentation/inference.py`:

```python
"""
Segmentation Inference Wrapper
Handles preprocessing, inference, and postprocessing
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Union, Tuple

from atlas.v2.segmentation.model_unet import create_model


class PitchSegmenter:
    """
    Wrapper for pitch segmentation inference
    """
    
    def __init__(
        self,
        model_path: str,
        input_size: int = 512,
        device: str = None,
        threshold: float = 0.5
    ):
        """
        Initialize segmenter
        
        Args:
            model_path: Path to trained model weights
            input_size: Input size for model (default 512)
            device: Device ('cuda' or 'cpu'), auto-detect if None
            threshold: Binary threshold for output (default 0.5)
        """
        self.input_size = input_size
        self.threshold = threshold
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"🔧 Initializing PitchSegmenter on {self.device}")
        
        # Load model
        self.model = create_model(model_path, self.device)
        
        # Normalization constants (ImageNet defaults)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess frame for inference
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed tensor, original size
        """
        original_size = (frame.shape[0], frame.shape[1])
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        resized = cv2.resize(frame_rgb, (self.input_size, self.input_size))
        
        # Normalize
        normalized = (resized / 255.0 - self.mean) / self.std
        
        # Convert to tensor [1, 3, H, W]
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor, original_size
    
    def postprocess(
        self,
        output: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Postprocess model output
        
        Args:
            output: Model output tensor
            original_size: Original frame size (H, W)
            
        Returns:
            Binary mask (255=pitch, 0=background)
        """
        # Apply sigmoid + threshold
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (prob > self.threshold).astype(np.uint8) * 255
        
        # Resize to original size
        mask = cv2.resize(mask, (original_size[1], original_size[0]),
                         interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment pitch from frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask (255=pitch, 0=background)
        """
        # Preprocess
        tensor, original_size = self.preprocess(frame)
        
        # Inference
        with torch.no_grad():
            output = self.model(tensor)
        
        # Postprocess
        mask = self.postprocess(output, original_size)
        
        return mask
    
    def segment_batch(self, frames: list) -> list:
        """
        Segment multiple frames (batched inference)
        
        Args:
            frames: List of BGR frames
            
        Returns:
            List of binary masks
        """
        # Preprocess all
        tensors = []
        sizes = []
        for frame in frames:
            tensor, size = self.preprocess(frame)
            tensors.append(tensor)
            sizes.append(size)
        
        # Stack into batch
        batch = torch.cat(tensors, dim=0)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(batch)
        
        # Postprocess all
        masks = []
        for output, size in zip(outputs, sizes):
            mask = self.postprocess(output.unsqueeze(0), size)
            masks.append(mask)
        
        return masks


# Convenience function
def segment_pitch(
    frame: np.ndarray,
    model_path: str = "atlas/v2/segmentation/weights/best_model.pth",
    input_size: int = 512,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Convenience function for pitch segmentation
    
    Args:
        frame: Input BGR frame
        model_path: Path to model weights
        input_size: Model input size
        threshold: Binary threshold
        
    Returns:
        Binary mask (255=pitch, 0=background)
    """
    segmenter = PitchSegmenter(model_path, input_size, threshold=threshold)
    return segmenter.segment(frame)


if __name__ == "__main__":
    # Test inference
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    segmenter = PitchSegmenter(
        model_path="weights/best_model.pth",
        device='cpu'  # Use CPU for testing
    )
    
    mask = segmenter.segment(test_frame)
    print(f"✅ Inference successful")
    print(f"   Input: {test_frame.shape}")
    print(f"   Output: {mask.shape}")
    print(f"   Pitch pixels: {np.sum(mask == 255)}")
```

### Step 2.3: Mask Refinement

Create `atlas/v2/refinement/mask_refiner.py`:

```python
"""
Mask Refinement Module
Post-process deep learning masks using morphological operations
"""

import cv2
import numpy as np
from typing import Tuple


class MaskRefiner:
    """
    Refine segmentation masks using morphological operations
    """
    
    def __init__(
        self,
        morph_kernel: int = 5,
        use_convex_hull: bool = False
    ):
        """
        Initialize refiner
        
        Args:
            morph_kernel: Kernel size for morphological operations
            use_convex_hull: Apply convex hull for smooth boundaries
        """
        self.morph_kernel = morph_kernel
        self.use_convex_hull = use_convex_hull
    
    def refine(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine binary mask
        
        Args:
            mask: Binary mask (255=pitch, 0=background)
            
        Returns:
            Refined binary mask
        """
        # Create morphology kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel, self.morph_kernel)
        )
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Keep largest connected component
        mask = self._keep_largest_component(mask)
        
        # Optional: Convex hull for smooth boundary
        if self.use_convex_hull:
            mask = self._apply_convex_hull(mask)
        
        return mask
    
    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component"""
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return mask
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Create new mask with only largest component
        refined = np.zeros_like(mask)
        cv2.drawContours(refined, [largest], -1, 255, -1)
        
        return refined
    
    def _apply_convex_hull(self, mask: np.ndarray) -> np.ndarray:
        """Apply convex hull to mask for smooth boundary"""
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return mask
        
        # Get convex hull of largest contour
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        
        # Create new mask
        refined = np.zeros_like(mask)
        cv2.drawContours(refined, [hull], -1, 255, -1)
        
        return refined
    
    def calculate_quality_metrics(self, mask: np.ndarray) -> dict:
        """
        Calculate quality metrics for mask
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary of metrics
        """
        total_pixels = mask.size
        pitch_pixels = np.sum(mask == 255)
        pitch_percentage = (pitch_pixels / total_pixels) * 100
        
        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        largest_area = 0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest)
        
        return {
            "pitch_percentage": pitch_percentage,
            "pitch_pixels": pitch_pixels,
            "total_pixels": total_pixels,
            "num_components": len(contours),
            "largest_component_area": largest_area
        }


# Convenience function
def refine_mask(
    mask: np.ndarray,
    morph_kernel: int = 5,
    use_convex_hull: bool = False
) -> np.ndarray:
    """
    Convenience function for mask refinement
    
    Args:
        mask: Binary mask
        morph_kernel: Kernel size
        use_convex_hull: Apply convex hull
        
    Returns:
        Refined mask
    """
    refiner = MaskRefiner(morph_kernel, use_convex_hull)
    return refiner.refine(mask)
```

---

## Phase 3: Enhanced Homography with Temporal Smoothing

### Step 3.1: Temporal Smoother

Create `atlas/v2/homography/smoother.py`:

```python
"""
Temporal Homography Smoother
Stabilize homography estimates across frames using EMA/Kalman filtering
"""

import numpy as np
from typing import Optional, Tuple


class HomographySmoother:
    """
    Smooth homography matrices across time using exponential moving average
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize smoother
        
        Args:
            alpha: Smoothing factor [0,1]
                   0 = only use history (very smooth, slow adaptation)
                   1 = no smoothing (use current only)
                   0.3 = good balance
        """
        self.alpha = alpha
        self.H_prev = None
        self.frame_count = 0
    
    def smooth(
        self,
        H_current: Optional[np.ndarray],
        quality_score: float
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Smooth homography estimate
        
        Args:
            H_current: Current homography (3×3) or None if failed
            quality_score: Quality score [0,1] of current estimate
            
        Returns:
            Smoothed homography, state dict
        """
        self.frame_count += 1
        
        # First frame or no previous H
        if self.H_prev is None:
            if H_current is not None and quality_score > 0.5:
                self.H_prev = H_current.copy()
                return H_current, {"smoothed": False, "fallback": False}
            else:
                return None, {"smoothed": False, "fallback": True}
        
        # Current estimation failed or low quality
        if H_current is None or quality_score < 0.4:
            # Use previous homography (fallback)
            return self.H_prev, {"smoothed": True, "fallback": True}
        
        # Smooth using EMA
        H_smooth = self.alpha * H_current + (1 - self.alpha) * self.H_prev
        
        # Normalize (last element should be 1)
        H_smooth = H_smooth / H_smooth[2, 2]
        
        # Update history
        self.H_prev = H_smooth
        
        return H_smooth, {"smoothed": True, "fallback": False}
    
    def reset(self):
        """Reset smoother state"""
        self.H_prev = None
        self.frame_count = 0


class KalmanHomographySmoother:
    """
    More advanced smoother using Kalman filter (optional upgrade)
    """
    
    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1
    ):
        """
        Initialize Kalman filter for homography parameters
        
        Args:
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        # Homography has 8 DOF (9 elements, normalized by H[2,2]=1)
        self.state_dim = 8
        
        # State: [h11, h12, h13, h21, h22, h23, h31, h32]
        self.x = None  # State vector
        self.P = None  # Error covariance
        
        # Process and measurement noise
        self.Q = np.eye(self.state_dim) * process_noise
        self.R = np.eye(self.state_dim) * measurement_noise
        
        self.initialized = False
    
    def _H_to_vector(self, H: np.ndarray) -> np.ndarray:
        """Convert 3×3 homography to 8D vector"""
        H_norm = H / H[2, 2]
        return H_norm.flatten()[:8]
    
    def _vector_to_H(self, vec: np.ndarray) -> np.ndarray:
        """Convert 8D vector to 3×3 homography"""
        H = np.zeros((3, 3))
        H.flat[:8] = vec
        H[2, 2] = 1.0
        return H
    
    def smooth(
        self,
        H_current: Optional[np.ndarray],
        quality_score: float
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Smooth using Kalman filter
        
        Args:
            H_current: Current homography
            quality_score: Quality score
            
        Returns:
            Smoothed homography, state dict
        """
        if H_current is None or quality_score < 0.3:
            if self.initialized:
                return self._vector_to_H(self.x), {"smoothed": True, "fallback": True}
            else:
                return None, {"smoothed": False, "fallback": True}
        
        z = self._H_to_vector(H_current)  # Measurement
        
        if not self.initialized:
            # Initialize state
            self.x = z
            self.P = np.eye(self.state_dim) * 0.1
            self.initialized = True
            return H_current, {"smoothed": False, "fallback": False}
        
        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update step
        y = z - x_pred  # Innovation
        S = P_pred + self.R  # Innovation covariance
        K = P_pred @ np.linalg.inv(S)  # Kalman gain
        
        self.x = x_pred + K @ y
        self.P = (np.eye(self.state_dim) - K) @ P_pred
        
        H_smooth = self._vector_to_H(self.x)
        return H_smooth, {"smoothed": True, "fallback": False}
    
    def reset(self):
        """Reset filter"""
        self.x = None
        self.P = None
        self.initialized = False
```

Continue in next message...
