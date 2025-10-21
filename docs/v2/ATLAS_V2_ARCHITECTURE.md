# 🏗️ NovaVista Atlas v2 - Architecture & Design

## 📌 Executive Summary

**NovaVista Atlas v2** represents a fundamental architectural shift from classical computer vision to a **hybrid deep learning + geometry approach**. This document outlines the technical architecture, design decisions, and implementation strategy.

---

## 🔄 Evolution: v1 → v2

### v1 Architecture (Classical CV)
```
[Input Frame]
    ↓
[HSV Color Thresholding] ← ❌ Brittle, lighting-dependent
    ↓
[Hough Line Detection]
    ↓
[ORB Feature Matching]
    ↓
[Homography Estimation]
    ↓
[Output]
```

**Problems:**
- HSV thresholds break with lighting changes
- Shadows fragment the field mask
- Different grass colors (dry vs wet) cause failures
- Advertising boards confuse line detection
- ~60-70% reliability in production

### v2 Architecture (Hybrid DL + Geometry)
```
[Input Frame]
    ↓
[Deep Learning Segmentation (UNet)] ← ✅ Robust, learned features
    ↓
[Mask Refinement (OpenCV)]
    ↓
[Line Detection (Hough on masked region)]
    ↓
[Keypoint Extraction (Line intersections)]
    ↓
[Homography Estimation (RANSAC)]
    ↓
[Temporal Smoothing (EMA/Kalman)]
    ↓
[Validated Output]
```

**Improvements:**
- Segmentation: ~90%+ IoU vs 60-70% in v1
- Lighting robustness: Works day/night/shadows
- Stadium agnostic: Adapts to any grass color
- Temporal stability: Smooth frame-to-frame
- ~95%+ production reliability target

---

## 🧩 System Components

### 1. **Segmentation Module** (NEW in v2)

**Purpose:** Isolate pitch area using learned visual features

**Implementation:**
- **Model:** UNet-Lite (lightweight encoder-decoder)
- **Input:** 512×512 RGB frame (resized)
- **Output:** Binary mask (pitch=1, background=0)
- **Training:** ~500-1000 annotated frames from Egyptian league
- **Inference:** ~15ms on GPU, ~80ms on CPU

**File Location:**
```
atlas/v2/segmentation/
├── model_unet.py       # UNet architecture
├── dataset.py          # Data loader
├── train.py            # Training script
├── inference.py        # Inference wrapper
└── weights/            # Trained model weights
```

**Key Methods:**
```python
class PitchSegmenter:
    def __init__(self, model_path: str):
        """Load trained UNet model"""
        
    def segment(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: RGB image (any size)
        Returns:
            Binary mask (same size as input)
        """
```

---

### 2. **Mask Refinement Module** (ENHANCED in v2)

**Purpose:** Clean up DL predictions using morphology

**Changes from v1:**
- Now operates on DL mask instead of HSV mask
- More aggressive noise removal
- Convex hull approximation for smooth boundaries

**Implementation:**
```python
def refine_mask(dl_mask: np.ndarray) -> np.ndarray:
    """
    Post-process deep learning mask
    
    Steps:
    1. Morphological opening (remove small noise)
    2. Morphological closing (fill small holes)
    3. Keep largest connected component
    4. Optional: Convex hull for smooth boundary
    """
```

---

### 3. **Line Detection Module** (UNCHANGED from v1)

**Purpose:** Detect pitch markings within segmented area

**Key Difference:** Now operates on **masked DL output** instead of HSV output
- Same Hough transform parameters
- Same line filtering logic
- Better results due to cleaner input mask

---

### 4. **Homography Estimation** (ENHANCED in v2)

**New Features:**
- **Temporal smoothing:** EMA filter on homography matrix
- **Quality validation:** Reject bad homographies automatically
- **Fallback mechanism:** Use previous good homography if current fails

**Implementation:**
```python
class HomographyEstimatorV2:
    def __init__(self, alpha=0.3):
        """
        alpha: EMA smoothing factor
        """
        self.H_prev = None
        
    def estimate_smooth(self, 
                       keypoints_img: np.ndarray,
                       keypoints_template: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Estimate homography with temporal smoothing
        
        Returns:
            H: 3×3 matrix
            quality_score: float [0,1]
        """
```

---

### 5. **Validation Module** (NEW in v2)

**Purpose:** Assess homography quality and trigger fallback

**Checks:**
1. Inlier ratio > 0.6 (from RANSAC)
2. All 4 corners project inside image bounds
3. Mask overlap between projected template and segmentation > 0.7
4. Homography determinant > 0 (non-degenerate)

**Output:** Quality score ∈ [0, 1]

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT FRAME                             │
│                     (1920×1080 RGB)                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  1. SEGMENTATION (UNet)                         │
│  • Resize to 512×512                                            │
│  • Normalize RGB                                                │
│  • Forward pass through model                                   │
│  • Upsample mask to original size                               │
│  Output: Binary mask (H×W)                                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  2. MASK REFINEMENT                             │
│  • Morphological open/close                                     │
│  • Largest connected component                                  │
│  • Optional convex hull                                         │
│  Output: Refined binary mask                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  3. LINE DETECTION                              │
│  • Apply mask to frame                                          │
│  • Canny edge detection                                         │
│  • Hough line transform                                         │
│  • Filter by length/angle                                       │
│  Output: List of line segments                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  4. KEYPOINT EXTRACTION                         │
│  • Compute line intersections                                   │
│  • Filter inside mask                                           │
│  • Select 4 strongest corners                                   │
│  Output: Image keypoints (N×2)                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  5. HOMOGRAPHY ESTIMATION                       │
│  • Match to template keypoints                                  │
│  • RANSAC with cv2.findHomography                               │
│  • Temporal smoothing (EMA)                                     │
│  Output: H (3×3), quality_score                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  6. VALIDATION                                  │
│  • Check inlier ratio                                           │
│  • Check corner bounds                                          │
│  • Check mask overlap                                           │
│  • Decide: accept or fallback                                   │
│  Output: Final H (3×3), confidence                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       JSON OUTPUT                               │
│  {                                                              │
│    "mask": ...,                                                 │
│    "keypoints": {...},                                          │
│    "homography": [[...], [...], [...]],                         │
│    "quality_score": 0.93,                                       │
│    "confidence": {...}                                          │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ File Structure for v2

```
atlas/
├── v2/                          # NEW: v2 modules
│   ├── __init__.py
│   ├── processor_v2.py          # Main v2 processor
│   │
│   ├── segmentation/            # NEW: DL segmentation
│   │   ├── __init__.py
│   │   ├── model_unet.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── weights/
│   │       └── best_model.pth
│   │
│   ├── refinement/              # ENHANCED: Mask refinement
│   │   ├── __init__.py
│   │   └── mask_refiner.py
│   │
│   ├── homography/              # ENHANCED: Temporal smoothing
│   │   ├── __init__.py
│   │   ├── estimator_v2.py
│   │   └── smoother.py
│   │
│   └── validation/              # NEW: Quality checks
│       ├── __init__.py
│       └── validator.py
│
├── detection/                   # REUSED from v1
│   ├── line_detector.py
│   ├── circle_detector.py
│   └── corner_detector.py
│
├── calibration/                 # REUSED from v1
│   └── ransac.py
│
└── coordinates/                 # REUSED from v1
    └── transformer.py
```

---

## 🎯 Performance Targets

| Metric | v1 (Current) | v2 (Target) | How to Measure |
|--------|--------------|-------------|----------------|
| Pitch IoU | 0.60-0.70 | ≥0.90 | Compare mask vs ground truth |
| Detection Rate | ~70% | ≥95% | Successful homography frames |
| Processing Time | 800ms | <1000ms | GPU inference |
| Temporal Stability | Poor | ±2px | StdDev of corner positions |
| Lighting Robustness | Fails often | Works always | Day/night test suite |

---

## 🧪 Validation Strategy

### 1. **Unit Tests**
- Segmentation: IoU on test set
- Line detection: Precision/recall on annotated lines
- Homography: Reprojection error < 5px

### 2. **Integration Tests**
- Full pipeline on 100-frame test video
- Measure frame-to-frame stability
- Visual inspection of overlays

### 3. **Production Tests**
- Run on full Egyptian league matches (multiple stadiums)
- Compare v1 vs v2 detection rates
- User acceptance testing

---

## 🚀 Migration Path

**Phase 1: Parallel Development** (Week 1-2)
- Keep v1 running
- Build v2 in separate directory
- Train initial segmentation model

**Phase 2: Alpha Testing** (Week 3)
- Run v2 on test videos
- Compare outputs side-by-side
- Fix major issues

**Phase 3: Beta Deployment** (Week 4)
- Deploy v2 to staging environment
- Run both v1 and v2, log differences
- Tune parameters

**Phase 4: Production Cutover** (Week 5)
- Switch production traffic to v2
- Keep v1 as fallback
- Monitor for 1 week

**Phase 5: v1 Deprecation** (Week 6)
- Remove v1 code
- Archive v1 documentation
- Full v2 production

---

## 🔐 Legal & Licensing

**Training Data:**
- Egyptian league footage: Verify broadcast rights
- Self-recorded training field footage: ✅ 100% yours
- Annotated masks: ✅ Your IP

**Model Weights:**
- Trained on your data: ✅ You own them
- Can use commercially: ✅ Yes
- Can sell/license: ✅ Yes

**Code:**
- UNet architecture: Public domain (you implement)
- Training pipeline: ✅ Your code
- OpenCV usage: ✅ BSD license (commercial-friendly)

---

## 📚 References

- **UNet Paper:** [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- **Semantic Segmentation:** [Long et al., 2015](https://arxiv.org/abs/1411.4038)
- **Homography Estimation:** [Hartley & Zisserman, 2004]
- **Temporal Filtering:** Kalman Filter tutorial

---

## ✅ Deliverables Checklist

- [ ] Trained UNet model (best_model.pth)
- [ ] v2 processor implementation (processor_v2.py)
- [ ] Segmentation inference wrapper (inference.py)
- [ ] Homography temporal smoother (smoother.py)
- [ ] Validation module (validator.py)
- [ ] Training dataset (500+ annotated frames)
- [ ] Unit tests for all v2 modules
- [ ] Integration test suite
- [ ] Performance benchmark report
- [ ] Migration guide for users

---

**Document Version:** 1.0  
**Last Updated:** October 2024  
**Author:** NovaVista Team  
**Status:** Architecture Finalized, Implementation In Progress
