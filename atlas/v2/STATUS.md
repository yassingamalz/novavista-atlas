# Atlas v2 Implementation Status

## Current State: Phase 1 - Sportlight Repository Cloned

### ✅ Phase 0 Complete
- Clean module structure created
- Sportlight integration wrappers with placeholder implementations
- Main pipeline with detection → calibration → mapping flow

### ✅ Phase 1 In Progress
- ✅ Sportlight repository cloned to `atlas/v2/sportlight/`
- ✅ Requirements identified (requirements.txt available)
- ⚠️ **Pre-trained models**: Not included in repo, need to be:
  - Downloaded from releases (if available)
  - OR trained from scratch (requires GPU + datasets)

### 📁 Current Structure
```
atlas/v2/
├── sportlight/               # 🆕 Cloned from GitHub
│   ├── src/
│   │   ├── models/hrnet/    # Keypoint detection model
│   │   ├── models/line/     # Line detection model
│   │   └── datatools/       # Field geometry algorithms
│   ├── requirements.txt     # Dependencies
│   └── README.md
├── detection/
│   └── sportlight_detector.py
├── calibration/
│   └── sportlight_calibrator.py
└── coordinates/
    └── mapper.py
```

### 🔍 Key Findings from Sportlight Repo

**Model Architecture:**
- Keypoints: HRNetV2-w48 backbone → 57 field points
- Lines: Separate line detection model
- Calibration: Heuristic algorithms + DLT in `src/models/hrnet/prediction.py`

**Requirements:**
- Linux + Docker (recommended) OR direct Python install
- Nvidia GPU with 24GB+ VRAM (RTX 3090/4090)
- Dependencies: PyTorch, OpenCV, SoccerNet, Hydra, etc.

**Model Files Missing:**
- Need `.pth` files for keypoints and lines models
- Check GitHub releases: https://github.com/NikolasEnt/soccernet-calibration-sportlight/releases
- OR train from scratch using SoccerNet dataset

### 📋 Next Steps

**Option A: Use Pre-trained Models (Preferred)**
1. Check GitHub releases for model weights
2. Download to `atlas/v2/sportlight/models/`
3. Update paths in our wrapper

**Option B: Train Models (If no pre-trained available)**
1. Download SoccerNet dataset
2. Use Docker container: `make build && make run`
3. Train: `python src/models/hrnet/train.py`
4. Train: `python src/models/line/train.py`

**Phase 2: Integration**
1. Adapt inference code from `src/utils/make_submit.py`
2. Replace placeholder in `detection/sportlight_detector.py`
3. Test on Egyptian League frames

### 📊 Technical Details

**Dependencies:** pytorch-argus, numpy, scipy, hydra-core, opencv-python, SoccerNet, etc.

**Inference Flow:** 
- Load HRNet model → Predict keypoints heatmaps → Extract 57 points
- Load Line model → Detect field lines  
- CameraCreator (heuristics) → DLT calibration → Homography

**Expected Performance:**
- Accuracy: 73.22% (SoccerNet benchmark)
- Completeness: 75.59%
- Processing: ~100-200ms per frame (GPU)

---
**Status:** Phase 1 partial - repository cloned, need model weights
**Date:** 2025-10-26
**Next:** Locate/download pre-trained models or set up training pipeline
