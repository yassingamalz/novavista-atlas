# 🎓 NovaVista Atlas v2 - YOLOv8-seg Training Guide

## 🎯 Objective

Train a production-ready **YOLOv8-seg model** for Egyptian Premier League field detection using:
- ✅ SoccerNet dataset (base training) - FREE
- ✅ Egyptian league data (fine-tuning) - 50-100 frames
- ✅ Transfer learning (best of both worlds)
- ✅ Commercial-grade results (90-95% IoU)

---

## 📋 Prerequisites

### Required Accounts (All Free)
1. **Google Account** - for Google Drive & Colab (or local GPU)
2. **GitHub Account** - for code repository
3. **Roboflow Account** - for annotation (free tier sufficient)

### Hardware Options
**Option A: Local GPU (Recommended if available)**
- NVIDIA GPU with ≥8GB VRAM (RTX 3060+, RTX 4060+, or better)
- CUDA 11.8+ installed
- Faster iteration, no session limits

**Option B: Google Colab**
- Free tier: Tesla T4 (~15GB VRAM)
- Pro tier: A100/V100 ($10/month)
- Session limits, but sufficient for this project

**Option C: Cloud GPU (AWS/Lambda Labs)**
- On-demand rental (~$0.5-1/hour)
- Good for final training runs

### Software Requirements
```bash
# Python packages (install via pip)
pip install ultralytics opencv-python pillow numpy matplotlib
pip install roboflow  # For dataset management (optional)
```

---

## 🎬 Phase 1: SoccerNet Base Dataset

### Step 1.1: Download SoccerNet Dataset

**SoccerNet** provides free, pre-annotated soccer field datasets for research and commercial use.

**Download Options:**

**Option A: Pre-processed YOLO format (RECOMMENDED)**
```bash
# Use roboflow's public SoccerNet dataset (YOLO format ready)
# Visit: https://universe.roboflow.com/soccernet/soccer-field-segmentation
# Click "Download" → Format: YOLOv8 → Get download link

# Or use their API:
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("soccernet").project("soccer-field-segmentation")
dataset = project.version(1).download("yolov8")
```

**Option B: Manual download from SoccerNet**
```bash
# Install SoccerNet pip package
pip install SoccerNet

# Download field segmentation dataset
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/soccernet")
mySoccerNetDownloader.downloadDataTask(task="segmentation", split=["train", "valid"])

# Convert to YOLO format (script provided below)
```

### Step 1.2: SoccerNet Dataset Structure

After download, you should have:
```
soccernet_dataset/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt    # YOLO segmentation format
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
├── data.yaml              # Dataset configuration
└── README.md
```

**data.yaml** content:
```yaml
path: ./soccernet_dataset
train: train/images
val: valid/images

names:
  0: soccer_field

# Optional metadata
nc: 1  # number of classes
```

### Step 1.3: Verify SoccerNet Data

**Quick verification script:**
```python
import os
from pathlib import Path
import cv2
import numpy as np

def verify_dataset(dataset_path):
    """Check dataset integrity"""
    train_images = list(Path(dataset_path, "train/images").glob("*.jpg"))
    train_labels = list(Path(dataset_path, "train/labels").glob("*.txt"))
    
    print(f"✅ Training images: {len(train_images)}")
    print(f"✅ Training labels: {len(train_labels)}")
    print(f"✅ Match: {len(train_images) == len(train_labels)}")
    
    # Check a random label file
    if train_labels:
        with open(train_labels[0]) as f:
            lines = f.readlines()
            print(f"\n✅ Sample label (polygon points): {len(lines[0].split())} coords")
            print(f"   First line: {lines[0][:100]}...")

verify_dataset("soccernet_dataset/")
```

---

## 🎬 Phase 2: Egyptian League Data Collection

### Step 2.1: Extract Video Frames

**Install ffmpeg** (if not installed):
```bash
# Windows (via chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Extract diverse frames:**
```bash
# Strategy: Extract 10 frames per match from different game moments
# This gives variety without too much redundancy

# Match 1: Cairo Stadium (day game)
ffmpeg -i cairo_match.mp4 -vf "select='not(mod(n\,300))',scale=1280:720" -vsync 0 egyptian_data/raw/cairo_%04d.jpg

# Match 2: Alexandria Stadium (night game)
ffmpeg -i alex_match.mp4 -vf "select='not(mod(n\,300))',scale=1280:720" -vsync 0 egyptian_data/raw/alex_%04d.jpg

# Match 3: Different broadcaster angle
ffmpeg -i broadcast2_match.mp4 -vf "select='not(mod(n\,300))',scale=1280:720" -vsync 0 egyptian_data/raw/broad2_%04d.jpg
```

### Step 2.2: Frame Selection Guidelines

**Target: 50-100 high-quality frames**

**Diversity checklist:**
- [ ] 3-5 different stadiums
  - Cairo International Stadium
  - Alexandria Stadium  
  - Suez Stadium
  - Port Said Stadium
  - Others
- [ ] Day games (5-10 frames per stadium)
- [ ] Night games (5-10 frames per stadium)
- [ ] Different camera angles:
  - Broadcast main (center field view)
  - Corner camera
  - High angle (tactical view)
- [ ] Different field conditions:
  - Well-maintained grass
  - Worn patches
  - Shadows across field
- [ ] Different game moments (avoid clustering similar frames)

**Quality requirements:**
- ≥40% of field visible
- Sharp focus (not motion-blurred)
- Representative of actual match footage

### Step 2.3: Legal Considerations

**✅ Legally safe approach:**
1. **Training data:** Use SoccerNet (open license) + small Egyptian sample
2. **Egyptian footage:** For R&D and model improvement (fair use)
3. **Product data:** Generate from matches you have rights to process
4. **Before commercial launch:** Verify broadcast rights or self-record training fields

**Practical recommendation:**
- Use Egyptian broadcast footage for initial training (50-100 frames is minimal use)
- Before selling data product, confirm rights with Egyptian FA or broadcasters
- The trained model weights are YOUR IP
- Data generated by the model is yours to sell

---

## 🎨 Phase 3: Egyptian Data Annotation

### Step 3.1: Setup Roboflow (RECOMMENDED)

**Why Roboflow:**
- Free tier includes 1000 images
- YOLO export built-in
- AI-assisted annotation (after 20 frames)
- Team collaboration
- Auto-generates train/val split

**Steps:**
1. Go to [Roboflow](https://roboflow.com/)
2. Create account (free)
3. Create new project:
   - Project Name: "Egyptian League Fields"
   - Project Type: "Instance Segmentation"
   - Annotation Group: "soccer_field"

### Step 3.2: Upload & Annotate

**Upload frames:**
1. Click "Upload" → Select your 50-100 Egyptian league frames
2. Roboflow will auto-detect duplicates

**Annotation process:**
1. Select "Smart Polygon" tool
2. Click around the field perimeter (8-12 points usually enough)
3. Double-click to close polygon
4. Label as "soccer_field"
5. Next image

**Annotation guidelines:**
- **Include:** Playable grass area + field markings
- **Exclude:** Stands, advertising, warm-up areas, sidelines
- **Shadows:** Include grass under shadow
- **Precision:** Rough polygons are fine (model will learn boundaries)

**Speed tips:**
- Zoom in for edges
- Use keyboard shortcuts (V for vertex mode)
- After 20-30 frames, Roboflow AI can assist
- Consistent rules across all frames

**Time estimate:** 15-30 min per frame = 12-50 hours total for 50-100 frames

### Step 3.3: Quality Check & Export

**Pre-export checklist:**
- [ ] All frames annotated
- [ ] Consistent labeling (spot-check 5 random frames)
- [ ] No missing annotations
- [ ] Polygons cover full field area

**Export from Roboflow:**
1. Click "Generate" → "Create New Version"
2. Preprocessing: None (or minimal resize to 640x640)
3. Augmentation: None (YOLO training will handle this)
4. Format: **YOLOv8**
5. Download → Save as `egyptian_league_dataset.zip`

**Extract locally:**
```bash
unzip egyptian_league_dataset.zip -d data/egyptian_league/
```

**Expected structure:**
```
data/
├── soccernet_dataset/           # Base training (1000+ frames)
│   ├── train/
│   ├── valid/
│   └── data.yaml
└── egyptian_league_dataset/     # Fine-tuning (50-100 frames)
    ├── train/                   # 80% of your frames
    ├── valid/                   # 20% of your frames
    └── data.yaml
```

---

## 🧠 Phase 4: Base Model Training (SoccerNet)

### Step 4.1: Install YOLOv8

```bash
pip install ultralytics
```

### Step 4.2: Train on SoccerNet Dataset

**Training script** (`train_base_soccernet.py`):
```python
from ultralytics import YOLO
import torch

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Load pre-trained YOLOv8-seg model
model = YOLO('yolov8s-seg.pt')  # Small variant (balanced speed/accuracy)

# Train on SoccerNet
results = model.train(
    data='data/soccernet_dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,          # Adjust based on GPU memory
    patience=10,       # Early stopping
    save=True,
    device=0,          # GPU 0 (use 'cpu' if no GPU)
    project='runs/soccernet',
    name='base_model',
    
    # Optimization
    optimizer='AdamW',
    lr0=0.01,
    weight_decay=0.0005,
    
    # Augmentation (built-in)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
)

# Save best model
print(f"✅ Training complete!")
print(f"   Best model: runs/soccernet/base_model/weights/best.pt")
print(f"   mAP: {results.results_dict['metrics/mAP50-95(M)']:.3f}")
```

**Run training:**
```bash
python train_base_soccernet.py
```

### Step 4.3: Expected Results

**Training time:**
| GPU | Dataset Size | Time (50 epochs) |
|-----|--------------|------------------|
| RTX 3060 (12GB) | 1000 images | ~2-3 hours |
| RTX 4090 | 1000 images | ~45 min |
| Tesla T4 (Colab) | 1000 images | ~3-4 hours |

**Target metrics after SoccerNet training:**
- mAP50: >0.92
- mAP50-95: >0.85
- Inference: ~30ms per frame

### Step 4.4: Validate Base Model

```python
from ultralytics import YOLO
import cv2
import numpy as np

# Load best model
model = YOLO('runs/soccernet/base_model/weights/best.pt')

# Validate on SoccerNet validation set
metrics = model.val(data='data/soccernet_dataset/data.yaml')
print(f"Validation mAP50: {metrics.box.map50:.3f}")
print(f"Validation mAP50-95: {metrics.box.map:.3f}")

# Test on a sample Egyptian frame (before fine-tuning)
egyptian_test = cv2.imread('data/egyptian_league/test_frame.jpg')
results = model(egyptian_test)
print(f"Confidence on Egyptian frame: {results[0].boxes.conf[0]:.3f}")

# If confidence < 0.7, fine-tuning is needed ✅
```

---

## 🎯 Phase 5: Fine-Tuning on Egyptian League Data

### Step 5.1: Fine-Tuning Strategy

**Why fine-tune:**
- SoccerNet has general soccer knowledge
- Egyptian stadiums have unique characteristics (specific camera angles, lighting, grass quality)
- Fine-tuning adapts the model to your target domain

**Fine-tuning script** (`train_egyptian_finetune.py`):
```python
from ultralytics import YOLO

# Load SoccerNet-trained model
model = YOLO('runs/soccernet/base_model/weights/best.pt')

# Fine-tune on Egyptian data
results = model.train(
    data='data/egyptian_league_dataset/data.yaml',
    epochs=30,          # Fewer epochs for fine-tuning
    imgsz=640,
    batch=8,            # Smaller batch due to small dataset
    patience=10,
    save=True,
    device=0,
    project='runs/egyptian',
    name='final_model',
    
    # Lower learning rate for fine-tuning
    optimizer='AdamW',
    lr0=0.001,          # 10x lower than base training
    weight_decay=0.0005,
    
    # Augmentation (aggressive for small dataset)
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.5,
    degrees=15,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)

print(f"✅ Fine-tuning complete!")
print(f"   Production model: runs/egyptian/final_model/weights/best.pt")
```

**Run fine-tuning:**
```bash
python train_egyptian_finetune.py
```

### Step 5.2: Fine-Tuning Results

**Expected improvements:**
| Metric | After SoccerNet | After Egyptian Fine-tune | Target |
|--------|----------------|--------------------------|--------|
| mAP50 (Egyptian val) | 0.85-0.88 | **0.92-0.95** | >0.90 |
| Confidence (Egyptian) | 0.65-0.75 | **0.88-0.95** | >0.85 |
| Broadcast angle IoU | 0.82-0.86 | **0.90-0.93** | >0.88 |

**Training time:** 30-60 minutes (50-100 images, 30 epochs)

---

## 📊 Phase 6: Evaluation & Validation

### Step 6.1: Comprehensive Testing

**Test on multiple scenarios:**
```python
from ultralytics import YOLO
import glob

model = YOLO('runs/egyptian/final_model/weights/best.pt')

test_scenarios = {
    'cairo_day': glob.glob('test_data/cairo_stadium_day/*.jpg'),
    'alex_night': glob.glob('test_data/alexandria_night/*.jpg'),
    'broadcast_angle': glob.glob('test_data/broadcast/*.jpg'),
    'aerial_view': glob.glob('test_data/aerial/*.jpg'),
}

for scenario, images in test_scenarios.items():
    ious = []
    confidences = []
    
    for img_path in images:
        results = model(img_path)
        
        if len(results[0].boxes) > 0:
            confidences.append(results[0].boxes.conf[0].item())
            # Calculate IoU if ground truth available
            # ious.append(calculate_iou(results[0].masks, ground_truth))
    
    print(f"{scenario}:")
    print(f"  Avg confidence: {np.mean(confidences):.3f}")
    print(f"  Min confidence: {np.min(confidences):.3f}")
    # print(f"  Avg IoU: {np.mean(ious):.3f}")
```

### Step 6.2: Visual Quality Check

**Generate comparison grid:**
```python
import matplotlib.pyplot as plt

def visualize_results(model, test_images, save_path='validation_grid.png'):
    """Create grid of predictions"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, img_path in enumerate(test_images[:9]):
        img = cv2.imread(img_path)
        results = model(img)
        
        # Overlay mask
        if len(results[0].masks) > 0:
            mask = results[0].masks.data[0].cpu().numpy()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            overlay = img_rgb.copy()
            overlay[mask > 0.5] = [0, 255, 0]  # Green overlay
            result_img = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
        else:
            result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(result_img)
        conf = results[0].boxes.conf[0].item() if len(results[0].boxes) > 0 else 0
        axes[i].set_title(f"Conf: {conf:.3f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ Validation grid saved: {save_path}")

# Run visualization
test_images = glob.glob('data/egyptian_league_dataset/valid/images/*.jpg')
visualize_results(model, test_images)
```

### Step 6.3: Performance Benchmarks

**Inference speed test:**
```python
import time

model = YOLO('runs/egyptian/final_model/weights/best.pt')
test_img = cv2.imread('test_frame.jpg')

# Warmup
for _ in range(10):
    _ = model(test_img)

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    results = model(test_img)
    times.append((time.time() - start) * 1000)  # ms

print(f"Inference time:")
print(f"  Mean: {np.mean(times):.1f}ms")
print(f"  Std: {np.std(times):.1f}ms")
print(f"  FPS: {1000/np.mean(times):.1f}")
```

**Target:** <50ms per frame (GPU), <150ms (CPU)

---

## 📦 Phase 7: Model Export & Optimization

### Step 7.1: Export Formats

**PyTorch (.pt) - Default**
```python
# Already saved during training
model_path = 'runs/egyptian/final_model/weights/best.pt'
```

**ONNX - For faster inference**
```python
model = YOLO('runs/egyptian/final_model/weights/best.pt')
model.export(format='onnx', imgsz=640, dynamic=True)

# Use ONNX model
model_onnx = YOLO('runs/egyptian/final_model/weights/best.onnx')
results = model_onnx('test_frame.jpg')
```

**TensorRT - For NVIDIA GPU (fastest)**
```python
model.export(format='engine', imgsz=640, device=0)

# Use TensorRT engine
model_trt = YOLO('runs/egyptian/final_model/weights/best.engine')
```

### Step 7.2: Model Deployment Package

**Create production package:**
```bash
mkdir -p atlas/v2/segmentation/weights/

# Copy final model
cp runs/egyptian/final_model/weights/best.pt atlas/v2/segmentation/weights/final_egyptian.pt

# Optional: Also copy ONNX for CPU deployment
cp runs/egyptian/final_model/weights/best.onnx atlas/v2/segmentation/weights/final_egyptian.onnx

# Save model metadata
cat > atlas/v2/segmentation/weights/model_info.yaml << EOF
model_name: Egyptian Premier League Field Detector
version: 1.0
architecture: YOLOv8s-seg
training:
  base_dataset: SoccerNet (1000 frames)
  fine_tune_dataset: Egyptian League (83 frames)
  epochs: 50 (base) + 30 (fine-tune)
  date_trained: $(date +%Y-%m-%d)
performance:
  map50_egyptian: 0.94
  inference_time_gpu: 32ms
  inference_time_cpu: 145ms
license: Your Company - Commercial Use
EOF
```

---

## 🎯 Success Criteria Checklist

Before deploying to production:

**Training Metrics:**
- [ ] SoccerNet base mAP50 > 0.92
- [ ] Egyptian fine-tune mAP50 > 0.90
- [ ] Confidence on Egyptian validation set > 0.85

**Visual Quality:**
- [ ] Masks cover full field area accurately
- [ ] Clean boundaries (no jagged edges)
- [ ] Works on aerial, broadcast, and ground-level views
- [ ] Handles shadows and lighting variations

**Performance:**
- [ ] Inference < 50ms (GPU) or < 150ms (CPU)
- [ ] Model size < 50MB (.pt format)
- [ ] Memory usage < 2GB during inference

**Coverage:**
- [ ] Tested on all Egyptian league stadiums in dataset
- [ ] Tested on day and night games
- [ ] Tested on different broadcasters' camera angles
- [ ] 95%+ detection rate across all test scenarios

---

## 🆘 Troubleshooting

### Low mAP after SoccerNet training (<0.90)
**Solutions:**
- Train for more epochs (try 100)
- Use larger model (yolov8m-seg or yolov8l-seg)
- Check data.yaml paths are correct
- Verify annotations are in correct YOLO format

### Fine-tuning doesn't improve Egyptian performance
**Possible causes:**
- Learning rate too high (try 0.0005)
- Not enough epochs (try 50 instead of 30)
- Egyptian data too similar to SoccerNet (model already knows it)
- Need more diverse Egyptian frames (add more stadiums)

### Model overfits Egyptian data (train mAP >> val mAP)
**Solutions:**
- Increase augmentation strength
- Add more Egyptian validation frames
- Use early stopping (patience=5)
- Freeze backbone layers (freeze=10)

### Slow inference (>100ms GPU)
**Solutions:**
- Export to TensorRT format
- Use smaller model (yolov8n-seg instead of yolov8s-seg)
- Reduce input size (imgsz=480 instead of 640)
- Batch multiple frames together

### CUDA out of memory during training
**Solutions:**
- Reduce batch size (batch=8 or batch=4)
- Use smaller model (yolov8n-seg)
- Clear GPU cache: `torch.cuda.empty_cache()`
- Close other GPU applications

---

## 📚 Resources

**YOLOv8 Documentation:**
- [Ultralytics Docs](https://docs.ultralytics.com/)
- [Segmentation Tutorial](https://docs.ultralytics.com/tasks/segment/)
- [Training Guide](https://docs.ultralytics.com/modes/train/)

**Datasets:**
- [SoccerNet](https://www.soccer-net.org/)
- [Roboflow Universe](https://universe.roboflow.com/)

**Tools:**
- [Roboflow](https://roboflow.com/) - Annotation & dataset management
- [Ultralytics HUB](https://hub.ultralytics.com/) - Cloud training (optional)

---

## ✅ Next Steps

Once training is complete:
1. ✅ Export final model to `atlas/v2/segmentation/weights/`
2. → Proceed to `IMPLEMENTATION_PLAN.md`
3. → Integrate YOLOv8 into Atlas pipeline
4. → Test on full Egyptian league matches
5. → Build player detection system (Stage 2)
6. → Launch commercial data center

---

**Estimated Timeline:**
- Week 1-2: SoccerNet base training
- Week 3: Egyptian data collection & annotation
- Week 4: Fine-tuning & validation
- **Total: 3-4 weeks to production-ready model**

**Cost Estimate:**
- Hardware: $0 (using existing GPU or Colab free tier)
- Roboflow: $0 (free tier)
- Datasets: $0 (SoccerNet free)
- Annotation labor: 25-50 hours (DIY) or $200-500 (outsource)

**Document Version:** 2.0  
**Last Updated:** October 2024  
**Training Approach:** YOLOv8-seg + Transfer Learning  
**Production-Ready:** Yes ✅
