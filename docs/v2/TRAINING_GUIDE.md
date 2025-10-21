# 🎓 NovaVista Atlas v2 - Training Guide

## 🎯 Objective

Train your own **UNet segmentation model** to detect football pitches in Egyptian league footage, using:
- ✅ Free annotation tools
- ✅ Google Colab (free GPU)
- ✅ Your own data (legally safe for commercial use)

---

## 📋 Prerequisites

### Required Accounts (All Free)
1. **Google Account** - for Google Drive & Colab
2. **GitHub Account** - for code repository
3. One of these annotation tools:
   - [Label Studio](https://labelstud.io/) (self-hosted or cloud)
   - [Roboflow](https://roboflow.com/) (free tier)
   - [Makesense.ai](https://makesense.ai/) (browser-based, no signup)

### Skills Needed
- Basic Python (you'll be guided through everything)
- Photography eye (you already have this! ✅)
- Patience (annotation takes time)

---

## 🎬 Phase 1: Data Collection

### Step 1.1: Extract Video Frames

You mentioned using Egyptian league matches. Here's how to extract frames:

**Install ffmpeg** (if not installed):
```bash
# Windows (via chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Extract frames** (1 per second):
```bash
ffmpeg -i egyptian_league_match.mp4 -vf "fps=1" output/frames/%06d.jpg
```

**Recommended extraction strategy:**
```bash
# Extract 1 frame every 2 seconds (500 frames from 15min footage)
ffmpeg -i match1.mp4 -vf "fps=0.5" dataset/match1/%06d.jpg

# Extract specific time range (2nd half only)
ffmpeg -i match2.mp4 -ss 00:45:00 -t 00:45:00 -vf "fps=0.5" dataset/match2/%06d.jpg
```

### Step 1.2: Frame Selection Guidelines

**Aim for diversity:**
- ✅ Different stadiums (3-5 unique venues)
- ✅ Different times of day (day games, night games)
- ✅ Different camera angles (broadcast, tactical, corner cams)
- ✅ Different weather (clear, cloudy, shadows)
- ✅ Different pitch conditions (wet, dry, worn)

**Target dataset size:**
| Purpose | Minimum | Recommended | Pro |
|---------|---------|-------------|-----|
| MVP | 300 frames | 500 frames | 1000+ frames |
| Training | 240 (80%) | 400 (80%) | 800 (80%) |
| Validation | 60 (20%) | 100 (20%) | 200 (20%) |

**Quality checklist for each frame:**
- [ ] At least 40% of pitch visible
- [ ] Not during replay/crowd shot/closeup
- [ ] Reasonably sharp (not motion-blurred)
- [ ] Different from previous frame (temporal diversity)

### Step 1.3: Legal Considerations

**✅ Safe sources:**
- Training fields you recorded yourself
- Amateur league matches (public domain)
- Creative Commons sports footage

**⚠️ Risky sources:**
- Broadcast TV footage (check license)
- YouTube videos (check uploader rights)
- Professional league footage (often copyrighted)

**Practical approach for Egyptian league:**
1. Use for R&D/training privately ✅
2. Before commercial launch, verify rights or replace with self-recorded footage
3. Keep dataset separate from product (don't redistribute)

---

## 🎨 Phase 2: Data Annotation

### Step 2.1: Choose Your Tool

| Tool | Best For | Setup Time | Team Support |
|------|----------|------------|--------------|
| **Label Studio** | Full control, self-host | 15 min | ✅ Multi-user |
| **Roboflow** | Quick start, cloud | 5 min | ✅ Free tier |
| **Makesense.ai** | Solo, no install | 0 min | ❌ Single user |

**Recommendation:** Start with Makesense.ai for first 50 frames to learn, then move to Label Studio/Roboflow for team annotation.

### Step 2.2: Annotation Instructions

**What to annotate:**
- Draw a **polygon** around the **playable grass area**
- Include the field markings (lines)
- Exclude stands, advertising, sidelines, warm-up areas

**Example:**
```
✅ CORRECT:                    ❌ WRONG:
┌─────────────┐              ┌─────────────┐
│░░░░░░░░░░░░░│              │░░█STAND█░░░░│ ← Includes stands
│░░░PITCH░░░░░│              │░░░░░░░░░░░░░│
│░░░░░░░░░░░░░│              │░▓SIDELINE▓░░│ ← Includes sideline
└─────────────┘              └─────────────┘
```

**Annotation tips:**
1. **Zoom in** to get accurate edges
2. **Don't overthink** - rough polygons are fine (8-12 points usually enough)
3. **Consistent rules** - always include/exclude the same features
4. **Label shadows as pitch** - if grass is under shadow, include it
5. **Skip broken frames** - if <30% pitch visible, delete the frame

### Step 2.3: Tool-Specific Guides

#### Using Label Studio

1. **Setup (Docker):**
```bash
docker run -it -p 8080:8080 -v $(pwd)/labelstudio:/label-studio/data heartexlabs/label-studio:latest
```

2. **Create Project:**
   - Name: "Atlas v2 Pitch Segmentation"
   - Data Type: Images
   - Labeling Config:
   ```xml
   <View>
     <Image name="image" value="$image"/>
     <PolygonLabels name="label" toName="image">
       <Label value="pitch" background="green"/>
     </PolygonLabels>
   </View>
   ```

3. **Import frames:**
   - Upload folder of JPGs
   - Start annotating with polygon tool

4. **Export:**
   - Format: COCO JSON or PNG masks
   - Download to `dataset/annotations/`

#### Using Roboflow

1. **Create Project:**
   - Go to app.roboflow.com
   - New Project → Semantic Segmentation
   - Upload image folder

2. **Annotate:**
   - Smart Polygon tool
   - Class: "pitch"
   - Can use AI-assisted labeling after first 20-30 frames

3. **Export:**
   - Generate → Export → Format: "PNG Mask"
   - Download ZIP

#### Using Makesense.ai

1. **Open** makesense.ai in browser
2. **Drop images** into the interface
3. **Create label** "pitch"
4. **Draw polygons** with polygon tool
5. **Export** as:
   - Format: VGG JSON (then convert to masks)
   - Or use their PNG mask export

### Step 2.4: Quality Control

**Annotate in batches:**
- Day 1: 50 frames
- Day 2: Review + fix + 50 more
- Day 3: 50 more
- ...continue...

**Self-review checklist:**
- [ ] All pitch edges are captured
- [ ] No large gaps in polygon
- [ ] Consistent across similar frames
- [ ] Export test: masks look correct

---

## 💾 Phase 3: Dataset Organization

### Step 3.1: Standard Structure

Create this folder layout:
```
atlas_v2_dataset/
├── README.md                    # Dataset info
├── images/
│   ├── train/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── val/
│       ├── 000401.jpg
│       ├── 000402.jpg
│       └── ...
├── masks/
│   ├── train/
│   │   ├── 000001_mask.png    # Binary: 255=pitch, 0=bg
│   │   ├── 000002_mask.png
│   │   └── ...
│   └── val/
│       ├── 000401_mask.png
│       ├── 000402_mask.png
│       └── ...
└── metadata/
    ├── train.txt              # List of training image names
    ├── val.txt                # List of validation image names
    └── dataset_stats.json     # Optional: stats about dataset
```

### Step 3.2: Train/Val Split

**Script to split dataset:**
```python
import os
import shutil
from pathlib import Path
import random

def split_dataset(source_images, source_masks, output_dir, split_ratio=0.8):
    """
    Split dataset into train/val sets
    
    Args:
        source_images: Path to all images
        source_masks: Path to all masks
        output_dir: Output directory
        split_ratio: Fraction for training (0.8 = 80% train, 20% val)
    """
    # Get all image files
    images = sorted(Path(source_images).glob("*.jpg"))
    random.shuffle(images)
    
    # Calculate split
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Create directories
    (Path(output_dir) / "images" / "train").mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "images" / "val").mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "masks" / "train").mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "masks" / "val").mkdir(parents=True, exist_ok=True)
    
    # Copy train set
    for img_path in train_images:
        mask_path = Path(source_masks) / f"{img_path.stem}_mask.png"
        
        shutil.copy(img_path, Path(output_dir) / "images" / "train" / img_path.name)
        shutil.copy(mask_path, Path(output_dir) / "masks" / "train" / mask_path.name)
    
    # Copy val set
    for img_path in val_images:
        mask_path = Path(source_masks) / f"{img_path.stem}_mask.png"
        
        shutil.copy(img_path, Path(output_dir) / "images" / "val" / img_path.name)
        shutil.copy(mask_path, Path(output_dir) / "masks" / "val" / mask_path.name)
    
    print(f"✅ Split complete:")
    print(f"   Training: {len(train_images)} images")
    print(f"   Validation: {len(val_images)} images")

# Usage
split_dataset(
    source_images="raw_images/",
    source_masks="raw_masks/",
    output_dir="atlas_v2_dataset/",
    split_ratio=0.8
)
```

### Step 3.3: Upload to Google Drive

1. **Compress dataset:**
```bash
zip -r atlas_v2_dataset.zip atlas_v2_dataset/
```

2. **Upload to Drive:**
   - Go to drive.google.com
   - Create folder: "NovaVista"
   - Upload `atlas_v2_dataset.zip`
   - Right-click → Get link → Copy

3. **Note the file ID:**
   - URL format: `https://drive.google.com/file/d/FILE_ID_HERE/view`
   - Save this FILE_ID for Colab

---

## 🧠 Phase 4: Model Training (Google Colab)

### Step 4.1: Training Notebook Setup

I'll create a complete Colab notebook for you. Here's the outline:

**Notebook sections:**
1. Mount Google Drive
2. Install dependencies
3. Load dataset
4. Define UNet architecture
5. Training loop
6. Validation
7. Save best model
8. Visualize results

### Step 4.2: Training Parameters

**Recommended hyperparameters:**
```python
CONFIG = {
    # Data
    "image_size": 512,          # Input size (resize to 512×512)
    "batch_size": 8,            # Adjust based on GPU memory
    "num_workers": 2,
    
    # Training
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    
    # Augmentation
    "use_augmentation": True,
    "flip_prob": 0.5,
    "rotate_prob": 0.3,
    "brightness_range": (0.8, 1.2),
    
    # Loss
    "loss_fn": "bce_dice",     # BCE + Dice loss
    "dice_weight": 0.5,
    
    # Optimizer
    "optimizer": "adam",
    
    # Scheduler
    "scheduler": "cosine",
    "warmup_epochs": 5,
    
    # Early stopping
    "patience": 10,
    
    # Logging
    "log_interval": 10,         # Log every N batches
}
```

### Step 4.3: Expected Training Time

| GPU Type | Colab Tier | 50 Epochs | Est. Time |
|----------|------------|-----------|-----------|
| Tesla T4 | Free | 500 images | ~2 hours |
| Tesla T4 | Free | 1000 images | ~4 hours |
| Tesla L4 | Free (sometimes) | 1000 images | ~2 hours |
| A100 | Pro ($10/mo) | 1000 images | ~1 hour |

**Tips for free Colab:**
- Train during off-peak hours
- Save checkpoints every 5 epochs
- If disconnected, resume from checkpoint
- Can split into multiple sessions

### Step 4.4: Monitoring Training

**Watch these metrics:**

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Training Loss | Decreasing | Model is learning |
| Val IoU | >0.85 (epoch 30+) | Good segmentation |
| Val Loss | Stable/decreasing | Not overfitting |
| Train/Val gap | <0.05 | Not overfitting |

**Red flags:**
- ❌ Val loss increasing while train loss decreases → Overfitting
- ❌ Both losses not decreasing → Learning rate too high/low
- ❌ Val IoU plateaus below 0.80 → Need more data or better augmentation

---

## 📊 Phase 5: Evaluation & Export

### Step 5.1: Validation Metrics

**Calculate on validation set:**
```python
def evaluate_model(model, val_loader, device):
    """Calculate validation metrics"""
    model.eval()
    ious = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Predict
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            # Calculate IoU per image
            for pred, mask in zip(preds, masks):
                iou = calculate_iou(pred, mask)
                ious.append(iou)
    
    return {
        "mean_iou": np.mean(ious),
        "median_iou": np.median(ious),
        "min_iou": np.min(ious),
        "std_iou": np.std(ious)
    }

def calculate_iou(pred, target):
    """Intersection over Union"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / (union + 1e-8)).item()
```

**Target scores:**
- Mean IoU ≥ 0.90
- Median IoU ≥ 0.92
- Min IoU ≥ 0.75 (some frames can be harder)

### Step 5.2: Visual Inspection

**Generate comparison images:**
```python
import matplotlib.pyplot as plt

def visualize_predictions(model, val_dataset, num_samples=10):
    """Show side-by-side: image, ground truth, prediction"""
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        img, mask = val_dataset[i]
        pred = model(img.unsqueeze(0))[0]
        pred = (pred > 0.5).float()
        
        # Original image
        axes[i, 0].imshow(img.permute(1,2,0))
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(mask.squeeze(), cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(pred.squeeze().cpu(), cmap='gray')
        axes[i, 2].set_title(f"Prediction (IoU: {calculate_iou(pred, mask):.3f})")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("validation_comparison.png", dpi=150)
```

### Step 5.3: Export Model

**Save for production use:**
```python
# Save full model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'val_iou': best_iou,
    'epoch': best_epoch
}, 'best_model.pth')

# Optional: Export to ONNX for faster inference
dummy_input = torch.randn(1, 3, 512, 512).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "best_model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

print("✅ Model exported:")
print("   - PyTorch: best_model.pth")
print("   - ONNX: best_model.onnx")
```

---

## 📦 Phase 6: Integration into Atlas v2

### Step 6.1: Download Trained Model

From Colab:
```python
# Download to local machine
from google.colab import files
files.download('best_model.pth')
```

### Step 6.2: Add to Repository

```bash
# In your atlas project
mkdir -p atlas/v2/segmentation/weights
cp ~/Downloads/best_model.pth atlas/v2/segmentation/weights/
```

### Step 6.3: Create Inference Wrapper

See `IMPLEMENTATION_PLAN.md` for the full code to integrate the model.

---

## 🎯 Success Criteria Checklist

Before moving to v2 implementation:

**Dataset Quality:**
- [ ] ≥500 annotated frames
- [ ] 3+ different stadiums
- [ ] Day and night games included
- [ ] Train/val split done correctly

**Model Performance:**
- [ ] Mean IoU ≥ 0.90 on validation set
- [ ] Visual inspection: masks look accurate
- [ ] Inference speed: <100ms per frame
- [ ] Model file size: <50MB

**Documentation:**
- [ ] Dataset README written
- [ ] Training config saved
- [ ] Best model checkpointed
- [ ] Validation results logged

---

## 🆘 Troubleshooting

### Low IoU (<0.80)
**Possible causes:**
- Not enough training data → Collect more frames
- Poor annotation quality → Re-annotate sample
- Wrong hyperparameters → Try higher learning rate
- Need more augmentation → Enable all augmentations

### Overfitting (train IoU >> val IoU)
**Solutions:**
- Add more augmentation
- Reduce model size
- Add dropout
- Collect more validation data
- Early stopping

### Colab Disconnects
**Solutions:**
- Save checkpoints frequently
- Use Colab Pro for longer sessions
- Resume training from last checkpoint
- Split training into multiple runs

### Slow Training
**Solutions:**
- Reduce batch size if GPU memory issue
- Reduce image size to 384×384
- Use mixed precision training (FP16)
- Check GPU is actually being used

---

## 📚 Additional Resources

- **UNet Paper:** https://arxiv.org/abs/1505.04597
- **Semantic Segmentation Tutorial:** https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- **Label Studio Docs:** https://labelstud.io/guide/
- **Roboflow Guides:** https://roboflow.com/formats
- **Google Colab Tips:** https://colab.research.google.com/notebooks/pro.ipynb

---

## ✅ Next Steps

Once training is complete:
1. ✅ Download trained model
2. → Proceed to `IMPLEMENTATION_PLAN.md`
3. → Integrate model into Atlas v2
4. → Test on real Egyptian league footage
5. → Deploy to production

---

**Document Version:** 1.0  
**Last Updated:** October 2024  
**Estimated Time:** 2-3 days for data collection + annotation, 0.5 days for training  
**Difficulty:** Intermediate  
**Prerequisites:** Photography skills ✅, Basic Python, Patience
