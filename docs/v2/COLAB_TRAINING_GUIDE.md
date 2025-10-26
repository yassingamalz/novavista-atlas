# 🚀 Sportlight Training on Google Colab

## 📋 Overview

This guide shows how to train Sportlight's HRNet model on Google Colab instead of using Docker on Linux.

**Why Colab:**
- No Linux/Docker setup needed
- Free GPU access (T4 with 16GB VRAM)
- Cloud-based training
- Easy dataset management

**Challenge:**
- Sportlight requires 24GB VRAM (original setup)
- Colab T4 has only 16GB VRAM
- **Solution**: Reduce batch size and image resolution

---

## ⚙️ Training Requirements

### Original Sportlight Config:
```yaml
batch_size: 8
input_size: [960, 540]  # Half HD
GPU: 24GB VRAM
```

### Modified Colab Config:
```yaml
batch_size: 4           # Reduced from 8
input_size: [720, 405]  # Reduced from [960, 540]
GPU: 16GB VRAM (T4)
```

---

## 📦 Dataset - Downloaded Automatically in Colab!

The Sportlight model is trained on the **SoccerNet Camera Calibration Challenge** dataset.

**Good News:** The notebook downloads the dataset automatically using the official SoccerNet API!

**No manual download needed** - just run the notebook and it will:
1. Install SoccerNet package
2. Download train, valid, test splits (~5-10 GB)
3. Set up the correct directory structure
4. Verify all files are valid

**Dataset Details:**
- **Source:** Official SoccerNet Camera Calibration Challenge
- **Size:** ~5-10 GB (train + valid + test)
- **Format:** Paired `.jpg` images and `.json` annotations
- **Download Time:** 10-20 minutes (depends on connection)
- **Location:** `/content/soccernet_data` in Colab

**Dataset Structure (created automatically):**
```
/content/soccernet_data/
├── train/
│   ├── image1.jpg
│   ├── image1.json
│   ├── image2.jpg
│   ├── image2.json
│   └── ...
├── valid/
│   ├── image1.jpg
│   ├── image1.json
│   └── ...
└── test/
    ├── image1.jpg
    ├── image1.json
    └── ...
```

**Download Code (runs automatically in notebook):**
```python
from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory="/content/soccernet_data")
downloader.downloadDataTask(
    task="camera-calibration",
    split=["train", "valid", "test"]
)
```

### 2. JSON Annotation Format

Each image has a corresponding JSON file with field annotations:

```json
{
  "pitch_width": 68.0,
  "pitch_length": 105.0,
  "annotations": {
    "lines": [
      {
        "name": "Big rect. left main",
        "x1": 0.123, "y1": 0.456,
        "x2": 0.789, "y2": 0.234
      }
    ],
    "circles": [...],
    "conics": [...]
  }
}
```

The training code processes these annotations to generate 57 keypoints.

---

## 🎓 Colab Training Notebook

### Step 1: Start Colab & Enable GPU

```python
# No Drive mounting needed - dataset downloads directly!
```

### Step 2: Clone Sportlight Repository

```python
!git clone https://github.com/NikolasEnt/soccernet-calibration-sportlight.git
%cd soccernet-calibration-sportlight
```

### Step 3: Install Dependencies

```python
# Install required packages
!pip install torch torchvision
!pip install opencv-python
!pip install hydra-core
!pip install omegaconf
!pip install albumentations

# Argus framework (install from GitHub, not available on PyPI)
!pip install git+https://github.com/lRomul/argus.git

# Install SoccerNet for dataset download
!pip install SoccerNet
```

### Step 4: Download SoccerNet Dataset (Automatic!)

```python
from SoccerNet.Downloader import SoccerNetDownloader

print("📦 Downloading SoccerNet dataset...")
print("⏰ This takes 10-20 minutes")

downloader = SoccerNetDownloader(LocalDirectory="/content/soccernet_data")
downloader.downloadDataTask(
    task="camera-calibration",
    split=["train", "valid", "test"]
)

print("✅ Dataset downloaded!")
```

### Step 5: Setup Dataset Path

```python
import os

# Create workdir structure
os.makedirs("/workdir/data/dataset", exist_ok=True)
os.makedirs("/workdir/data/experiments", exist_ok=True)

# Link to downloaded dataset
!ln -s /content/soccernet_data/train /workdir/data/dataset/train
!ln -s /content/soccernet_data/valid /workdir/data/dataset/valid
```

```python
# Read current config
import yaml

config_path = "src/models/hrnet/train_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Modify for Colab constraints
config['data_params']['batch_size'] = 4  # Reduced from 8
config['data_params']['input_size'] = [720, 405]  # Reduced from [960, 540]
config['data_params']['num_workers'] = 2  # Reduced from 8
config['model']['params']['loss']['pred_size'] = [203, 360]  # Adjusted
config['model']['params']['prediction_transform']['size'] = [405, 720]

# Save modified config
with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("✅ Config modified for Colab")
```

### Step 6: Start Training

```python
# Train HRNet keypoint model
!python src/models/hrnet/train.py
```

**Training Output:**
```
Epoch 1/200: loss=0.045, val_loss=0.038, val_evalai=0.62
Epoch 2/200: loss=0.042, val_loss=0.035, val_evalai=0.65
...
```

**Training Time:**
- With Colab T4 GPU: ~6-10 hours for 200 epochs
- Early stopping typically triggers around epoch 100-150

### Step 7: Monitor Training

```python
# View training logs
!tail -f /workdir/data/experiments/HRNet_57_*/log.txt
```

### Step 8: Download Trained Model

```python
# After training completes
from google.colab import files
import glob

# Find best model
models = glob.glob("/workdir/data/experiments/HRNet_57_*/evalai-*.pth")
if models:
    best_model = sorted(models)[-1]
    print(f"📦 Best model: {best_model}")
    
    # Download to your computer
    files.download(best_model)
    print("✅ Model downloaded to your Downloads folder!")
else:
    print("❌ No models found")
```

---

## 🔧 Troubleshooting

### Out of Memory Error

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Further reduce batch size:**
```python
config['data_params']['batch_size'] = 2  # From 4 to 2
```

2. **Reduce image resolution:**
```python
config['data_params']['input_size'] = [640, 360]  # From [720, 405]
config['model']['params']['loss']['pred_size'] = [180, 320]
config['model']['params']['prediction_transform']['size'] = [360, 640]
```

3. **Enable gradient checkpointing:**
```python
# Add to model config
config['model']['params']['gradient_checkpointing'] = True
```

4. **Use mixed precision training:**
```python
# Already enabled in default config
config['model']['params']['amp'] = True
```

### Colab Timeout

**Issue:** Free Colab sessions timeout after 12 hours

**Solutions:**

1. **Save checkpoints frequently:**
```python
# Already in config - saves every 2 epochs
config['train_params']['save_period'] = 2
```

2. **Resume from checkpoint:**
```python
# Add to config before training
config['model']['params']['pretrain'] = "/path/to/checkpoint.pth"
```

3. **Use Colab Pro** ($10/month):
- 24-hour session limit
- Priority GPU access
- Background execution

### Dataset Issues

**Issue:** JSON annotations not loading

**Check:**
```python
import json
import os

# Verify JSON format
sample_json = "/workdir/data/dataset/train/image1.json"
with open(sample_json) as f:
    data = json.load(f)
    print(data.keys())

# Should contain: pitch_width, pitch_length, annotations
```

**Fix corrupted JSON:**
```python
import json
import glob

for json_path in glob.glob("/workdir/data/dataset/train/*.json"):
    try:
        with open(json_path) as f:
            json.load(f)
    except:
        print(f"Corrupted: {json_path}")
```

---

## 📊 Expected Results

### Training Metrics

After 200 epochs with the SoccerNet dataset:

```
Keypoint Detection Accuracy: ~70-75%
Mean L2 Distance: < 5 pixels
EvalAI Score: ~0.70-0.73
```

### Validation on Egyptian League

After training, test on Egyptian League footage:
- Expected completeness: 75-80%
- Expected accuracy: 70-75%

If results are lower, you may need to:
1. Fine-tune on Egyptian League data
2. Activate hybrid approach (Spiideo fallback)

---

## 🚀 Complete Colab Notebook

Save this as `sportlight_training.ipynb`:

```python
# ========================================
# 🎯 Sportlight HRNet Training on Colab
# ========================================

# 1️⃣ Check GPU
print("📦 Checking GPU...")
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2️⃣ Clone Repository
print("📥 Cloning Sportlight...")
!git clone https://github.com/NikolasEnt/soccernet-calibration-sportlight.git
%cd soccernet-calibration-sportlight

# 3️⃣ Install Dependencies
print("⚙️ Installing dependencies...")
!pip install -q torch torchvision opencv-python hydra-core argus-learn omegaconf albumentations scipy scikit-image
!pip install -q SoccerNet  # For dataset download

# 4️⃣ Download SoccerNet Dataset
print("📦 Downloading SoccerNet dataset...")
print("⏰ This takes 10-20 minutes")
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory="/content/soccernet_data")
downloader.downloadDataTask(task="camera-calibration", split=["train", "valid", "test"])
print("✅ Dataset downloaded!")

# 5️⃣ Setup Data Directories
print("📂 Setting up data directories...")
import os
os.makedirs("/workdir/data/dataset", exist_ok=True)
os.makedirs("/workdir/data/experiments", exist_ok=True)

# Create links to dataset
!ln -sf /content/soccernet_data/train /workdir/data/dataset/train
!ln -sf /content/soccernet_data/valid /workdir/data/dataset/valid
print("✅ Data directories ready!")

# 6️⃣ Modify Config for Colab
print("🔧 Modifying config for Colab...")
import yaml
config_path = "src/models/hrnet/train_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Colab-friendly settings
config['data_params']['batch_size'] = 4
config['data_params']['input_size'] = [720, 405]
config['data_params']['num_workers'] = 2
config['model']['params']['loss']['pred_size'] = [203, 360]
config['model']['params']['prediction_transform']['size'] = [405, 720]
config['model']['params']['amp'] = True

with open(config_path, 'w') as f:
    yaml.dump(config, f)
print("✅ Config updated!")

# 7️⃣ Start Training
print("🚀 Starting training...")
print("⏰ Expected time: 6-10 hours")
!python src/models/hrnet/train.py

# 8️⃣ Download Model
print("💾 Downloading trained model...")
from google.colab import files
import glob

models = glob.glob("/workdir/data/experiments/HRNet_57_*/evalai-*.pth")
if models:
    best_model = sorted(models)[-1]
    print(f"📦 Best model: {best_model}")
    files.download(best_model)
    print("✅ Model downloaded!")
else:
    print("❌ No models found")
```

---

## 📝 Next Steps After Training

1. **Download trained model** to your local machine
2. **Move to Phase 2:** Test on Egyptian League frames
3. **Integrate into Atlas pipeline** (Phase 3)
4. **Validate performance** (Phase 5)

---

## 🆘 Getting Help

**If training fails:**

1. Check Sportlight GitHub issues
2. Review SoccerNet documentation
3. Post in SoccerNet Discord/Slack
4. Contact paper authors (emails in paper)

**Alternative: Pre-trained Models**

If training doesn't work, search for:
- "Sportlight pre-trained model download"
- Contact original authors: Nikolay Falaleev
- Check if anyone shared models on Hugging Face

---

**Status:** Ready for training on Colab
**Expected Training Time:** 6-10 hours
**Expected Model Size:** ~200-300 MB
**Next Document:** `PHASE_2_TESTING.md` (after training completes)
