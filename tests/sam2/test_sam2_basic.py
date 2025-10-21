"""
SAM2 Test Script - Process all images in test directory
Uses LOCAL checkpoint (automatic CPU/GPU support)
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import time

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_sam2_all_images():
    """Test SAM2 on all images in test_data/frames/"""

    # Disable GPU if torch not built with CUDA
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[WARNING] CUDA not available or not compiled. Running on CPU mode.")

    # Paths (absolute paths)
    project_root = Path(__file__).parent.parent.parent
    checkpoint = str(project_root / "atlas/models/sam2/checkpoints/sam2.1_hiera_large.pt")
    config_path = str(project_root / "atlas/models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    test_dir = project_root / "test_data/frames"

    # Output directory with date
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = project_root / f"output/sam2/results/{today}"
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in test_dir.iterdir() if f.suffix.lower() in image_extensions]

    print(f"Found {len(image_files)} images in {test_dir}")
    print(f"Using checkpoint: {checkpoint}")
    print(f"Using config: {config_path}")

    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running inference on device: {device.upper()}")

    # Load SAM2 model
    print("\nLoading SAM2 model from local checkpoint...")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(config_path, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    print("[OK] Model loaded successfully from local checkpoint!\n")

    # Choose precision dynamically
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")

        try:
            start_time = time.time()

            # Load image
            image = np.array(Image.open(image_path).convert("RGB"))
            print(f"  Image shape: {image.shape}")

            # Run inference
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
                predictor.set_image(image)

                # Use center point as prompt
                point_coords = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
                point_labels = np.array([1])  # 1 = foreground

                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

            duration = time.time() - start_time
            print(f"  [OK] Generated {len(masks)} masks | Scores: {scores} | Time: {duration:.2f}s")

            # Save masks as image overlays
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(image)
            plt.plot(point_coords[0, 0], point_coords[0, 1], 'r*', markersize=15)
            plt.title("Input + Prompt")
            plt.axis('off')

            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.subplot(1, 4, i + 2)
                plt.imshow(image)
                plt.imshow(mask, alpha=0.5, cmap='jet')
                plt.title(f"Mask {i+1} (Score: {score:.3f})")
                plt.axis('off')

            plt.tight_layout()
            output_path = output_dir / f"{image_path.stem}_sam2_result.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Saved: {output_path}\n")

        except Exception as e:
            print(f"  [ERROR] Error processing {image_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 60}")
    print(f"[OK] Processing complete! Results saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    test_sam2_all_images()
