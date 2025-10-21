"""
ATLAS V2 - Stage A: Segmentation Pipeline Test (Oversegmentation Strategy)
Handles aerial, broadcast, and ground-level views adaptively

Strategy: Oversegment in Stage A, refine with lines in Stage B
Better to capture too much area than miss field edges

Pipeline:
1. Original Image
2. HSV Preprocessing (aggressive green detection)
3. SAM 2.1 Segmentation (generous strategies by view type)
4. Geometric Expansion (add margin to ensure full field coverage)

Output: Oversegmented mask saved for Stage B line refinement
"""
import os
import sys
import time
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
from dataclasses import dataclass

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


@dataclass
class ViewClassification:
    """Classification of camera view type"""
    view_type: str  # 'aerial', 'broadcast', 'ground'
    confidence: float
    field_coverage: float
    vertical_angle: float


@dataclass
class SegmentationResult:
    """Results from segmentation stage"""
    mask: np.ndarray
    score: float
    strategy: str
    view_type: str
    inference_time: float


class ViewClassifier:
    """Classify camera view type to select optimal strategy"""
    
    @staticmethod
    def classify(image: np.ndarray) -> ViewClassification:
        """Determine if aerial, broadcast (stadium), or ground level view"""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Aggressive green detection for all lighting conditions
        lower_green = np.array([25, 25, 25])  # More permissive
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        field_coverage = np.sum(green_mask > 0) / (h * w)
        
        # Analyze field distribution
        green_y_dist = np.sum(green_mask, axis=1)
        top_third = np.sum(green_y_dist[:h//3])
        middle_third = np.sum(green_y_dist[h//3:2*h//3])
        bottom_third = np.sum(green_y_dist[2*h//3:])
        
        # Aerial: field fills frame evenly, high coverage
        if field_coverage > 0.55:  # Lower threshold
            view_type = 'aerial'
            confidence = 0.95
            vertical_angle = 90.0
        
        # Broadcast: field in lower portion
        elif bottom_third > top_third * 1.3:  # More permissive
            view_type = 'broadcast'
            confidence = 0.85
            vertical_angle = 45.0
        
        # Ground level: field in lower portion, lower coverage
        elif bottom_third > middle_third:
            view_type = 'ground'
            confidence = 0.80
            vertical_angle = 15.0
        
        # Uncertain - default to broadcast (most common)
        else:
            view_type = 'broadcast'
            confidence = 0.70
            vertical_angle = 45.0
        
        return ViewClassification(
            view_type=view_type,
            confidence=confidence,
            field_coverage=field_coverage,
            vertical_angle=vertical_angle
        )


class HSVPreprocessor:
    """HSV-based preprocessing for field isolation - AGGRESSIVE OVERSEGMENTATION"""
    
    @staticmethod
    def extract_field_mask(image: np.ndarray, aggressive: bool = True) -> np.ndarray:
        """Extract green field using HSV with very wide ranges for oversegmentation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Multiple wide ranges for all lighting conditions
        green_ranges = [
            ([20, 20, 20], [100, 255, 255]),    # Very wide range
            ([25, 15, 15], [95, 255, 255]),      # Include darker greens
            ([30, 30, 30], [90, 255, 240]),      # Mid-range
            ([35, 40, 40], [85, 255, 255]),      # Standard green
        ]
        
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Aggressive morphological expansion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Dilate to ensure we capture field edges
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=1)
        
        return combined_mask > 0
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, boost: bool = False) -> np.ndarray:
        """CLAHE enhancement - boosted for difficult lighting"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clip_limit = 4.0 if boost else 3.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb


class AdaptiveSAMStrategy:
    """Select SAM strategy based on view type - GENEROUS STRATEGIES"""
    
    @staticmethod
    def get_strategies(view_type: str) -> list:
        """Return ordered strategy list - prioritize oversegmentation"""
        
        if view_type == 'aerial':
            # Aerial: box strategies with generous margins
            return ['box_strategy_wide', 'box_point_hybrid', 'multi_point_dense']
        
        elif view_type == 'broadcast':
            # Broadcast: generous strategies to capture full field
            return ['broadcast_overseg', 'multi_point_dense', 'box_strategy_wide', 'pos_neg_generous']
        
        elif view_type == 'ground':
            # Ground level: focus on visible portion but be generous
            return ['lower_generous', 'multi_point_dense', 'broadcast_overseg']
        
        else:
            return ['multi_point_dense', 'box_strategy_wide', 'broadcast_overseg']


class MorphologicalExpander:
    """Expand masks to ensure full field coverage"""
    
    @staticmethod
    def expand_mask(mask: np.ndarray, expansion_percent: float = 0.12) -> np.ndarray:
        """Expand mask by percentage to ensure field edges are captured"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find bounding box
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Expand by percentage
        expand_x = int(w * expansion_percent)
        expand_y = int(h * expansion_percent)
        
        x_new = max(0, x - expand_x)
        y_new = max(0, y - expand_y)
        w_new = min(mask.shape[1] - x_new, w + 2 * expand_x)
        h_new = min(mask.shape[0] - y_new, h + 2 * expand_y)
        
        # Create expanded mask
        expanded = np.zeros_like(mask_uint8)
        expanded[y_new:y_new+h_new, x_new:x_new+w_new] = 255
        
        # Combine with original (union)
        combined = cv2.bitwise_or(mask_uint8, expanded)
        
        # Smooth
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return (combined > 127).astype(bool)


def execute_strategy(strategy_name: str, predictor, image: np.ndarray, 
                     hsv_mask: np.ndarray, w: int, h: int, dtype):
    """Execute SAM strategy with generous prompts for oversegmentation"""
    
    if strategy_name == 'single_center':
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
        
    elif strategy_name == 'multi_point_dense':
        # Dense grid of points across entire image
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        point_coords = np.array([
            [margin_x, margin_y],
            [w // 2, margin_y],
            [w - margin_x, margin_y],
            [margin_x, h // 2],
            [w // 2, h // 2],
            [w - margin_x, h // 2],
            [margin_x, h - margin_y],
            [w // 2, h - margin_y],
            [w - margin_x, h - margin_y]
        ])
        point_labels = np.ones(len(point_coords))
    
    elif strategy_name == 'pos_neg_generous':
        # Fewer negative points, more positive
        point_coords = np.array([
            [w // 2, h // 2],
            [w * 0.25, h * 0.35],
            [w * 0.75, h * 0.35],
            [w * 0.25, h * 0.65],
            [w * 0.75, h * 0.65],
            [30, 30],  # Only corners as negative
            [w - 30, 30]
        ])
        point_labels = np.array([1, 1, 1, 1, 1, 0, 0])
    
    elif strategy_name == 'box_strategy_wide':
        # Very generous box margins
        margin_x, margin_y = int(w * 0.02), int(h * 0.02)  # Minimal margin
        box = np.array([margin_x, margin_y, w - margin_x, h - margin_y])
        with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        return masks, scores
    
    elif strategy_name == 'box_point_hybrid':
        margin_x, margin_y = int(w * 0.03), int(h * 0.03)
        box = np.array([margin_x, margin_y, w - margin_x, h - margin_y])
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
        with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
            masks, scores, _ = predictor.predict(
                point_coords=point_coords, point_labels=point_labels,
                box=box, multimask_output=True
            )
        return masks, scores
    
    elif strategy_name == 'broadcast_overseg':
        # Generous points for broadcast view - no upper exclusion
        cy = int(h * 0.55)  # Slightly lower center
        point_coords = np.array([
            [w // 2, cy],
            [w // 4, cy],
            [3 * w // 4, cy],
            [w // 4, cy - h // 6],
            [3 * w // 4, cy - h // 6],
            [w // 4, cy + h // 6],
            [3 * w // 4, cy + h // 6],
            [w // 2, h - 80]
        ])
        point_labels = np.ones(len(point_coords))
    
    elif strategy_name == 'lower_generous':
        # Focus on lower but be generous
        cy = int(h * 0.65)
        point_coords = np.array([
            [w // 2, cy],
            [w // 4, cy],
            [3 * w // 4, cy],
            [w // 2, cy - h // 8],
            [w // 2, h - 60]
        ])
        point_labels = np.ones(len(point_coords))
    
    else:
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
    
    # Execute prediction
    with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
        masks, scores, _ = predictor.predict(
            point_coords=point_coords, 
            point_labels=point_labels, 
            multimask_output=True
        )
    
    return masks, scores


def test_stage_a_segmentation():
    """Stage A: Oversegmentation pipeline test"""
    
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[WARNING] Running on CPU")
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    checkpoint = str(project_root / "atlas/models/sam2/checkpoints/sam2.1_hiera_large.pt")
    config_path = str(project_root / "atlas/models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    test_dir = project_root / "test_data/frames"
    
    # Output directory
    today = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = project_root / f"output/sam2/stage_a/{today}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    image_files = [f for f in test_dir.iterdir() 
                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not image_files:
        print("❌ No images found")
        return
    
    # Load SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device.upper()}")
    print("[INFO] Loading SAM 2.1...")
    
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    try:
        model = build_sam2(config_path, checkpoint, device=device)
    except TypeError:
        model = build_sam2(config_path, checkpoint)
        model = model.to(device)
    
    predictor = SAM2ImagePredictor(model)
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    print("✅ Model loaded!")
    print("\n🎯 STRATEGY: Oversegmentation → Let Stage B refine with lines\n")
    
    # Initialize components
    view_classifier = ViewClassifier()
    hsv_processor = HSVPreprocessor()
    strategy_selector = AdaptiveSAMStrategy()
    expander = MorphologicalExpander()
    
    # Process each image
    for img_idx, image_path in enumerate(image_files):
        print(f"\n{'='*70}")
        print(f"📸 Image {img_idx + 1}/{len(image_files)}: {image_path.name}")
        print(f"{'='*70}")
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        
        # Step 1: Classify view type
        print("\n[1/4] Classifying camera view...")
        view_info = view_classifier.classify(image)
        print(f"  View: {view_info.view_type.upper()} (conf: {view_info.confidence:.2f})")
        print(f"  Field coverage: {view_info.field_coverage:.1%}")
        
        # Step 2: Aggressive HSV preprocessing
        print("\n[2/4] Aggressive HSV preprocessing...")
        hsv_mask = hsv_processor.extract_field_mask(image, aggressive=True)
        enhanced_image = hsv_processor.enhance_contrast(image, boost=(view_info.confidence < 0.8))
        print(f"  HSV mask area: {hsv_mask.sum():,} pixels ({hsv_mask.sum()/(h*w):.1%} coverage)")
        
        # Step 3: SAM segmentation with generous strategies
        print(f"\n[3/4] SAM 2.1 oversegmentation ({view_info.view_type} strategies)...")
        strategies = strategy_selector.get_strategies(view_info.view_type)
        print(f"  Strategies: {strategies}")
        
        predictor.set_image(enhanced_image)
        
        best_result = None
        all_results = []
        
        for strategy_name in strategies:
            start = time.time()
            
            try:
                masks, scores = execute_strategy(
                    strategy_name, predictor, enhanced_image, hsv_mask, w, h, dtype
                )
                
                # Select largest mask (prefer oversegmentation)
                areas = [mask.sum() for mask in masks]
                best_idx = int(np.argmax(areas))  # Prioritize size over score
                
                result = SegmentationResult(
                    mask=masks[best_idx],
                    score=scores[best_idx],
                    strategy=strategy_name,
                    view_type=view_info.view_type,
                    inference_time=time.time() - start
                )
                
                all_results.append(result)
                
                print(f"  ✓ {strategy_name:25} | Score: {result.score:.3f} | "
                      f"Area: {result.mask.sum():,}px | Time: {result.inference_time:.2f}s")
                
                # Pick largest mask with reasonable score
                if best_result is None or result.mask.sum() > best_result.mask.sum():
                    if result.score > 0.7:  # Lower threshold
                        best_result = result
                    
            except Exception as e:
                print(f"  ✗ {strategy_name:25} | Failed: {str(e)}")
                continue
        
        if not best_result:
            print("❌ All strategies failed")
            continue
        
        # Step 4: Geometric expansion for safety margin
        print(f"\n[4/4] Expanding mask (12% margin)...")
        print(f"  Before expansion: {best_result.mask.sum():,}px")
        expanded_mask = expander.expand_mask(best_result.mask, expansion_percent=0.12)
        print(f"  After expansion: {expanded_mask.sum():,}px (+{expanded_mask.sum() - best_result.mask.sum():,}px)")
        
        # Calculate IoU with HSV
        iou = np.logical_and(expanded_mask, hsv_mask).sum() / np.logical_or(expanded_mask, hsv_mask).sum()
        print(f"  IoU with HSV: {iou:.3f}")
        
        # Save results
        print("\n[SAVE] Saving Stage A oversegmented mask...")
        
        # Save mask for Stage B
        mask_path = output_dir / f"{image_path.stem}_stage_a_mask.npy"
        np.save(mask_path, expanded_mask.astype(np.uint8))
        
        # Save visualization
        save_stage_a_visualization(
            image, hsv_mask, enhanced_image, all_results, 
            expanded_mask, best_result, view_info,
            output_dir / f"{image_path.stem}_stage_a.png"
        )
        
        print(f"  ✓ Mask saved: {mask_path.name}")
        print(f"  ✓ Visualization saved")
    
    print(f"\n{'='*70}")
    print(f"✅ Stage A complete! Oversegmented masks ready for Stage B refinement")
    print(f"Results: {output_dir}")
    print(f"{'='*70}")


def save_stage_a_visualization(image, hsv_mask, enhanced, results, 
                                expanded_mask, best_result, view_info, save_path):
    """Create Stage A visualization showing all steps"""
    
    fig = plt.figure(figsize=(24, 14))
    
    # Row 1: Original + HSV + Enhanced + Final
    plt.subplot(3, 4, 1)
    plt.imshow(image)
    plt.title(f"1. Original Image\nView: {view_info.view_type.upper()} "
              f"({view_info.confidence:.0%} conf)", fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(image)
    plt.imshow(hsv_mask, alpha=0.5, cmap='Greens')
    plt.title(f"2. Aggressive HSV Mask\nCoverage: {view_info.field_coverage:.1%}", 
              fontsize=11)
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(enhanced)
    plt.title("3. Contrast Enhanced (CLAHE)\nFed to SAM 2.1", fontsize=11)
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(image)
    plt.imshow(expanded_mask, alpha=0.6, cmap='plasma')
    plt.title(f"4. OVERSEGMENTED MASK (12% margin)\nStrategy: {best_result.strategy}\n"
              f"Area: {expanded_mask.sum():,}px | Score: {best_result.score:.3f}", 
              fontsize=11, fontweight='bold', color='green')
    plt.axis('off')
    
    # Row 2-3: Strategy results
    for idx, result in enumerate(results[:8]):
        plt.subplot(3, 4, idx + 5)
        plt.imshow(image)
        plt.imshow(result.mask, alpha=0.5, cmap='viridis')
        
        color = 'green' if result.strategy == best_result.strategy else 'black'
        marker = "★ " if result.strategy == best_result.strategy else ""
        plt.title(f"{marker}{result.strategy}\nScore: {result.score:.3f} | Area: {result.mask.sum():,}px", 
                  fontsize=9, color=color)
        plt.axis('off')
    
    plt.suptitle(f"STAGE A: Oversegmentation Pipeline - {view_info.view_type.upper()} View\n"
                 f"Strategy: Generous segmentation → Stage B will refine with lines", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    test_stage_a_segmentation()
