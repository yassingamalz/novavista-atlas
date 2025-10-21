"""
ATLAS V2 - Stage A: Segmentation Pipeline Test
Handles aerial, broadcast, and ground-level views adaptively

Pipeline:
1. Original Image
2. HSV Preprocessing (green mask + contrast enhancement)
3. SAM 2.1 Segmentation (adaptive strategy by view type)
4. Geometric Validation (morphological refinement)

Output: Best segmentation mask saved for Stage B
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
        
        # Detect green field
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        field_coverage = np.sum(green_mask > 0) / (h * w)
        
        # Analyze field distribution
        green_y_dist = np.sum(green_mask, axis=1)
        top_third = np.sum(green_y_dist[:h//3])
        middle_third = np.sum(green_y_dist[h//3:2*h//3])
        bottom_third = np.sum(green_y_dist[2*h//3:])
        
        # Aerial: field fills frame evenly, high coverage
        if field_coverage > 0.65:
            view_type = 'aerial'
            confidence = 0.95
            vertical_angle = 90.0
        
        # Broadcast: field mostly in lower 2/3, medium coverage
        elif field_coverage > 0.35 and bottom_third > top_third * 2:
            view_type = 'broadcast'
            confidence = 0.85
            vertical_angle = 45.0
        
        # Ground level: field in lower portion, lower coverage
        elif bottom_third > middle_third * 1.5:
            view_type = 'ground'
            confidence = 0.80
            vertical_angle = 15.0
        
        # Uncertain - default to broadcast
        else:
            view_type = 'broadcast'
            confidence = 0.60
            vertical_angle = 45.0
        
        return ViewClassification(
            view_type=view_type,
            confidence=confidence,
            field_coverage=field_coverage,
            vertical_angle=vertical_angle
        )


class HSVPreprocessor:
    """HSV-based preprocessing for field isolation and enhancement"""
    
    @staticmethod
    def extract_field_mask(image: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """Extract green field using HSV with multiple ranges"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        if aggressive:
            # Wider ranges for difficult lighting
            green_ranges = [
                ([25, 30, 30], [95, 255, 255]),
                ([35, 40, 40], [85, 255, 255]),
                ([30, 50, 50], [80, 255, 200])
            ]
        else:
            green_ranges = [
                ([35, 40, 40], [85, 255, 255]),
                ([30, 50, 50], [80, 255, 200])
            ]
        
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask > 0
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """CLAHE enhancement in LAB color space"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb


class AdaptiveSAMStrategy:
    """Select SAM strategy based on view type"""
    
    @staticmethod
    def get_strategies(view_type: str) -> list:
        """Return ordered strategy list for view type"""
        
        if view_type == 'aerial':
            # Aerial: box strategies work best
            return ['box_point_hybrid', 'box_strategy', 'multi_point_grid']
        
        elif view_type == 'broadcast':
            # Broadcast (stadium): need perspective-aware strategies
            return ['adaptive_broadcast', 'pos_neg_points', 'multi_point_grid', 'single_center']
        
        elif view_type == 'ground':
            # Ground level: focus on visible lower portion
            return ['lower_focus', 'pos_neg_points', 'single_center']
        
        else:
            # Fallback
            return ['single_center', 'multi_point_grid', 'box_point_hybrid']


class MorphologicalRefiner:
    """Refine masks with morphological operations"""
    
    @staticmethod
    def refine(mask: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """Clean up mask with morphological operations"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        if aggressive:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        else:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Close holes
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel)
        
        # Remove noise
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, open_kernel)
        
        # Keep largest contour
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            refined = np.zeros_like(mask_uint8)
            cv2.drawContours(refined, [largest], -1, 255, -1)
            
            # Smooth edges
            refined = cv2.GaussianBlur(refined, (5, 5), 0)
            return (refined > 127).astype(bool)
        
        return (mask_opened > 127).astype(bool)


def execute_strategy(strategy_name: str, predictor, image: np.ndarray, 
                     hsv_mask: np.ndarray, w: int, h: int, dtype):
    """Execute SAM strategy with HSV-guided prompts"""
    
    if strategy_name == 'single_center':
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
        
    elif strategy_name == 'multi_point_grid':
        margin_x, margin_y = int(w * 0.2), int(h * 0.2)
        point_coords = np.array([
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y],
            [w // 2, h // 2]
        ])
        point_labels = np.ones(len(point_coords))
    
    elif strategy_name == 'pos_neg_points':
        point_coords = np.array([
            [w // 2, h // 2],
            [w * 0.3, h * 0.3],
            [w * 0.7, h * 0.7],
            [50, 50],
            [w - 50, 50],
            [50, h - 50],
            [w - 50, h - 50]
        ])
        point_labels = np.array([1, 1, 1, 0, 0, 0, 0])
    
    elif strategy_name == 'box_strategy':
        margin_x, margin_y = int(w * 0.05), int(h * 0.05)
        box = np.array([margin_x, margin_y, w - margin_x, h - margin_y])
        with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        return masks, scores
    
    elif strategy_name == 'box_point_hybrid':
        margin_x, margin_y = int(w * 0.05), int(h * 0.05)
        box = np.array([margin_x, margin_y, w - margin_x, h - margin_y])
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
        with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
            masks, scores, _ = predictor.predict(
                point_coords=point_coords, point_labels=point_labels,
                box=box, multimask_output=True
            )
        return masks, scores
    
    elif strategy_name == 'adaptive_broadcast':
        # Use HSV mask to find field centroid for broadcast view
        moments = cv2.moments(hsv_mask.astype(np.uint8))
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = w // 2, int(h * 0.6)  # Lower center for broadcast
        
        # Points focused on visible field area
        offset_x, offset_y = w // 6, h // 6
        point_coords = np.array([
            [cx, cy],
            [cx - offset_x, cy],
            [cx + offset_x, cy],
            [cx, cy - offset_y],
            [cx, cy + offset_y],
            [w // 4, h - 100],  # Lower corners
            [3 * w // 4, h - 100]
        ])
        point_labels = np.array([1, 1, 1, 1, 1, 1, 1])
    
    elif strategy_name == 'lower_focus':
        # Focus on lower portion for ground-level views
        cy = int(h * 0.7)
        point_coords = np.array([
            [w // 2, cy],
            [w // 4, cy],
            [3 * w // 4, cy],
            [w // 2, h - 50]
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
    """Stage A: Segmentation pipeline test"""
    
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
    print("✅ Model loaded!\n")
    
    # Initialize components
    view_classifier = ViewClassifier()
    hsv_processor = HSVPreprocessor()
    strategy_selector = AdaptiveSAMStrategy()
    refiner = MorphologicalRefiner()
    
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
        
        # Step 2: HSV preprocessing
        print("\n[2/4] HSV preprocessing...")
        hsv_mask = hsv_processor.extract_field_mask(image, aggressive=(view_info.confidence < 0.8))
        enhanced_image = hsv_processor.enhance_contrast(image)
        print(f"  HSV mask area: {hsv_mask.sum():,} pixels")
        
        # Step 3: SAM segmentation with adaptive strategies
        print(f"\n[3/4] SAM 2.1 segmentation ({view_info.view_type} strategies)...")
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
                
                # Select best mask
                areas = [mask.sum() for mask in masks]
                best_idx = int(np.argmax([s * a for s, a in zip(scores, areas)]))
                
                result = SegmentationResult(
                    mask=masks[best_idx],
                    score=scores[best_idx],
                    strategy=strategy_name,
                    view_type=view_info.view_type,
                    inference_time=time.time() - start
                )
                
                all_results.append(result)
                
                print(f"  ✓ {strategy_name:25} | Score: {result.score:.3f} | "
                      f"Time: {result.inference_time:.2f}s")
                
                if best_result is None or result.score > best_result.score:
                    best_result = result
                    
            except Exception as e:
                print(f"  ✗ {strategy_name:25} | Failed: {str(e)}")
                continue
        
        if not best_result:
            print("❌ All strategies failed")
            continue
        
        # Step 4: Geometric refinement
        print(f"\n[4/4] Refining best result ({best_result.strategy})...")
        refined_mask = refiner.refine(best_result.mask, aggressive=True)
        
        # Calculate IoU with HSV
        iou = np.logical_and(refined_mask, hsv_mask).sum() / np.logical_or(refined_mask, hsv_mask).sum()
        print(f"  IoU with HSV: {iou:.3f}")
        
        # Save results
        print("\n[SAVE] Saving Stage A results...")
        
        # Save best mask for Stage B
        mask_path = output_dir / f"{image_path.stem}_stage_a_mask.npy"
        np.save(mask_path, refined_mask.astype(np.uint8))
        
        # Save visualization
        save_stage_a_visualization(
            image, hsv_mask, enhanced_image, all_results, 
            refined_mask, best_result, view_info,
            output_dir / f"{image_path.stem}_stage_a.png"
        )
        
        print(f"  ✓ Mask saved: {mask_path.name}")
        print(f"  ✓ Visualization saved")
    
    print(f"\n{'='*70}")
    print(f"✅ Stage A complete! Results: {output_dir}")
    print(f"{'='*70}")


def save_stage_a_visualization(image, hsv_mask, enhanced, results, 
                                refined_mask, best_result, view_info, save_path):
    """Create Stage A visualization showing all steps"""
    
    fig = plt.figure(figsize=(24, 14))
    
    # Row 1: Original + HSV + Enhanced
    plt.subplot(3, 4, 1)
    plt.imshow(image)
    plt.title(f"1. Original Image\nView: {view_info.view_type.upper()} "
              f"({view_info.confidence:.0%} conf)", fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(image)
    plt.imshow(hsv_mask, alpha=0.5, cmap='Greens')
    plt.title(f"2. HSV Field Mask\nCoverage: {view_info.field_coverage:.1%}", 
              fontsize=11)
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(enhanced)
    plt.title("3. Contrast Enhanced (CLAHE)\nFed to SAM 2.1", fontsize=11)
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(image)
    plt.imshow(refined_mask, alpha=0.6, cmap='plasma')
    plt.title(f"4. FINAL REFINED MASK\nStrategy: {best_result.strategy}\n"
              f"Score: {best_result.score:.3f}", 
              fontsize=11, fontweight='bold', color='green')
    plt.axis('off')
    
    # Row 2-3: Strategy results
    for idx, result in enumerate(results[:8]):
        plt.subplot(3, 4, idx + 5)
        plt.imshow(image)
        plt.imshow(result.mask, alpha=0.5, cmap='viridis')
        
        color = 'green' if result.strategy == best_result.strategy else 'black'
        marker = "★ " if result.strategy == best_result.strategy else ""
        plt.title(f"{marker}{result.strategy}\nScore: {result.score:.3f}", 
                  fontsize=9, color=color)
        plt.axis('off')
    
    plt.suptitle(f"STAGE A: Segmentation Pipeline - {view_info.view_type.upper()} View", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    test_stage_a_segmentation()
