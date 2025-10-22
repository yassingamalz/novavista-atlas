"""
SAM2 Enhanced Pipeline Test - Visual Comparison Only
Tests different prompting strategies with visual output only
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
from typing import Tuple, Dict, List
from dataclasses import dataclass

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    

@dataclass
class SegmentationResult:
    """Store segmentation results with metrics"""
    mask: np.ndarray
    score: float
    strategy: str
    inference_time: float
    area: int
    boundary_smoothness: float
    color_consistency: float
    quality_grade: str


class ImageAnalyzer:
    """Analyze image characteristics to choose optimal strategy"""
    
    @staticmethod
    def analyze(image: np.ndarray) -> Dict:
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        field_ratio = np.sum(green_mask > 0) / (h * w)
        is_aerial = field_ratio > 0.65
        brightness = np.mean(image)
        
        return {
            'width': w,
            'height': h,
            'field_ratio': field_ratio,
            'is_aerial': is_aerial,
            'brightness': brightness
        }


class AdaptiveStrategySelector:
    """Select best SAM2 strategy based on image characteristics"""
    
    @staticmethod
    def select_strategy(image_info: Dict) -> List[str]:
        strategies = []
        
        if image_info['is_aerial']:
            strategies = ['box_point_hybrid', 'box_strategy', 'multi_point_grid', 'single_center']
        else:
            strategies = ['single_center', 'pos_neg_points', 'multi_point_grid', 'box_point_hybrid']
        
        if image_info['brightness'] < 100:
            strategies.insert(0, 'multi_point_adaptive')
        
        return strategies


class ColorBasedValidator:
    """Validate and refine segmentation using color information"""
    
    @staticmethod
    def extract_field_mask(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        masks = []
        green_ranges = [
            ([35, 40, 40], [85, 255, 255]),
            ([25, 30, 30], [95, 255, 255]),
            ([30, 50, 50], [80, 255, 200])
        ]
        
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            masks.append(mask)
        
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return combined_mask > 0
    
    @staticmethod
    def validate_mask(sam_mask: np.ndarray, color_mask: np.ndarray) -> float:
        intersection = np.logical_and(sam_mask, color_mask).sum()
        union = np.logical_or(sam_mask, color_mask).sum()
        iou = intersection / union if union > 0 else 0.0
        return iou


class MorphologicalRefiner:
    """Apply OpenCV morphological operations to refine masks"""
    
    @staticmethod
    def refine_mask(mask: np.ndarray, aggressive: bool = False) -> np.ndarray:
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        if aggressive:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        else:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, open_kernel)
        
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            refined_mask = np.zeros_like(mask_uint8)
            cv2.drawContours(refined_mask, [largest_contour], -1, 255, -1)
            refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
            refined_mask = (refined_mask > 127).astype(np.uint8)
            return refined_mask.astype(bool)
        
        return (mask_opened > 127).astype(bool)
    
    @staticmethod
    def calculate_boundary_smoothness(mask: np.ndarray) -> float:
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        
        if perimeter > 0:
            smoothness = (4 * np.pi * area) / (perimeter ** 2)
            return min(smoothness, 1.0)
        return 0.0


class QualityAssessor:
    """Assess segmentation quality and assign grades"""
    
    @staticmethod
    def assess(mask: np.ndarray, score: float, color_consistency: float, 
               boundary_smoothness: float, image_shape: Tuple) -> str:
        
        area_ratio = mask.sum() / (image_shape[0] * image_shape[1])
        quality_score = 0
        
        quality_score += score * 40
        quality_score += color_consistency * 25
        quality_score += boundary_smoothness * 20
        
        if 0.3 < area_ratio < 0.8:
            quality_score += 15
        elif 0.2 < area_ratio < 0.9:
            quality_score += 10
        else:
            quality_score += 5
        
        if quality_score >= 90:
            return "A+"
        elif quality_score >= 85:
            return "A"
        elif quality_score >= 80:
            return "A-"
        elif quality_score >= 75:
            return "B+"
        elif quality_score >= 70:
            return "B"
        elif quality_score >= 65:
            return "C"
        elif quality_score >= 60:
            return "D"
        else:
            return "F"


def test_sam2_full_pipeline():
    """Enhanced SAM2 testing - Visual comparison only"""
    
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[WARNING] Running on CPU mode")
    
    project_root = Path(__file__).parent.parent.parent
    checkpoint = str(project_root / "atlas/models/sam2/checkpoints/sam2.1_hiera_large.pt")
    config_path = str(project_root / "atlas/models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    test_dir = project_root / "test_data/frames"
    
    today = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = project_root / f"output/sam2/results/{today}"
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in test_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("❌ No images found in test_data/frames/")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device.upper()}")
    print("[INFO] Loading SAM2 model...")
    
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
    
    analyzer = ImageAnalyzer()
    strategy_selector = AdaptiveStrategySelector()
    color_validator = ColorBasedValidator()
    refiner = MorphologicalRefiner()
    assessor = QualityAssessor()
    
    for img_idx, image_path in enumerate(image_files):
        print(f"\n{'='*70}")
        print(f"📸 Image {img_idx + 1}/{len(image_files)}: {image_path.name}")
        print(f"{'='*70}")
        
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        
        print("\n[1/4] Analyzing image...")
        image_info = analyzer.analyze(image)
        print(f"  View type: {'Aerial' if image_info['is_aerial'] else 'Stadium/Angled'}")
        
        print("\n[2/4] Extracting HSV baseline...")
        color_mask = color_validator.extract_field_mask(image)
        
        print("\n[3/4] Testing strategies...")
        strategies = strategy_selector.select_strategy(image_info)
        
        predictor.set_image(image)
        
        best_result = None
        strategy_results = []
        
        for strategy_name in strategies:
            start = time.time()
            
            try:
                masks, scores, meta = execute_strategy(strategy_name, predictor, image, w, h, dtype)
                
                areas = [mask.sum() for mask in masks]
                best_idx = int(np.argmax([s * a for s, a in zip(scores, areas)]))
                raw_mask = masks[best_idx]
                raw_score = scores[best_idx]
                
                refined_mask = refiner.refine_mask(raw_mask)
                color_consistency = color_validator.validate_mask(refined_mask, color_mask)
                boundary_smoothness = refiner.calculate_boundary_smoothness(refined_mask)
                
                quality_grade = assessor.assess(
                    refined_mask, raw_score, color_consistency, 
                    boundary_smoothness, image.shape[:2]
                )
                
                result = SegmentationResult(
                    mask=refined_mask,
                    score=raw_score,
                    strategy=strategy_name,
                    inference_time=time.time() - start,
                    area=int(refined_mask.sum()),
                    boundary_smoothness=boundary_smoothness,
                    color_consistency=color_consistency,
                    quality_grade=quality_grade
                )
                
                strategy_results.append(result)
                
                print(f"  ✓ {strategy_name:25} | Score: {raw_score:.3f} | Grade: {quality_grade}")
                
                if best_result is None or result.score > best_result.score:
                    best_result = result
                    
            except Exception as e:
                print(f"  ✗ {strategy_name:25} | Failed: {str(e)}")
                continue
        
        if best_result:
            print(f"\n[4/4] Best: {best_result.strategy} (Grade {best_result.quality_grade})")
            
            # Apply fusion
            sam_bool = best_result.mask.astype(bool)
            color_bool = color_mask.astype(bool)
            
            fused_and = np.logical_and(sam_bool, color_bool)
            fused_or = np.logical_or(sam_bool, color_bool)
            
            refined_and = MorphologicalRefiner.refine_mask(fused_and, aggressive=True)
            refined_or = MorphologicalRefiner.refine_mask(fused_or, aggressive=True)
            
            iou_and = ColorBasedValidator.validate_mask(refined_and, color_bool)
            iou_or = ColorBasedValidator.validate_mask(refined_or, color_bool)
            
            if max(iou_and, iou_or) > best_result.color_consistency:
                if iou_and >= iou_or:
                    best_result.mask = refined_and
                    best_result.color_consistency = float(iou_and)
                    fuse_type = "AND"
                else:
                    best_result.mask = refined_or
                    best_result.color_consistency = float(iou_or)
                    fuse_type = "OR"
                
                best_result.boundary_smoothness = MorphologicalRefiner.calculate_boundary_smoothness(best_result.mask)
                best_result.quality_grade = QualityAssessor.assess(
                    best_result.mask, best_result.score,
                    best_result.color_consistency,
                    best_result.boundary_smoothness,
                    image.shape[:2]
                )
                print(f"  Fusion ({fuse_type}) → IoU: {best_result.color_consistency:.3f}")
            
            save_enhanced_visualization(
                image, color_mask, strategy_results, best_result,
                output_dir / f"{image_path.stem}_comparison.png"
            )

    print(f"\n✅ Done! Results: {output_dir}")


def execute_strategy(strategy_name: str, predictor, image, w, h, dtype):
    strategies_map = {
        'single_center': lambda: single_point_strategy(predictor, image, w, h, dtype),
        'multi_point_grid': lambda: multi_point_strategy(predictor, image, w, h, dtype),
        'pos_neg_points': lambda: pos_neg_strategy(predictor, image, w, h, dtype),
        'box_strategy': lambda: box_strategy(predictor, image, w, h, dtype),
        'box_point_hybrid': lambda: box_point_strategy(predictor, image, w, h, dtype),
        'multi_point_adaptive': lambda: adaptive_multi_point(predictor, image, w, h, dtype)
    }
    return strategies_map[strategy_name]()


def single_point_strategy(predictor, image, w, h, dtype):
    point_coords = np.array([[w // 2, h // 2]])
    point_labels = np.array([1])
    with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
        masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
    return masks, scores, {"points": point_coords}


def multi_point_strategy(predictor, image, w, h, dtype):
    margin_x, margin_y = int(w * 0.2), int(h * 0.2)
    point_coords = np.array([
        [margin_x, margin_y],
        [w - margin_x, margin_y],
        [w - margin_x, h - margin_y],
        [margin_x, h - margin_y],
        [w // 2, h // 2]
    ])
    point_labels = np.ones(len(point_coords))
    with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
        masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
    return masks, scores, {"points": point_coords}


def pos_neg_strategy(predictor, image, w, h, dtype):
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
    with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
        masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
    return masks, scores, {"points": point_coords}


def box_strategy(predictor, image, w, h, dtype):
    margin_x, margin_y = int(w * 0.05), int(h * 0.05)
    box = np.array([margin_x, margin_y, w - margin_x, h - margin_y])
    with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    return masks, scores, {"box": box}


def box_point_strategy(predictor, image, w, h, dtype):
    margin_x, margin_y = int(w * 0.05), int(h * 0.05)
    box = np.array([margin_x, margin_y, w - margin_x, h - margin_y])
    point_coords = np.array([[w // 2, h // 2]])
    point_labels = np.array([1])
    with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
        masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box, multimask_output=True)
    return masks, scores, {"points": point_coords, "box": box}


def adaptive_multi_point(predictor, image, w, h, dtype):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            offset = min(w, h) // 8
            point_coords = np.array([
                [cx, cy],
                [cx - offset, cy],
                [cx + offset, cy],
                [cx, cy - offset],
                [cx, cy + offset]
            ])
        else:
            point_coords = np.array([[w // 2, h // 2]])
    else:
        point_coords = np.array([[w // 2, h // 2]])
    
    point_labels = np.ones(len(point_coords))
    with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
        masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
    return masks, scores, {"points": point_coords}


def save_enhanced_visualization(image, color_mask, strategy_results, best_result, save_path):
    """Create visual comparison grid"""
    n_strategies = len(strategy_results)
    fig = plt.figure(figsize=(20, 4 * ((n_strategies + 2) // 3)))
    
    plt.subplot(3, 3, 1)
    plt.imshow(image)
    plt.title("Original Image", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(image)
    plt.imshow(color_mask, alpha=0.5, cmap='Greens')
    plt.title("HSV Color Baseline", fontsize=12)
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.imshow(image)
    plt.imshow(best_result.mask, alpha=0.5, cmap='plasma')
    plt.title(f"BEST: {best_result.strategy}\nGrade: {best_result.quality_grade} | Score: {best_result.score:.3f}",
              fontsize=12, fontweight='bold', color='green')
    plt.axis('off')
    
    for idx, result in enumerate(strategy_results[:6]):
        plt.subplot(3, 3, idx + 4)
        plt.imshow(image)
        plt.imshow(result.mask, alpha=0.45, cmap='viridis')
        
        title = f"{result.strategy}\n"
        title += f"Grade: {result.quality_grade} | SAM: {result.score:.2f}\n"
        title += f"Color IoU: {result.color_consistency:.2f} | Smooth: {result.boundary_smoothness:.2f}"
        
        color = 'green' if result.strategy == best_result.strategy else 'black'
        plt.title(title, fontsize=9, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    test_sam2_full_pipeline()
