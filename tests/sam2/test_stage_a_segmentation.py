"""
ATLAS V2 - Stage A: ROBUST Segmentation with Spatial Validation
Enhanced composite confidence with position, aspect ratio, and size validation

CRITICAL ENHANCEMENTS:
1. Spatial position penalty (broadcast views = lower region)
2. Aspect ratio validation (football fields are 1.5:1 to 2:1)
3. Size reasonableness check (30-70% of image)
4. Multi-signal view classification
5. Adaptive confidence thresholds
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
    view_type: str
    confidence: float
    field_coverage: float
    vertical_angle: float
    expected_field_region: Tuple[float, float]  # (y_min, y_max) normalized


@dataclass
class SegmentationResult:
    mask: np.ndarray
    sam_score: float
    composite_confidence: float
    strategy: str
    view_type: str
    inference_time: float
    hsv_iou: float
    rectangularity: float
    view_confidence: float
    spatial_penalty: float
    aspect_ratio_score: float
    size_score: float


class EnhancedViewClassifier:
    """Multi-signal view classification with spatial expectations"""
    
    @staticmethod
    def classify(image: np.ndarray) -> ViewClassification:
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Multiple HSV ranges for robustness
        lower_green = np.array([25, 25, 25])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        field_coverage = np.sum(green_mask > 0) / (h * w)
        
        # Analyze vertical distribution of green
        green_y_dist = np.sum(green_mask, axis=1)
        top_third = np.sum(green_y_dist[:h//3])
        middle_third = np.sum(green_y_dist[h//3:2*h//3])
        bottom_third = np.sum(green_y_dist[2*h//3:])
        
        total_green = top_third + middle_third + bottom_third
        
        # Calculate distribution percentages
        if total_green > 0:
            top_pct = top_third / total_green
            middle_pct = middle_third / total_green
            bottom_pct = bottom_third / total_green
        else:
            top_pct = middle_pct = bottom_pct = 0.33
        
        # Multi-signal classification
        if field_coverage > 0.55 and middle_pct > 0.3:
            # Aerial: High coverage, evenly distributed
            view_type = 'aerial'
            confidence = min(0.95, 0.75 + (field_coverage - 0.55) * 0.4)
            expected_region = (0.2, 0.8)  # Field can be anywhere
            
        elif bottom_pct > 0.5 and field_coverage < 0.55:
            # Broadcast: Field in lower portion, moderate coverage
            view_type = 'broadcast'
            confidence = min(0.90, 0.65 + bottom_pct * 0.25)
            expected_region = (0.35, 0.95)  # Field in lower 60%
            
        elif bottom_pct > 0.4 and middle_pct > 0.3:
            # Ground level: Field mostly in lower-middle
            view_type = 'ground'
            confidence = 0.80
            expected_region = (0.45, 0.95)  # Lower portion only
            
        else:
            # Uncertain - default to broadcast (most common)
            view_type = 'broadcast'
            confidence = 0.60
            expected_region = (0.35, 0.90)
        
        return ViewClassification(
            view_type=view_type,
            confidence=confidence,
            field_coverage=field_coverage,
            vertical_angle=90.0 if view_type == 'aerial' else (45.0 if view_type == 'broadcast' else 15.0),
            expected_field_region=expected_region
        )


class HSVPreprocessor:
    @staticmethod
    def extract_field_mask(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Wide ranges for all lighting conditions
        green_ranges = [
            ([20, 20, 20], [100, 255, 255]),
            ([25, 15, 15], [95, 255, 255]),
            ([30, 30, 30], [90, 255, 240]),
            ([35, 40, 40], [85, 255, 255]),
        ]
        
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=1)
        
        return combined_mask > 0
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, boost: bool = False) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clip_limit = 4.0 if boost else 3.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


class RobustConfidenceEvaluator:
    """ROBUST confidence with spatial, aspect ratio, and size validation"""
    
    @staticmethod
    def compute_rectangularity(mask: np.ndarray) -> float:
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        
        if area == 0:
            return 0.0
        
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        
        return min(area / rect_area, 1.0) if rect_area > 0 else 0.0
    
    @staticmethod
    def compute_hsv_iou(sam_mask: np.ndarray, hsv_mask: np.ndarray) -> float:
        intersection = np.logical_and(sam_mask, hsv_mask).sum()
        union = np.logical_or(sam_mask, hsv_mask).sum()
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_spatial_penalty(mask: np.ndarray, view_info: ViewClassification, image_shape: Tuple) -> float:
        """
        Penalize masks not in expected region for the view type
        Returns: penalty multiplier (1.0 = no penalty, 0.0 = maximum penalty)
        """
        h, w = image_shape[:2]
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find mask center
        M = cv2.moments(mask_uint8)
        if M["m00"] == 0:
            return 0.0
        
        center_y = M["m01"] / M["m00"]
        center_y_normalized = center_y / h
        
        expected_min, expected_max = view_info.expected_field_region
        
        # Aerial views: lenient on position
        if view_info.view_type == 'aerial':
            if expected_min <= center_y_normalized <= expected_max:
                return 1.0
            else:
                distance_from_range = min(
                    abs(center_y_normalized - expected_min),
                    abs(center_y_normalized - expected_max)
                )
                return max(0.7, 1.0 - distance_from_range * 2)
        
        # Broadcast/Ground views: strict on position
        else:
            if expected_min <= center_y_normalized <= expected_max:
                # Inside expected region - full score
                return 1.0
            elif center_y_normalized < expected_min:
                # Too high (upper stadium) - heavy penalty
                distance = expected_min - center_y_normalized
                return max(0.5, 1.0 - distance * 5)  # Harsh penalty for upper region
            else:
                # Too low - moderate penalty
                distance = center_y_normalized - expected_max
                return max(0.5, 1.0 - distance * 2)
    
    @staticmethod
    def compute_aspect_ratio_score(mask: np.ndarray) -> float:
        """
        Football fields are typically 1.5:1 to 2.2:1 (width:height)
        Returns: score from 0 to 1
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        if h == 0:
            return 0.0
        
        aspect_ratio = w / h
        
        # Ideal range: 1.5 to 2.2
        if 1.5 <= aspect_ratio <= 2.2:
            return 1.0
        elif 1.2 <= aspect_ratio < 1.5:
            return 0.8
        elif 2.2 < aspect_ratio <= 2.5:
            return 0.8
        elif 1.0 <= aspect_ratio < 1.2:
            return 0.6
        elif 2.5 < aspect_ratio <= 3.0:
            return 0.5
        else:
            return 0.3  # Too extreme
    
    @staticmethod
    def compute_size_score(mask: np.ndarray, image_shape: Tuple) -> float:
        """
        Field should be 30-70% of image for good segmentation
        Returns: score from 0 to 1
        """
        h, w = image_shape[:2]
        total_pixels = h * w
        mask_pixels = mask.sum()
        coverage = mask_pixels / total_pixels
        
        # Ideal range: 0.35 to 0.65
        if 0.35 <= coverage <= 0.65:
            return 1.0
        elif 0.25 <= coverage < 0.35:
            return 0.8
        elif 0.65 < coverage <= 0.75:
            return 0.8
        elif 0.20 <= coverage < 0.25:
            return 0.6
        elif 0.75 < coverage <= 0.85:
            return 0.6
        else:
            return 0.3  # Too small or too large
    
    @classmethod
    def evaluate(cls, sam_mask: np.ndarray, sam_score: float, 
                 hsv_mask: np.ndarray, view_info: ViewClassification,
                 image_shape: Tuple) -> Tuple[float, float, float, float, float, float]:
        """
        Returns: (composite_confidence, hsv_iou, rectangularity, 
                  spatial_penalty, aspect_ratio_score, size_score)
        
        Enhanced formula (reweighted for Stage A visual quality):
        base_confidence = 0.30*SAM + 0.30*HSV_IoU + 0.10*Rect + 0.10*ViewConf + 0.10*AspectRatio + 0.10*Size
        final_confidence = base_confidence * (0.5 + 0.5*spatial_penalty)  # Softened penalty
        """
        hsv_iou = cls.compute_hsv_iou(sam_mask, hsv_mask)
        rectangularity = cls.compute_rectangularity(sam_mask)
        spatial_penalty = cls.compute_spatial_penalty(sam_mask, view_info, image_shape)
        aspect_ratio_score = cls.compute_aspect_ratio_score(sam_mask)
        size_score = cls.compute_size_score(sam_mask, image_shape)
        
        # Base confidence with all components (reweighted for visual quality)
        base_confidence = (
            0.30 * sam_score +           # Increased trust in SAM
            0.30 * hsv_iou +             # Increased trust in visual overlap
            0.10 * rectangularity +      # Reduced geometry weight
            0.10 * view_info.confidence + # Reduced view weight
            0.10 * aspect_ratio_score +
            0.10 * size_score
        )
        
        # Apply softened spatial penalty (less harsh than pure multiplication)
        # Instead of: composite = base × penalty
        # Use: composite = base × (0.5 + 0.5 × penalty)
        # This way even penalty=0.5 results in 0.75x instead of 0.5x
        composite = base_confidence * (0.5 + 0.5 * spatial_penalty)
        composite = min(composite, 1.0)
        
        return composite, hsv_iou, rectangularity, spatial_penalty, aspect_ratio_score, size_score


class AdaptiveSAMStrategy:
    @staticmethod
    def get_strategies(view_type: str) -> list:
        if view_type == 'aerial':
            return ['box_strategy_wide', 'box_point_hybrid', 'multi_point_dense']
        elif view_type == 'broadcast':
            return ['broadcast_lower', 'single_center_lower', 'multi_point_lower', 'box_point_hybrid']
        elif view_type == 'ground':
            return ['lower_generous', 'single_center_lower', 'multi_point_lower']
        else:
            return ['multi_point_dense', 'box_strategy_wide', 'broadcast_lower']


class MorphologicalExpander:
    @staticmethod
    def expand_mask(mask: np.ndarray, expansion_percent: float = 0.12) -> np.ndarray:
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        expand_x = int(w * expansion_percent)
        expand_y = int(h * expansion_percent)
        
        x_new = max(0, x - expand_x)
        y_new = max(0, y - expand_y)
        w_new = min(mask.shape[1] - x_new, w + 2 * expand_x)
        h_new = min(mask.shape[0] - y_new, h + 2 * expand_y)
        
        expanded = np.zeros_like(mask_uint8)
        expanded[y_new:y_new+h_new, x_new:x_new+w_new] = 255
        
        combined = cv2.bitwise_or(mask_uint8, expanded)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return (combined > 127).astype(bool)


class MorphologicalRefiner:
    """Apply OpenCV morphological operations to smooth and refine masks"""

    @staticmethod
    def refine_mask(mask: np.ndarray, aggressive: bool = False) -> np.ndarray:
        mask_uint8 = (mask * 255).astype(np.uint8)
        if aggressive:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        else:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Optional: small erosion to remove tiny patches before closing
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.erode(mask_uint8, erode_kernel)
        
        mask_closed = cv2.morphologyEx(mask_eroded, cv2.MORPH_CLOSE, close_kernel)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, open_kernel)

        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            refined_mask = np.zeros_like(mask_uint8)
            cv2.drawContours(refined_mask, [largest_contour], -1, 255, -1)
            refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
            return (refined_mask > 127).astype(bool)

        return (mask_opened > 127).astype(bool)


def execute_strategy(strategy_name: str, predictor, image: np.ndarray, 
                     hsv_mask: np.ndarray, w: int, h: int, dtype):
    
    if strategy_name == 'single_center_lower':
        # Center point in lower-middle region for broadcast views
        cy = int(h * 0.58)
        point_coords = np.array([[w // 2, cy]])
        point_labels = np.array([1])
        
    elif strategy_name == 'multi_point_dense':
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
    
    elif strategy_name == 'multi_point_lower':
        # Points in lower 70% only
        cy = int(h * 0.60)
        margin_x = int(w * 0.15)
        point_coords = np.array([
            [margin_x, cy],
            [w // 2, cy],
            [w - margin_x, cy],
            [w // 4, cy + int(h * 0.15)],
            [3 * w // 4, cy + int(h * 0.15)],
            [w // 2, h - 80]
        ])
        point_labels = np.ones(len(point_coords))
    
    elif strategy_name == 'box_strategy_wide':
        margin_x, margin_y = int(w * 0.02), int(h * 0.02)
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
    
    elif strategy_name == 'broadcast_lower':
        # Focused on lower 70% with no upper points
        cy = int(h * 0.58)
        point_coords = np.array([
            [w // 2, cy],
            [w // 4, cy],
            [3 * w // 4, cy],
            [w // 4, cy + h // 8],
            [3 * w // 4, cy + h // 8],
            [w // 2, h - 70]
        ])
        point_labels = np.ones(len(point_coords))
    
    elif strategy_name == 'lower_generous':
        cy = int(h * 0.65)
        point_coords = np.array([
            [w // 2, cy],
            [w // 4, cy],
            [3 * w // 4, cy],
            [w // 2, cy - h // 10],
            [w // 2, h - 60]
        ])
        point_labels = np.ones(len(point_coords))
    
    else:
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
    
    with torch.inference_mode(), torch.autocast(device_type="cpu" if not torch.cuda.is_available() else "cuda", dtype=dtype):
        masks, scores, _ = predictor.predict(
            point_coords=point_coords, 
            point_labels=point_labels, 
            multimask_output=True
        )
    
    return masks, scores


def test_stage_a_segmentation():
    
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[WARNING] Running on CPU")
    
    project_root = Path(__file__).parent.parent.parent
    checkpoint = str(project_root / "atlas/models/sam2/checkpoints/sam2.1_hiera_large.pt")
    config_path = str(project_root / "atlas/models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    test_dir = project_root / "test_data/frames"
    
    today = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = project_root / f"output/sam2/stage_a/{today}"
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in test_dir.iterdir() 
                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not image_files:
        print("❌ No images found")
        return
    
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
    print("\n🎯 REBALANCED CONFIDENCE: Visual Quality Priority")
    print("Formula: (0.30*SAM + 0.30*IoU + 0.10*Rect + 0.10*View + 0.10*AR + 0.10*Size) × (0.5 + 0.5*SpatialPenalty)\n")
    
    view_classifier = EnhancedViewClassifier()
    hsv_processor = HSVPreprocessor()
    strategy_selector = AdaptiveSAMStrategy()
    expander = MorphologicalExpander()
    evaluator = RobustConfidenceEvaluator()
    
    for img_idx, image_path in enumerate(image_files):
        print(f"\n{'='*70}")
        print(f"📸 Image {img_idx + 1}/{len(image_files)}: {image_path.name}")
        print(f"{'='*70}")
        
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        
        print("\n[1/4] Enhanced view classification...")
        view_info = view_classifier.classify(image)
        print(f"  View: {view_info.view_type.upper()} (confidence: {view_info.confidence:.3f})")
        print(f"  Expected field Y-range: {view_info.expected_field_region[0]:.1%} - {view_info.expected_field_region[1]:.1%}")
        print(f"  Field coverage: {view_info.field_coverage:.1%}")
        
        print("\n[2/4] HSV preprocessing...")
        hsv_mask = hsv_processor.extract_field_mask(image)
        enhanced_image = hsv_processor.enhance_contrast(image, boost=(view_info.confidence < 0.75))
        print(f"  HSV coverage: {hsv_mask.sum()/(h*w):.1%}")
        
        print(f"\n[3/4] Testing strategies ({view_info.view_type})...")
        strategies = strategy_selector.get_strategies(view_info.view_type)
        
        predictor.set_image(enhanced_image)
        
        best_result = None
        all_results = []
        
        for strategy_name in strategies:
            start = time.time()
            
            try:
                masks, scores = execute_strategy(
                    strategy_name, predictor, enhanced_image, hsv_mask, w, h, dtype
                )
                
                # Evaluate ALL masks with robust confidence
                for mask, sam_score in zip(masks, scores):
                    composite_conf, hsv_iou, rectangularity, spatial_penalty, aspect_score, size_score = evaluator.evaluate(
                        mask, sam_score, hsv_mask, view_info, image.shape
                    )
                    
                    result = SegmentationResult(
                        mask=mask,
                        sam_score=sam_score,
                        composite_confidence=composite_conf,
                        strategy=strategy_name,
                        view_type=view_info.view_type,
                        inference_time=time.time() - start,
                        hsv_iou=hsv_iou,
                        rectangularity=rectangularity,
                        view_confidence=view_info.confidence,
                        spatial_penalty=spatial_penalty,
                        aspect_ratio_score=aspect_score,
                        size_score=size_score
                    )
                    
                    all_results.append(result)
                    
                    # Select by composite confidence
                    if best_result is None or result.composite_confidence > best_result.composite_confidence:
                        best_result = result
                
                # Print best from this strategy
                best_from_strategy = max(
                    [r for r in all_results if r.strategy == strategy_name],
                    key=lambda x: x.composite_confidence
                )
                print(f"  ✓ {strategy_name:25} | Comp: {best_from_strategy.composite_confidence:.3f} "
                      f"(Spatial: {best_from_strategy.spatial_penalty:.2f} AR: {best_from_strategy.aspect_ratio_score:.2f})")
                    
            except Exception as e:
                print(f"  ✗ {strategy_name:25} | Failed: {str(e)}")
                continue
        
        if not best_result:
            print("❌ All strategies failed")
            continue
        
        print(f"\n[4/4] Best: {best_result.strategy}")
        print(f"  🎯 Composite confidence: {best_result.composite_confidence:.3f}")
        print(f"  ├─ SAM score:        {best_result.sam_score:.3f} (0.30)")
        print(f"  ├─ HSV IoU:          {best_result.hsv_iou:.3f} (0.30)")
        print(f"  ├─ Rectangularity:   {best_result.rectangularity:.3f} (0.10)")
        print(f"  ├─ View confidence:  {best_result.view_confidence:.3f} (0.10)")
        print(f"  ├─ Aspect ratio:     {best_result.aspect_ratio_score:.3f} (0.10)")
        print(f"  ├─ Size score:       {best_result.size_score:.3f} (0.10)")
        print(f"  └─ ⚠️ Spatial penalty: {best_result.spatial_penalty:.3f} (softened: ×(0.5+0.5*penalty))")
        
        # Calculate mask center for validation
        M = cv2.moments((best_result.mask * 255).astype(np.uint8))
        if M["m00"] > 0:
            center_y_norm = (M["m01"] / M["m00"]) / h
            print(f"     Mask center Y: {center_y_norm:.1%} (expected: {view_info.expected_field_region[0]:.1%}-{view_info.expected_field_region[1]:.1%})")
        
        print(f"\n  Expanding mask (12% margin)...")
        expanded_mask = expander.expand_mask(best_result.mask, expansion_percent=0.12)
        print(f"  Before: {best_result.mask.sum():,}px → After: {expanded_mask.sum():,}px")
        
        # === [New Step] Morphological refinement and optional color fusion ===
        refiner = MorphologicalRefiner()
        
        # Conditional fusion: only for aerial views or weak SAM masks
        sam_coverage = expanded_mask.sum() / (h * w)
        if view_info.view_type == 'aerial' or sam_coverage < 0.25:
            fused_mask = np.logical_or(expanded_mask, hsv_mask)
            print(f"  HSV fusion applied (view: {view_info.view_type}, SAM coverage: {sam_coverage:.1%})")
        else:
            fused_mask = expanded_mask
            print(f"  HSV fusion skipped (view: {view_info.view_type}, SAM coverage: {sam_coverage:.1%})")
        
        # Apply smooth morphological refinement (adaptive aggressiveness)
        aggressive = view_info.view_type == 'aerial'
        final_mask = refiner.refine_mask(fused_mask, aggressive=aggressive)
        print(f"  Refined mask applied ({'aggressive' if aggressive else 'normal'} mode, {final_mask.sum():,}px)")
        
        # Save
        mask_path = output_dir / f"{image_path.stem}_stage_a_mask.npy"
        np.save(mask_path, final_mask.astype(np.uint8))
        
        save_stage_a_visualization(
            image, hsv_mask, enhanced_image, all_results[:8], 
            final_mask, best_result, view_info,
            output_dir / f"{image_path.stem}_stage_a.png"
        )
        
        print(f"  ✓ Saved: {mask_path.name}")
    
    print(f"\n{'='*70}")
    print(f"✅ Stage A complete - Robust masks ready for Stage B")
    print(f"Results: {output_dir}")
    print(f"{'='*70}")


def save_stage_a_visualization(image, hsv_mask, enhanced, results, 
                                final_mask, best_result, view_info, save_path):
    
    fig = plt.figure(figsize=(24, 14))
    
    # Calculate expected region overlay
    h, w = image.shape[:2]
    expected_overlay = np.zeros_like(image)
    y_min = int(h * view_info.expected_field_region[0])
    y_max = int(h * view_info.expected_field_region[1])
    expected_overlay[y_min:y_max, :] = [0, 100, 0]
    
    plt.subplot(3, 4, 1)
    plt.imshow(image)
    plt.imshow(expected_overlay, alpha=0.2)
    plt.title(f"1. Original + Expected Region\nView: {view_info.view_type.upper()} "
              f"(conf: {view_info.confidence:.3f})", fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(image)
    plt.imshow(hsv_mask, alpha=0.5, cmap='Greens')
    plt.title(f"2. HSV Mask\nCoverage: {view_info.field_coverage:.1%}", fontsize=11)
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(enhanced)
    plt.title("3. Enhanced (CLAHE)", fontsize=11)
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(image)
    plt.imshow(final_mask, alpha=0.6, cmap='plasma')
    plt.title(f"4. FINAL MASK\n{best_result.strategy} ({view_info.view_type})\n"
              f"Confidence: {best_result.composite_confidence:.3f}", 
              fontsize=11, fontweight='bold', color='green')
    plt.axis('off')
    
    for idx, result in enumerate(results):
        plt.subplot(3, 4, idx + 5)
        plt.imshow(image)
        plt.imshow(result.mask, alpha=0.5, cmap='viridis')
        
        color = 'green' if result.strategy == best_result.strategy and \
                           result.composite_confidence == best_result.composite_confidence else 'black'
        marker = "★ " if result.strategy == best_result.strategy and \
                        result.composite_confidence == best_result.composite_confidence else ""
        
        plt.title(f"{marker}{result.strategy}\n"
                  f"Comp: {result.composite_confidence:.3f} | Spatial: {result.spatial_penalty:.2f}\n"
                  f"AR: {result.aspect_ratio_score:.2f} | Size: {result.size_score:.2f}", 
                  fontsize=9, color=color)
        plt.axis('off')
    
    plt.suptitle(f"STAGE A: Rebalanced Confidence (Visual Quality Priority)\n"
                 f"Formula: (0.30*SAM + 0.30*IoU + 0.10*Rect + 0.10*View + 0.10*AR + 0.10*Size) × (0.5 + 0.5*SpatialPenalty)", 
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    test_stage_a_segmentation()
