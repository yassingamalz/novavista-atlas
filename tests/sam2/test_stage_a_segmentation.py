"""
ATLAS V2 – Stage A Segmentation (HEADLESS)
Adaptive confidence + improved broadcast filtering and sanity checks.

- Adaptive HSV ranges per view
- Stronger spatial penalty influence
- Tighter coverage penalty
- Contour sanity checks to reject unreasonable masks
- Headless: no interactive plots; outputs saved to disk
"""
import os
import sys
import time
import torch
import cv2
import csv
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ViewClassification:
    view_type: str
    confidence: float
    field_coverage: float
    vertical_angle: float
    expected_field_region: Tuple[float, float]


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


# -----------------------------
# View classifier
# -----------------------------
class EnhancedViewClassifier:
    """Classify aerial / broadcast / ground and give expected region."""

    @staticmethod
    def classify(image: np.ndarray) -> ViewClassification:
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 25, 25])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        field_coverage = np.sum(green_mask > 0) / (h * w)

        ydist = np.sum(green_mask, axis=1)
        total = np.sum(ydist)
        if total == 0:
            top_pct = middle_pct = bottom_pct = 1.0 / 3.0
        else:
            top = np.sum(ydist[:h // 3])
            mid = np.sum(ydist[h // 3:2 * h // 3])
            bot = np.sum(ydist[2 * h // 3:])
            top_pct = top / total
            middle_pct = mid / total
            bottom_pct = bot / total

        if field_coverage > 0.55 and middle_pct > 0.3:
            view_type = "aerial"
            confidence = min(0.95, 0.75 + (field_coverage - 0.55) * 0.4)
            expected_region = (0.2, 0.8)
        elif bottom_pct > 0.5 and field_coverage < 0.55:
            view_type = "broadcast"
            confidence = min(0.90, 0.65 + bottom_pct * 0.25)
            expected_region = (0.35, 0.95)
        elif bottom_pct > 0.4 and middle_pct > 0.3:
            view_type = "ground"
            confidence = 0.80
            expected_region = (0.45, 0.95)
        else:
            view_type = "broadcast"
            confidence = 0.60
            expected_region = (0.35, 0.90)

        vertical_angle = 90.0 if view_type == "aerial" else (45.0 if view_type == "broadcast" else 15.0)
        return ViewClassification(view_type, confidence, field_coverage, vertical_angle, expected_region)


# -----------------------------
# HSV Preprocessor (adaptive)
# -----------------------------
class HSVPreprocessor:
    """Extract field-like HSV masks with optional adaptive ranges depending on view."""

    @staticmethod
    def extract_field_mask(image: np.ndarray, view_info: ViewClassification = None) -> np.ndarray:
        """
        Returns boolean mask where True indicates green/field-like pixels.
        If view_info is provided, use more conservative ranges for 'broadcast'.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Default broad ranges to be robust across conditions
        default_ranges = [
            ([20, 20, 20], [100, 255, 255]),
            ([25, 15, 15], [95, 255, 255]),
            ([30, 30, 30], [90, 255, 240]),
            ([35, 40, 40], [85, 255, 255]),
        ]

        # Narrower/focused ranges for broadcast views to avoid structure/stands
        broadcast_ranges = [
            ([35, 60, 40], [85, 255, 255]),  # tighter hue and saturation
            ([30, 50, 30], [90, 255, 230]),
        ]

        # Aerial can use slightly wider ranges (top-down)
        aerial_ranges = [
            ([20, 20, 30], [100, 255, 255]),
            ([25, 30, 30], [95, 255, 255]),
        ]

        if view_info is not None:
            if view_info.view_type == "broadcast":
                ranges = broadcast_ranges + default_ranges[:1]
            elif view_info.view_type == "aerial":
                ranges = aerial_ranges + default_ranges[:2]
            else:
                ranges = default_ranges
        else:
            ranges = default_ranges

        combined = np.zeros(image.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            combined |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        # Morphological smoothing and dilation to reduce speckle
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)
        combined = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)

        return combined > 0

    @staticmethod
    def enhance_contrast(image: np.ndarray, boost: bool = False) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clip_limit = 4.0 if boost else 3.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)


# -----------------------------
# Robust Confidence Evaluator (adaptive)
# -----------------------------
class RobustConfidenceEvaluator:
    """Adaptive confidence evaluator with stronger spatial and coverage penalties."""

    LOG_PATH = Path("mask_quality_log.csv")

    @staticmethod
    def _ensure_log_header():
        if not RobustConfidenceEvaluator.LOG_PATH.exists():
            RobustConfidenceEvaluator.LOG_PATH.write_text(
                "image,strategy,sam_score,hsv_iou,rectangularity,view_confidence,"
                "aspect_ratio_score,size_score,spatial_penalty,sam_coverage,hsv_coverage,composite\n"
            )

    @staticmethod
    def _adaptive_weights(view_info: ViewClassification, hsv_coverage: float, sam_coverage: float):
        """Return normalized weights dictionary for metrics S,I,R,V,A,Z."""
        vt = (view_info.view_type if view_info is not None else "unknown")
        vconf = float(view_info.confidence) if view_info is not None else 0.6

        if vt == "aerial" and vconf >= 0.75:
            w = {'S': 0.22, 'I': 0.30, 'R': 0.12, 'V': 0.06, 'A': 0.18, 'Z': 0.12}
        elif vt == "broadcast":
            w = {'S': 0.38, 'I': 0.25, 'R': 0.00, 'V': 0.08, 'A': 0.05, 'Z': 0.24}
        else:
            w = {'S': 0.40, 'I': 0.20, 'R': 0.00, 'V': 0.10, 'A': 0.05, 'Z': 0.25}

        # Context adaptations
        if hsv_coverage < 0.10:
            w['I'] *= 0.5
            w['S'] += 0.10
        if sam_coverage < 0.08:
            if hsv_coverage > 0.30:
                w['I'] += 0.15; w['S'] -= 0.10
            else:
                w['Z'] += 0.10; w['V'] += 0.05; w['S'] -= 0.05
        if vconf < 0.6:
            w['V'] *= 0.4

        total = sum(w.values()) if sum(w.values()) > 0 else 1.0
        for k in w:
            w[k] = max(0.0, w[k] / total)
        return w

    @staticmethod
    def _coverage_penalty(coverage: float):
        """Tighter penalty: ideal coverage [0.30, 0.65] for Stage A precise selection."""
        if 0.30 <= coverage <= 0.65:
            pen = 1.0
        elif coverage > 0.65:
            pen = 1.0 - 3.0 * (coverage - 0.65)
        else:
            pen = 1.0 - 3.0 * (0.30 - coverage)
        return float(np.clip(pen, 0.2, 1.0))

    # Metric computations (kept accurate)
    @staticmethod
    def compute_hsv_iou(sam_mask: np.ndarray, hsv_mask: np.ndarray) -> float:
        inter = np.logical_and(sam_mask, hsv_mask).sum()
        union = np.logical_or(sam_mask, hsv_mask).sum()
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def compute_rectangularity(mask: np.ndarray) -> float:
        mask_u8 = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        return float(area / rect_area) if rect_area > 0 else 0.0

    @staticmethod
    def compute_aspect_ratio_score(mask: np.ndarray) -> float:
        mask_u8 = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        cnt = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            return 0.0
        aspect_ratio = w / h
        # Score: 1.0 if near 1.6–2.0, decay smoothly otherwise
        ideal = 1.8
        diff = abs(aspect_ratio - ideal)
        return float(max(0.0, 1.0 - (diff / 2.0)))  # clipped

    @staticmethod
    def compute_size_score(mask: np.ndarray, image_shape: Tuple[int, int]) -> float:
        h, w = image_shape[:2]
        coverage = mask.sum() / (h * w)
        if coverage < 0.3:
            return float(coverage / 0.3)
        if coverage > 0.65:
            return float(max(0.0, 1 - (coverage - 0.65) / 0.35))
        return 1.0

    @staticmethod
    def compute_spatial_penalty(mask: np.ndarray, view_info: ViewClassification, image_shape: Tuple[int, int]) -> float:
        h = image_shape[0]
        mask_u8 = (mask * 255).astype(np.uint8)
        M = cv2.moments(mask_u8)
        if M["m00"] == 0:
            return 0.3  # weak mask
        center_y = (M["m01"] / M["m00"]) / float(h)
        expected_min, expected_max = view_info.expected_field_region
        if expected_min <= center_y <= expected_max:
            return 1.0
        # harsher penalty for being far from expected region
        distance = min(abs(center_y - expected_min), abs(center_y - expected_max))
        return float(max(0.2, 1.0 - distance * 4.0))

    @staticmethod
    def evaluate(mask: np.ndarray, sam_score: float, hsv_mask: np.ndarray,
                 view_info: ViewClassification, image_shape: Tuple[int, int],
                 image_name: str = "<unknown>", strategy: str = "<unknown>"):
        """Return composite and factors: composite, hsv_iou, rect, spatial, ar, size"""
        hsv_iou = RobustConfidenceEvaluator.compute_hsv_iou(mask, hsv_mask)
        rect = RobustConfidenceEvaluator.compute_rectangularity(mask)
        ar = RobustConfidenceEvaluator.compute_aspect_ratio_score(mask)
        size = RobustConfidenceEvaluator.compute_size_score(mask, image_shape)
        spatial = RobustConfidenceEvaluator.compute_spatial_penalty(mask, view_info, image_shape)

        h, w = image_shape[:2]
        sam_cov = float(mask.sum() / (h * w))
        hsv_cov = float(view_info.field_coverage if view_info is not None else 0.0)
        wts = RobustConfidenceEvaluator._adaptive_weights(view_info, hsv_cov, sam_cov)

        base = (wts['S'] * sam_score +
                wts['I'] * hsv_iou +
                wts['R'] * rect +
                wts['V'] * float(view_info.confidence) +
                wts['A'] * ar +
                wts['Z'] * size)

        # stronger spatial influence (softened)
        composite = base * (0.3 + 0.7 * spatial) * RobustConfidenceEvaluator._coverage_penalty(sam_cov)
        composite = float(max(0.0, min(composite, 1.0)))

        # log for training/analysis
        RobustConfidenceEvaluator._ensure_log_header()
        row = ",".join(map(str, [
            image_name, strategy, float(sam_score), float(hsv_iou), float(rect),
            float(view_info.confidence), float(ar), float(size), float(spatial),
            sam_cov, hsv_cov, composite
        ]))
        with open(RobustConfidenceEvaluator.LOG_PATH, "a", newline="") as f:
            f.write(row + "\n")

        return composite, hsv_iou, rect, spatial, ar, size
# -----------------------------
# Adaptive SAM strategy selection
# -----------------------------
class AdaptiveSAMStrategy:
    @staticmethod
    def get_strategies(view_type: str) -> list:
        if view_type == "aerial":
            return ["box_strategy_wide", "box_point_hybrid", "multi_point_dense"]
        elif view_type == "broadcast":
            return ["broadcast_lower", "single_center_lower", "multi_point_lower", "box_point_hybrid"]
        elif view_type == "ground":
            return ["lower_generous", "single_center_lower", "multi_point_lower"]
        return ["multi_point_dense", "box_strategy_wide", "broadcast_lower"]


# -----------------------------
# Morphological helpers
# -----------------------------
class MorphologicalExpander:
    @staticmethod
    def expand_mask(mask: np.ndarray, expansion_percent: float = 0.12) -> np.ndarray:
        mask_u8 = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask
        largest = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        ex = int(w * expansion_percent)
        ey = int(h * expansion_percent)
        x0 = max(0, x - ex); y0 = max(0, y - ey)
        w2 = min(mask.shape[1] - x0, w + 2 * ex); h2 = min(mask.shape[0] - y0, h + 2 * ey)
        expanded = np.zeros_like(mask_u8)
        expanded[y0:y0 + h2, x0:x0 + w2] = 255
        merged = cv2.bitwise_or(mask_u8, expanded)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
        return merged > 127


class MorphologicalRefiner:
    @staticmethod
    def refine_mask(mask: np.ndarray, aggressive: bool = False) -> np.ndarray:
        mask_u8 = (mask * 255).astype(np.uint8)
        if aggressive:
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        else:
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.erode(mask_u8, erode_k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            out = np.zeros_like(mask_u8)
            cv2.drawContours(out, [largest], -1, 255, -1)
            out = cv2.GaussianBlur(out, (5, 5), 0)
            return out > 127
        return m > 127


# -----------------------------
# Strategy execution wrapper
# -----------------------------
def execute_strategy(strategy_name: str, predictor, image: np.ndarray,
                     hsv_mask: np.ndarray, w: int, h: int, dtype):
    if strategy_name == "single_center_lower":
        cy = int(h * 0.58)
        pts = np.array([[w // 2, cy]])
        lbl = np.array([1])
    elif strategy_name == "multi_point_dense":
        mx, my = int(w * 0.15), int(h * 0.15)
        pts = np.array([
            [mx, my], [w // 2, my], [w - mx, my],
            [mx, h // 2], [w // 2, h // 2], [w - mx, h // 2],
            [mx, h - my], [w // 2, h - my], [w - mx, h - my]
        ])
        lbl = np.ones(len(pts))
    elif strategy_name == "multi_point_lower":
        cy, mx = int(h * 0.60), int(w * 0.15)
        pts = np.array([[mx, cy], [w // 2, cy], [w - mx, cy],
                        [w // 4, cy + int(h * 0.15)], [3 * w // 4, cy + int(h * 0.15)], [w // 2, h - 80]])
        lbl = np.ones(len(pts))
    elif strategy_name == "box_strategy_wide":
        box = np.array([int(w * 0.02), int(h * 0.02), int(w * 0.98), int(h * 0.98)])
        with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        return masks, scores
    elif strategy_name == "box_point_hybrid":
        box = np.array([int(w * 0.03), int(h * 0.03), int(w * 0.97), int(h * 0.97)])
        pts = np.array([[w // 2, h // 2]])
        lbl = np.array([1])
        with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
            masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl, box=box, multimask_output=True)
        return masks, scores
    elif strategy_name == "broadcast_lower":
        cy = int(h * 0.58)
        pts = np.array([[w // 2, cy], [w // 4, cy], [3 * w // 4, cy],
                        [w // 4, cy + h // 8], [3 * w // 4, cy + h // 8], [w // 2, h - 70]])
        lbl = np.ones(len(pts))
    elif strategy_name == "lower_generous":
        cy = int(h * 0.65)
        pts = np.array([[w // 2, cy], [w // 4, cy], [3 * w // 4, cy],
                        [w // 2, cy - h // 10], [w // 2, h - 60]])
        lbl = np.ones(len(pts))
    else:
        pts = np.array([[w // 2, h // 2]])
        lbl = np.array([1])

    with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
        masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
    return masks, scores


# -----------------------------
# Visualization saver (headless)
# -----------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_stage_a_visualization(image, hsv_mask, enhanced, results, final_mask, best_result, view_info, save_path):
    fig = plt.figure(figsize=(22, 14))
    h, w = image.shape[:2]
    expected = np.zeros_like(image)
    y0 = int(h * view_info.expected_field_region[0])
    y1 = int(h * view_info.expected_field_region[1])
    expected[y0:y1, :] = [0, 100, 0]

    # Top row: input, hsv, enhanced, final
    ax = plt.subplot(3, 4, 1); ax.imshow(image); ax.imshow(expected, alpha=0.25); ax.axis("off"); ax.set_title("Original + Expected")
    ax = plt.subplot(3, 4, 2); ax.imshow(image); ax.imshow(hsv_mask, alpha=0.7, cmap="Greens"); ax.axis("off"); ax.set_title("HSV Mask")
    ax = plt.subplot(3, 4, 3); ax.imshow(enhanced); ax.axis("off"); ax.set_title("CLAHE Enhanced")
    ax = plt.subplot(3, 4, 4); ax.imshow(image); ax.imshow(final_mask, alpha=0.6, cmap="plasma"); ax.axis("off")
    ax.set_title(f"Final Mask ({best_result.strategy})", color="green")

    for idx, r in enumerate(results[:8]):
        ax = plt.subplot(3, 4, idx + 5)
        ax.imshow(image); ax.imshow(r.mask, alpha=0.5, cmap="viridis")
        mark = "★ " if r.strategy == best_result.strategy else ""
        ax.set_title(f"{mark}{r.strategy}\nComp={r.composite_confidence:.3f}", fontsize=9)
        ax.axis("off")

    plt.suptitle("STAGE A: Adaptive Confidence Evaluation", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main pipeline
# -----------------------------
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

    image_files = [f for f in test_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    if not image_files:
        print("❌ No images found in", test_dir)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print("[INFO] Loading SAM 2.1 model...")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    try:
        model = build_sam2(config_path, checkpoint, device=device)
    except TypeError:
        model = build_sam2(config_path, checkpoint)
        model = model.to(device)

    predictor = SAM2ImagePredictor(model)
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    view_classifier = EnhancedViewClassifier()
    hsv_processor = HSVPreprocessor()
    strategy_selector = AdaptiveSAMStrategy()
    expander = MorphologicalExpander()
    evaluator = RobustConfidenceEvaluator()
    refiner = MorphologicalRefiner()

    for img_idx, image_path in enumerate(image_files, start=1):
        print(f"\n{'='*80}\n📸 Image {img_idx}/{len(image_files)}: {image_path.name}\n{'='*80}")
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]

        # 1) view classification
        view_info = view_classifier.classify(image)
        print(f"  View: {view_info.view_type.upper()} (conf: {view_info.confidence:.3f}) "
              f"Expected Y-range: {view_info.expected_field_region[0]:.1%} - {view_info.expected_field_region[1]:.1%}")
        print(f"  Field coverage (HSV raw pre): {view_info.field_coverage:.1%}")

        # 2) HSV preprocessing (adaptive)
        hsv_mask = hsv_processor.extract_field_mask(image, view_info=view_info)
        enhanced_image = hsv_processor.enhance_contrast(image, boost=(view_info.confidence < 0.75))
        print(f"  HSV mask coverage: {hsv_mask.sum()/(h*w):.1%}")

        # 3) strategies
        predictor.set_image(enhanced_image)
        strategies = strategy_selector.get_strategies(view_info.view_type)

        best_result = None
        all_results = []

        for strategy_name in strategies:
            start_t = time.time()
            try:
                masks, scores = execute_strategy(strategy_name, predictor, enhanced_image, hsv_mask, w, h, dtype)
                # evaluate all returned masks
                for mask, sam_score in zip(masks, scores):
                    composite_conf, hsv_iou, rectangularity, spatial_penalty, aspect_score, size_score = evaluator.evaluate(
                        mask.astype(bool), float(sam_score), hsv_mask, view_info, image.shape, image_path.name, strategy_name
                    )
                    res = SegmentationResult(
                        mask=mask.astype(bool),
                        sam_score=float(sam_score),
                        composite_confidence=float(composite_conf),
                        strategy=strategy_name,
                        view_type=view_info.view_type,
                        inference_time=time.time() - start_t,
                        hsv_iou=float(hsv_iou),
                        rectangularity=float(rectangularity),
                        view_confidence=view_info.confidence,
                        spatial_penalty=float(spatial_penalty),
                        aspect_ratio_score=float(aspect_score),
                        size_score=float(size_score)
                    )
                    all_results.append(res)
                    if best_result is None or res.composite_confidence > best_result.composite_confidence:
                        best_result = res

                # debug per-strategy best
                candidates = [r for r in all_results if r.strategy == strategy_name]
                if candidates:
                    best_s = max(candidates, key=lambda x: x.composite_confidence)
                    print(f"  ✓ {strategy_name:25} | Comp={best_s.composite_confidence:.3f} Spatial={best_s.spatial_penalty:.2f} AR={best_s.aspect_ratio_score:.2f} Size={best_s.size_score:.2f}")
                else:
                    print(f"  - {strategy_name:25} | no candidates")
            except Exception as e:
                print(f"  ✗ {strategy_name:25} | Error: {e}")
                continue

        if not best_result:
            print("❌ No valid strategy produced a mask.")
            continue

        print(f"\n[SELECTED] {best_result.strategy} | Composite={best_result.composite_confidence:.3f}")
        
       # keep SAM’s original shape as the base; avoid HSV fusion except for aerial
        if view_info.view_type == "aerial":
            final_mask = expander.expand_mask(best_result.mask, expansion_percent=0.05)
            final_mask = refiner.refine_mask(final_mask, aggressive=True)
        else:
            # non-aerial: use raw SAM mask with light smoothing only
            final_mask = refiner.refine_mask(best_result.mask, aggressive=False)

        sam_coverage = float(final_mask.sum()) / (h * w)
        print(f"  Final refined SAM mask coverage: {sam_coverage:.2%}")


        # Sanity contour checks: reject obviously wrong sizes
        final_area_frac = float(final_mask.sum()) / (h * w)
        if final_area_frac < 0.15 or final_area_frac > 0.85:
            print(f"  ⚠️ Rejecting final mask for unreasonable coverage: {final_area_frac:.2%}")
            # Save debug images and move to next image
            dbg_path = output_dir / f"{image_path.stem}_rejected_debug.png"
            save_stage_a_visualization(image, hsv_mask, enhanced_image, all_results[:8], final_mask, best_result, view_info, dbg_path)
            print(f"  → Saved debug visualization: {dbg_path.name}")
            continue

        # Save mask and visualization
        mask_path = output_dir / f"{image_path.stem}_stage_a_mask.npy"
        np.save(mask_path, final_mask.astype(np.uint8))
        viz_path = output_dir / f"{image_path.stem}_stage_a.png"
        save_stage_a_visualization(image, hsv_mask, enhanced_image, all_results[:8], final_mask, best_result, view_info, viz_path)

        print(f"  ✓ Saved mask: {mask_path.name}")
        print(f"  ✓ Saved visualization: {viz_path.name}")

    print("\n" + "=" * 80)
    print(f"✅ Stage A complete - outputs written to: {output_dir}")
    print("=" * 80)


# -----------------------------
# Entrypoint (headless)
# -----------------------------
if __name__ == "__main__":
    # Run the full batch pipeline (headless)
    test_stage_a_segmentation()
