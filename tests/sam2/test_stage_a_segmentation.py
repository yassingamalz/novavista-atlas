"""
ATLAS V2 – Stage A Segmentation with Adaptive Confidence
---------------------------------------------------------
Enhanced composite confidence with adaptive, view-aware weighting
and visual-coverage penalty.

CRITICAL ENHANCEMENTS
1. Adaptive weighting by view type + confidence
2. Conditional metric weighting (Rect/AR/Size)
3. Spatial-position penalty (broadcast = lower region)
4. Coverage penalty (too small / too large)
5. Automatic CSV logging for ML regressor training
6. Visualization helper for adaptive weights
"""
import os, sys, time, torch, cv2, csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# === DATA STRUCTURES =========================================================
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


# === VIEW CLASSIFIER =========================================================
class EnhancedViewClassifier:
    """Classify image view type (aerial / broadcast / ground)."""

    @staticmethod
    def classify(image: np.ndarray) -> ViewClassification:
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_green, upper_green = np.array([25, 25, 25]), np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        field_cov = np.sum(green_mask > 0) / (h * w)

        ydist = np.sum(green_mask, axis=1)
        thirds = np.array_split(ydist, 3)
        if np.sum(ydist) == 0:
            top, mid, bot = 0.33, 0.33, 0.34
        else:
            top, mid, bot = [np.sum(t) / np.sum(ydist) for t in thirds]

        if field_cov > 0.55 and mid > 0.3:
            vt, conf, region = "aerial", min(0.95, 0.75 + (field_cov - 0.55) * 0.4), (0.2, 0.8)
        elif bot > 0.5 and field_cov < 0.55:
            vt, conf, region = "broadcast", min(0.9, 0.65 + bot * 0.25), (0.35, 0.95)
        elif bot > 0.4 and mid > 0.3:
            vt, conf, region = "ground", 0.8, (0.45, 0.95)
        else:
            vt, conf, region = "broadcast", 0.6, (0.35, 0.9)

        angle = 90 if vt == "aerial" else (45 if vt == "broadcast" else 15)
        return ViewClassification(vt, conf, field_cov, angle, region)


# === HSV PREPROCESSOR ========================================================
class HSVPreprocessor:
    """Extract and enhance field-colored regions."""

    @staticmethod
    def extract_field_mask(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ranges = [
            ([20, 20, 20], [100, 255, 255]),
            ([25, 15, 15], [95, 255, 255]),
            ([30, 30, 30], [90, 255, 240]),
            ([35, 40, 40], [85, 255, 255]),
        ]
        mask = np.zeros(image.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), 2
        )
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), 1)
        return mask > 0

    @staticmethod
    def enhance_contrast(image: np.ndarray, boost=False) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0 if boost else 3.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)


# === ROBUST CONFIDENCE EVALUATOR (Adaptive) =================================
class RobustConfidenceEvaluator:
    """Adaptive, view-aware composite confidence computation."""

    LOG_PATH = Path("mask_quality_log.csv")

    # --- ensure CSV header ----------------------------------------------------
    @staticmethod
    def _ensure_log_header():
        if not RobustConfidenceEvaluator.LOG_PATH.exists():
            RobustConfidenceEvaluator.LOG_PATH.write_text(
                "image,strategy,sam_score,hsv_iou,rectangularity,view_confidence,"
                "aspect_ratio_score,size_score,spatial_penalty,sam_coverage,"
                "hsv_coverage,composite\n"
            )

    # --- adaptive metric weights ---------------------------------------------
    @staticmethod
    def _adaptive_weights(view_info, hsv_cov, sam_cov):
        vt, vc = getattr(view_info, "view_type", "unknown"), float(view_info.confidence)
        if vt == "aerial" and vc >= 0.75:
            w = {"S": 0.22, "I": 0.30, "R": 0.12, "V": 0.06, "A": 0.18, "Z": 0.12}
        elif vt == "broadcast":
            w = {"S": 0.38, "I": 0.25, "R": 0.00, "V": 0.08, "A": 0.05, "Z": 0.24}
        else:
            w = {"S": 0.40, "I": 0.20, "R": 0.00, "V": 0.10, "A": 0.05, "Z": 0.25}

        if hsv_cov < 0.10:
            w["I"] *= 0.5
            w["S"] += 0.10
        if sam_cov < 0.08:
            if hsv_cov > 0.30:
                w["I"] += 0.15; w["S"] -= 0.10
            else:
                w["Z"] += 0.10; w["V"] += 0.05; w["S"] -= 0.05
        if vc < 0.6:
            w["V"] *= 0.4

        total = sum(w.values()) or 1.0
        for k in w: w[k] = max(0.0, w[k] / total)
        return w

    # --- coverage penalty ----------------------------------------------------
    @staticmethod
    def _coverage_penalty(cov):
        if 0.20 <= cov <= 0.70:
            pen = 1.0
        elif cov > 0.70:
            pen = 1.0 - 1.5 * (cov - 0.70)
        else:
            pen = 1.0 - 2.0 * (0.20 - cov)
        return float(np.clip(pen, 0.3, 1.0))

    # --- metric functions (keep existing logic) ------------------------------
    @staticmethod
    def compute_hsv_iou(mask, hsv_mask):
        inter = np.logical_and(mask, hsv_mask).sum()
        union = np.logical_or(mask, hsv_mask).sum()
        return inter / union if union > 0 else 0.0

    @staticmethod
    def compute_rectangularity(mask):
        mask_u8 = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return 0.0
        area = cv2.contourArea(max(cnts, key=cv2.contourArea))
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return area / (w * h + 1e-6)

    @staticmethod
    def compute_aspect_ratio_score(mask):
        mask_u8 = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return 0.0
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        ratio = w / (h + 1e-6)
        return 1.0 - min(abs(ratio - 1.6), 1.0)  # ideal ≈ 1.6–2.0

    @staticmethod
    def compute_size_score(mask, shape):
        h, w = shape[:2]
        cov = mask.sum() / (h * w)
        if cov < 0.3: return cov / 0.3
        if cov > 0.7: return max(0.0, 1 - (cov - 0.7) / 0.3)
        return 1.0

    @staticmethod
    def compute_spatial_penalty(mask, view_info, shape):
        h, _ = shape[:2]
        M = cv2.moments((mask * 255).astype(np.uint8))
        if M["m00"] == 0: return 0.5
        cy = M["m01"] / M["m00"] / h
        y_min, y_max = view_info.expected_field_region
        if y_min <= cy <= y_max:
            return 1.0
        return max(0.3, 1.0 - abs(cy - np.mean([y_min, y_max])) * 2)

    # --- main evaluation -----------------------------------------------------
    @staticmethod
    def evaluate(mask, sam_score, hsv_mask, view_info, image_shape,
                 image_name="<unknown>", strategy="<unknown>"):
        hsv_iou = RobustConfidenceEvaluator.compute_hsv_iou(mask, hsv_mask)
        rect = RobustConfidenceEvaluator.compute_rectangularity(mask)
        ar = RobustConfidenceEvaluator.compute_aspect_ratio_score(mask)
        size = RobustConfidenceEvaluator.compute_size_score(mask, image_shape)
        spatial = RobustConfidenceEvaluator.compute_spatial_penalty(mask, view_info, image_shape)

        h, w = image_shape[:2]
        sam_cov = mask.sum() / (h * w)
        hsv_cov = view_info.field_coverage
        wts = RobustConfidenceEvaluator._adaptive_weights(view_info, hsv_cov, sam_cov)

        base = (wts["S"]*sam_score + wts["I"]*hsv_iou + wts["R"]*rect +
                wts["V"]*view_info.confidence + wts["A"]*ar + wts["Z"]*size)
        cov_pen = RobustConfidenceEvaluator._coverage_penalty(sam_cov)
        composite = float(np.clip(base * (0.5 + 0.5 * spatial) * cov_pen, 0.0, 1.0))

        RobustConfidenceEvaluator._ensure_log_header()
        row = ",".join(map(str, [image_name, strategy, sam_score, hsv_iou, rect,
                                 view_info.confidence, ar, size, spatial,
                                 sam_cov, hsv_cov, composite]))
        with open(RobustConfidenceEvaluator.LOG_PATH, "a") as f:
            f.write(row + "\n")

        return composite, hsv_iou, rect, spatial, ar, size
# === STRATEGY SELECTION ======================================================
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


# === MORPHOLOGICAL TOOLS =====================================================
class MorphologicalExpander:
    @staticmethod
    def expand_mask(mask: np.ndarray, expansion_percent: float = 0.12) -> np.ndarray:
        mask_u8 = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask
        largest = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        ex, ey = int(w * expansion_percent), int(h * expansion_percent)
        x0, y0 = max(0, x - ex), max(0, y - ey)
        w2, h2 = min(mask.shape[1] - x0, w + 2 * ex), min(mask.shape[0] - y0, h + 2 * ey)
        expanded = np.zeros_like(mask_u8)
        expanded[y0:y0 + h2, x0:x0 + w2] = 255
        merged = cv2.bitwise_or(mask_u8, expanded)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
        return merged > 127


class MorphologicalRefiner:
    """Smooth and clean masks."""
    @staticmethod
    def refine_mask(mask: np.ndarray, aggressive=False) -> np.ndarray:
        mask_u8 = (mask * 255).astype(np.uint8)
        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15) if aggressive else (7, 7))
        ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10) if aggressive else (5, 5))
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.erode(mask_u8, ek)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ck)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ok)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            refined = np.zeros_like(mask_u8)
            cv2.drawContours(refined, [c], -1, 255, -1)
            refined = cv2.GaussianBlur(refined, (5, 5), 0)
            return refined > 127
        return m > 127


# === STRATEGY EXECUTION ======================================================
def execute_strategy(strategy_name: str, predictor, image: np.ndarray,
                     hsv_mask: np.ndarray, w: int, h: int, dtype):
    """Run the chosen segmentation strategy on the predictor."""
    # --- build prompts -------------------------------------------------------
    if strategy_name == "single_center_lower":
        cy = int(h * 0.58)
        pts, lbl = np.array([[w // 2, cy]]), np.array([1])
    elif strategy_name == "multi_point_dense":
        mx, my = int(w * 0.15), int(h * 0.15)
        pts = np.array([[mx, my], [w // 2, my], [w - mx, my],
                        [mx, h // 2], [w // 2, h // 2], [w - mx, h // 2],
                        [mx, h - my], [w // 2, h - my], [w - mx, h - my]])
        lbl = np.ones(len(pts))
    elif strategy_name == "multi_point_lower":
        cy, mx = int(h * 0.60), int(w * 0.15)
        pts = np.array([[mx, cy], [w // 2, cy], [w - mx, cy],
                        [w // 4, cy + int(h * 0.15)], [3 * w // 4, cy + int(h * 0.15)], [w // 2, h - 80]])
        lbl = np.ones(len(pts))
    elif strategy_name == "box_strategy_wide":
        box = np.array([int(w * 0.02), int(h * 0.02), int(w * 0.98), int(h * 0.98)])
        with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
            return predictor.predict(box=box, multimask_output=True)[:2]
    elif strategy_name == "box_point_hybrid":
        box = np.array([int(w * 0.03), int(h * 0.03), int(w * 0.97), int(h * 0.97)])
        pts, lbl = np.array([[w // 2, h // 2]]), np.array([1])
        with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
            return predictor.predict(point_coords=pts, point_labels=lbl, box=box, multimask_output=True)[:2]
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
        pts, lbl = np.array([[w // 2, h // 2]]), np.array([1])

    # --- predict -------------------------------------------------------------
    with torch.inference_mode(), torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
        return predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=True)[:2]


# === MAIN PIPELINE ===========================================================
def test_stage_a_segmentation():
    """Run full Stage A segmentation pipeline."""
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("[WARNING] Running on CPU")

    root = Path(__file__).parent.parent.parent
    ckpt = str(root / "atlas/models/sam2/checkpoints/sam2.1_hiera_large.pt")
    cfg = str(root / "atlas/models/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    test_dir = root / "test_data/frames"

    today = datetime.now().strftime("%Y-%m-%d_%H%M")
    outdir = root / f"output/sam2/stage_a/{today}"
    os.makedirs(outdir, exist_ok=True)

    imgs = [f for f in test_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    if not imgs:
        print("❌ No images found."); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print("[INFO] Loading SAM 2.1...")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    try:
        model = build_sam2(cfg, ckpt, device=device)
    except TypeError:
        model = build_sam2(cfg, ckpt); model = model.to(device)
    predictor = SAM2ImagePredictor(model)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    view_cls, hsv_prep, strat, expander, evaluator = (
        EnhancedViewClassifier(), HSVPreprocessor(), AdaptiveSAMStrategy(),
        MorphologicalExpander(), RobustConfidenceEvaluator()
    )

    for i, path in enumerate(imgs, 1):
        print(f"\n{'='*70}\n📸 Image {i}/{len(imgs)}: {path.name}\n{'='*70}")
        img = np.array(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]

        view = view_cls.classify(img)
        print(f"  → View: {view.view_type} (conf={view.confidence:.2f})  coverage={view.field_coverage:.2%}")

        hsv_mask = hsv_prep.extract_field_mask(img)
        enhanced = hsv_prep.enhance_contrast(img, boost=view.confidence < 0.75)
        print(f"  HSV coverage: {hsv_mask.sum()/(h*w):.1%}")

        predictor.set_image(enhanced)
        best = None
        results = []

        for sname in strat.get_strategies(view.view_type):
            try:
                start = time.time()
                masks, scores = execute_strategy(sname, predictor, enhanced, hsv_mask, w, h, dtype)
                for m, sc in zip(masks, scores):
                    comp, iou, rect, spat, ar, sz = evaluator.evaluate(m, sc, hsv_mask, view, img.shape, path.name, sname)
                    res = SegmentationResult(m, sc, comp, sname, view.view_type,
                                              time.time() - start, iou, rect, view.confidence, spat, ar, sz)
                    results.append(res)
                    if not best or comp > best.composite_confidence:
                        best = res
                b = max([r for r in results if r.strategy == sname], key=lambda x: x.composite_confidence)
                print(f"  ✓ {sname:25} | Comp={b.composite_confidence:.3f}  Spatial={b.spatial_penalty:.2f}")
            except Exception as e:
                print(f"  ✗ {sname:25} | Error: {e}")

        if not best:
            print("❌ No valid results."); continue

        print(f"\nBest Strategy: {best.strategy}  Confidence={best.composite_confidence:.3f}")
        exp_mask = expander.expand_mask(best.mask, 0.12)
        print(f"Expanded mask: {best.mask.sum():,} → {exp_mask.sum():,} px")

        refiner = MorphologicalRefiner()
        sam_cov = exp_mask.sum() / (h * w)
        fused = np.logical_or(exp_mask, hsv_mask) if view.view_type == "aerial" or sam_cov < 0.25 else exp_mask
        final = refiner.refine_mask(fused, aggressive=(view.view_type == "aerial"))

        np.save(outdir / f"{path.stem}_stage_a_mask.npy", final.astype(np.uint8))
        save_stage_a_visualization(img, hsv_mask, enhanced, results[:8], final, best, view,
                                   outdir / f"{path.stem}_stage_a.png")

    print("\n✅ Stage A complete. Results saved in", outdir)


# === VISUALIZATION ===========================================================
def save_stage_a_visualization(image, hsv_mask, enhanced, results, final_mask, best_result, view_info, save_path):
    fig = plt.figure(figsize=(24, 14))
    h, w = image.shape[:2]
    expected = np.zeros_like(image)
    y0, y1 = int(h * view_info.expected_field_region[0]), int(h * view_info.expected_field_region[1])
    expected[y0:y1, :] = [0, 100, 0]
    plt.subplot(3, 4, 1); plt.imshow(image); plt.imshow(expected, alpha=0.25)
    plt.title(f"Original + Expected ({view_info.view_type})", fontsize=11); plt.axis("off")
    plt.subplot(3, 4, 2); plt.imshow(hsv_mask, cmap="Greens"); plt.title("HSV Mask", fontsize=11); plt.axis("off")
    plt.subplot(3, 4, 3); plt.imshow(enhanced); plt.title("CLAHE Enhanced", fontsize=11); plt.axis("off")
    plt.subplot(3, 4, 4); plt.imshow(image); plt.imshow(final_mask, alpha=0.6, cmap="plasma")
    plt.title(f"Final Mask ({best_result.strategy})", fontsize=11, color="green"); plt.axis("off")

    for idx, r in enumerate(results):
        plt.subplot(3, 4, idx + 5)
        plt.imshow(image); plt.imshow(r.mask, alpha=0.5, cmap="viridis")
        mark = "★ " if r.strategy == best_result.strategy else ""
        plt.title(f"{mark}{r.strategy}\nComp={r.composite_confidence:.3f}", fontsize=9)
        plt.axis("off")

    plt.suptitle("STAGE A: Adaptive Confidence Evaluation", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


# === ADAPTIVE WEIGHT VISUALIZER =============================================
def visualize_adaptive_weights():
    """Plot adaptive weight curves for aerial/broadcast/ground."""
    dummy = [
        ("aerial", 0.9, 0.6, 0.6),
        ("broadcast", 0.8, 0.4, 0.3),
        ("ground", 0.7, 0.5, 0.2),
    ]
    metrics = ["S", "I", "R", "V", "A", "Z"]
    plt.figure(figsize=(8, 4))
    for vt, vc, hc, sc in dummy:
        vw = ViewClassification(vt, vc, hc, 45, (0.3, 0.9))
        wts = RobustConfidenceEvaluator._adaptive_weights(vw, hc, sc)
        plt.plot(metrics, [wts[m] for m in metrics], marker="o", label=f"{vt} (conf={vc})")
    plt.title("Adaptive Metric Weights per View Type"); plt.ylabel("Weight")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


# === ENTRYPOINT ==============================================================
if __name__ == "__main__":
    # 1️⃣ Run adaptive weight visualization
    #visualize_adaptive_weights()
    # 2️⃣ Uncomment next line to run full Stage A test
    test_stage_a_segmentation()
