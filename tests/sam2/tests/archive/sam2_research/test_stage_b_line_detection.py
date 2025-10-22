"""
ATLAS V2 - Stage B: Line Detection & Refinement Test
Uses results from Stage A to detect lines and refine field mask

Pipeline:
5. Line Detection (white lines using HSV + Canny + Hough)
6. Line-Based Field Refinement (use lines to exclude non-field areas)
7. Final Fused Mask
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# Data Structures
# ============================================================

@dataclass
class LineSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float
    length: float
    confidence: float


# ============================================================
# White Line Detector
# ============================================================

class WhiteLineDetector:
    """Detect white lines on football pitch"""

    @staticmethod
    def detect_white_regions(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract white regions within field mask"""
        masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)

        white_ranges = [
            ([0, 0, 200], [180, 30, 255]),
            ([0, 0, 180], [180, 50, 255]),
            ([0, 0, 160], [180, 60, 240]),
        ]

        combined_white = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower, upper in white_ranges:
            white_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_white = cv2.bitwise_or(combined_white, white_mask)

        kernel_denoise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_white = cv2.morphologyEx(combined_white, cv2.MORPH_OPEN, kernel_denoise)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_white = cv2.dilate(combined_white, kernel_dilate, iterations=1)

        return combined_white

    @staticmethod
    def enhance_lines(white_mask: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(white_mask, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        return edges


# ============================================================
# Hough Line Detector
# ============================================================

class HoughLineDetector:
    """Detect lines using Hough Transform"""

    @staticmethod
    def detect_lines(edges: np.ndarray, min_line_length: int = 30) -> List[LineSegment]:
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=50,
            minLineLength=min_line_length, maxLineGap=10
        )
        if lines is None:
            return []

        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            confidence = min(length / 200, 1.0)
            line_segments.append(LineSegment(x1, y1, x2, y2, angle, length, confidence))
        return line_segments

    @staticmethod
    def filter_football_lines(lines: List[LineSegment]) -> List[LineSegment]:
        """Keep mostly horizontal or vertical lines"""
        filtered = []
        for line in lines:
            angle = line.angle % 180
            if angle > 90:
                angle -= 180
            is_horizontal = abs(angle) < 15
            is_vertical = abs(abs(angle) - 90) < 15
            if (is_horizontal or is_vertical) and line.length > 20:
                filtered.append(line)
        return filtered


# ============================================================
# Line-Based Refiner
# ============================================================

class LineBasedRefiner:
    """Refine field mask using detected lines - enhanced version"""

    @staticmethod
    def create_line_mask(lines: List[LineSegment], shape: Tuple[int, int]) -> np.ndarray:
        line_mask = np.zeros(shape, dtype=np.uint8)
        for line in lines:
            cv2.line(line_mask, (line.x1, line.y1), (line.x2, line.y2),
                     255, thickness=int(line.confidence * 8 + 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        line_mask = cv2.dilate(line_mask, kernel, iterations=2)
        return line_mask > 0

    @staticmethod
    def create_geometric_field_mask(lines: List[LineSegment], shape: Tuple[int, int]) -> np.ndarray | None:
        if len(lines) < 4:
            return None

        h_lines, v_lines = [], []
        for line in lines:
            angle = line.angle % 180
            if angle > 90:
                angle -= 180
            if abs(angle) < 15:
                h_lines.append(line)
            elif abs(abs(angle) - 90) < 15:
                v_lines.append(line)

        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        top_y = min([min(line.y1, line.y2) for line in h_lines])
        bottom_y = max([max(line.y1, line.y2) for line in h_lines])
        left_x = min([min(line.x1, line.x2) for line in v_lines])
        right_x = max([max(line.x1, line.x2) for line in v_lines])

        geometric_mask = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(geometric_mask, (left_x, top_y), (right_x, bottom_y), 255, -1)
        return geometric_mask > 0

    @staticmethod
    def refine_mask_with_lines(field_mask: np.ndarray, line_mask: np.ndarray,
                               lines: List[LineSegment], shape: Tuple[int, int]) -> np.ndarray:
        """Enhanced refinement logic that trusts strong line geometry"""

        if len(lines) == 0:
            print("  [WARNING] No lines detected, returning original mask")
            return field_mask

        geometric_mask = LineBasedRefiner.create_geometric_field_mask(lines, shape)

        if geometric_mask is not None:
            overlap = np.logical_and(field_mask, geometric_mask)
            overlap_ratio = overlap.sum() / field_mask.sum() if field_mask.sum() > 0 else 0
            geo_ratio = geometric_mask.sum() / field_mask.sum()

            if geo_ratio < 1.2 or overlap_ratio > 0.2:
                print(f"  [GEOMETRIC] Using line-based rectangle (area ratio={geo_ratio:.2f}, overlap={overlap_ratio:.2f})")
                return geometric_mask

        all_x, all_y = [], []
        for line in lines:
            all_x.extend([line.x1, line.x2])
            all_y.extend([line.y1, line.y2])
        if not all_x:
            return field_mask

        margin_x = int((max(all_x) - min(all_x)) * 0.02)
        margin_y = int((max(all_y) - min(all_y)) * 0.02)
        x_min = max(0, min(all_x) - margin_x)
        x_max = min(field_mask.shape[1], max(all_x) + margin_x)
        y_min = max(0, min(all_y) - margin_y)
        y_max = min(field_mask.shape[0], max(all_y) + margin_y)

        bounded_mask = np.zeros_like(field_mask)
        bounded_mask[y_min:y_max, x_min:x_max] = 1

        line_overlap = np.logical_and(field_mask, line_mask)
        line_coverage = line_overlap.sum() / field_mask.sum() if field_mask.sum() > 0 else 0

        if line_coverage > 0.03:
            print(f"  [BOUNDED] Using line bounding box ({line_coverage:.1%} coverage)")
            return bounded_mask.astype(bool)

        print(f"  [FALLBACK] Low line coverage ({line_coverage:.1%}), keeping oversegmented mask")
        return field_mask


# ============================================================
# Final Fusion
# ============================================================

class FinalFusion:
    @staticmethod
    def fuse(stage_a_mask: np.ndarray, line_refined_mask: np.ndarray,
             line_mask: np.ndarray, lines: List[LineSegment]) -> np.ndarray:

        if len(lines) < 3:
            print("  [FUSION] Few lines detected, using Stage A mask")
            return stage_a_mask

        overlap = np.logical_and(stage_a_mask, line_refined_mask)
        overlap_ratio = overlap.sum() / stage_a_mask.sum() if stage_a_mask.sum() > 0 else 0

        if overlap_ratio > 0.6:
            print(f"  [FUSION] High overlap ({overlap_ratio:.1%}), adopting line-refined mask")
            fused = line_refined_mask
        else:
            print(f"  [FUSION] Partial overlap ({overlap_ratio:.1%}), union fusion")
            fused = np.logical_or(stage_a_mask, line_refined_mask)

        fused_uint8 = (fused * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fused_uint8 = cv2.morphologyEx(fused_uint8, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fused_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            final = np.zeros_like(fused_uint8)
            cv2.drawContours(final, [largest], -1, 255, -1)
            final = cv2.GaussianBlur(final, (5, 5), 0)
            return (final > 127).astype(bool)
        return fused.astype(bool)


# ============================================================
# Main Test Function
# ============================================================

def test_stage_b_line_detection():
    project_root = Path(__file__).parent.parent.parent
    stage_a_base = project_root / "output/sam2/stage_a"
    stage_a_dirs = sorted(stage_a_base.glob("*"), reverse=True)

    if not stage_a_dirs:
        print("❌ No Stage A results found.")
        return

    stage_a_dir = stage_a_dirs[0]
    today = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = project_root / f"output/sam2/stage_b/{today}"
    os.makedirs(output_dir, exist_ok=True)

    mask_files = list(stage_a_dir.glob("*_stage_a_mask.npy"))
    if not mask_files:
        print("❌ No Stage A masks found!")
        return

    white_detector = WhiteLineDetector()
    hough_detector = HoughLineDetector()
    line_refiner = LineBasedRefiner()
    fuser = FinalFusion()

    test_data_dir = project_root / "test_data/frames"

    for i, mask_path in enumerate(mask_files):
        print(f"\n{'=' * 70}")
        print(f"📸 Image {i + 1}/{len(mask_files)}: {mask_path.stem}")
        print(f"{'=' * 70}")

        stage_a_mask = np.load(mask_path).astype(bool)
        image_name = mask_path.stem.replace("_stage_a_mask", "")
        image_files = list(test_data_dir.glob(f"{image_name}.*"))
        if not image_files:
            print(f"❌ Original image not found for {image_name}")
            continue
        image = np.array(Image.open(image_files[0]).convert("RGB"))

        white_mask = white_detector.detect_white_regions(image, stage_a_mask)
        edges = white_detector.enhance_lines(white_mask)

        all_lines = hough_detector.detect_lines(edges, 30)
        filtered_lines = hough_detector.filter_football_lines(all_lines)
        print(f"  Detected lines: {len(filtered_lines)}")

        line_mask = line_refiner.create_line_mask(filtered_lines, stage_a_mask.shape)
        line_refined_mask = line_refiner.refine_mask_with_lines(stage_a_mask, line_mask, filtered_lines, stage_a_mask.shape)
        final_mask = fuser.fuse(stage_a_mask, line_refined_mask, line_mask, filtered_lines)

        save_stage_b_visualization(
            image, stage_a_mask, white_mask, edges,
            filtered_lines, line_refined_mask, final_mask,
            output_dir / f"{image_name}_stage_b.png"
        )

        np.save(output_dir / f"{image_name}_final_mask.npy", final_mask)
        print(f"  ✓ Saved: {image_name}_final_mask.npy")

    print(f"\n✅ Stage B complete! Results: {output_dir}")


# ============================================================
# Visualization
# ============================================================

def save_stage_b_visualization(image, stage_a_mask, white_mask, edges,
                               lines, line_refined, final_mask, save_path):
    fig = plt.figure(figsize=(24, 12))
    plt.subplot(2, 4, 1)
    plt.imshow(image)
    plt.title("1. Original Image"); plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(image); plt.imshow(stage_a_mask, alpha=0.5, cmap="Greens")
    plt.title(f"2. Stage A Mask\nArea: {stage_a_mask.sum():,} px"); plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(image); plt.imshow(white_mask, alpha=0.6, cmap="hot")
    plt.title(f"5. White Line Detection\n{white_mask.sum():,} px"); plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(edges, cmap="gray")
    plt.title(f"5b. Canny Edges\n{edges.sum():,} px"); plt.axis("off")

    plt.subplot(2, 4, 5)
    overlay = image.copy()
    for l in lines:
        color = (0, 255, 0) if abs(l.angle) < 15 else (255, 0, 0)
        cv2.line(overlay, (l.x1, l.y1), (l.x2, l.y2), color, 2)
    plt.imshow(overlay)
    plt.title(f"6. Hough Lines ({len(lines)})"); plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(image); plt.imshow(line_refined, alpha=0.5, cmap="plasma")
    plt.title(f"7. Line-Refined Mask\nArea: {line_refined.sum():,} px"); plt.axis("off")

    plt.subplot(2, 4, 7)
    diff = np.zeros((*stage_a_mask.shape, 3), dtype=np.uint8)
    diff[stage_a_mask & ~final_mask] = [255, 0, 0]
    diff[~stage_a_mask & final_mask] = [0, 255, 0]
    diff[stage_a_mask & final_mask] = [80, 80, 80]
    plt.imshow(diff)
    plt.title("7b. Mask Changes\nRed=Removed, Green=Added"); plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.imshow(image); plt.imshow(final_mask, alpha=0.6, cmap="plasma")
    plt.title(f"8. FINAL FUSED MASK\nArea: {final_mask.sum():,} px", color="green"); plt.axis("off")

    plt.suptitle("STAGE B: Line Detection & Refinement Pipeline", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    test_stage_b_line_detection()
