"""
ATLAS V2 - Stage B: Line Detection & Refinement Test
Uses results from Stage A to detect lines and refine field mask

Pipeline:
5. Line Detection (white lines using HSV + Canny + Hough)
6. Line-Based Field Refinement (use lines to exclude non-field areas)
7. Final Fused Mask

Input: Stage A mask (from test_stage_a_segmentation.py)
Output: Final refined mask with line validation
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

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


@dataclass
class LineSegment:
    """Detected line segment"""
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float
    length: float
    confidence: float


class WhiteLineDetector:
    """Detect white lines on football pitch"""
    
    @staticmethod
    def detect_white_regions(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract white regions within field mask"""
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        
        # Convert to HSV
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
        
        # Multiple white ranges for different lighting
        white_ranges = [
            ([0, 0, 200], [180, 30, 255]),      # Bright white
            ([0, 0, 180], [180, 50, 255]),      # White with slight color
            ([0, 0, 160], [180, 60, 240])       # Dimmer white
        ]
        
        combined_white = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower, upper in white_ranges:
            white_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_white = cv2.bitwise_or(combined_white, white_mask)
        
        # Morphological cleanup - remove noise, keep lines
        kernel_denoise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_white = cv2.morphologyEx(combined_white, cv2.MORPH_OPEN, kernel_denoise)
        
        # Dilate slightly to connect broken lines
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_white = cv2.dilate(combined_white, kernel_dilate, iterations=1)
        
        return combined_white
    
    @staticmethod
    def enhance_lines(white_mask: np.ndarray) -> np.ndarray:
        """Enhance line detection with Canny"""
        # Blur slightly to reduce noise
        blurred = cv2.GaussianBlur(white_mask, (3, 3), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        return edges


class HoughLineDetector:
    """Detect lines using Hough Transform"""
    
    @staticmethod
    def detect_lines(edges: np.ndarray, min_line_length: int = 30) -> List[LineSegment]:
        """Detect lines using Probabilistic Hough Transform"""
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_line_length,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # Convert to LineSegment objects
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle and length
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Confidence based on length
            confidence = min(length / 200, 1.0)
            
            line_segments.append(LineSegment(
                x1=int(x1), y1=int(y1),
                x2=int(x2), y2=int(y2),
                angle=angle,
                length=length,
                confidence=confidence
            ))
        
        return line_segments
    
    @staticmethod
    def filter_football_lines(lines: List[LineSegment]) -> List[LineSegment]:
        """Filter lines by angle - keep horizontal and vertical lines"""
        filtered = []
        
        for line in lines:
            # Normalize angle to -90 to 90
            angle = line.angle % 180
            if angle > 90:
                angle -= 180
            
            # Keep mostly horizontal (±15°) or vertical (75-90°, -75 to -90°)
            is_horizontal = abs(angle) < 15
            is_vertical = abs(abs(angle) - 90) < 15
            
            if (is_horizontal or is_vertical) and line.length > 20:
                filtered.append(line)
        
        return filtered


class LineBasedRefiner:
    """Refine field mask using detected lines - GEOMETRIC EXTENSION"""
    
    @staticmethod
    def create_line_mask(lines: List[LineSegment], shape: Tuple[int, int]) -> np.ndarray:
        """Create mask from detected lines"""
        line_mask = np.zeros(shape, dtype=np.uint8)
        
        for line in lines:
            cv2.line(
                line_mask,
                (line.x1, line.y1),
                (line.x2, line.y2),
                255,
                thickness=int(line.confidence * 10 + 3)
            )
        
        # Dilate to create zones around lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        line_mask = cv2.dilate(line_mask, kernel, iterations=2)
        
        return line_mask > 0
    
    @staticmethod
    def create_geometric_field_mask(lines: List[LineSegment], shape: Tuple[int, int]) -> np.ndarray:
        """Create perfect rectangular field mask from detected lines"""
        if len(lines) < 4:
            return None
        
        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []
        
        for line in lines:
            angle = line.angle % 180
            if angle > 90:
                angle -= 180
            
            if abs(angle) < 15:  # Horizontal
                h_lines.append(line)
            elif abs(abs(angle) - 90) < 15:  # Vertical
                v_lines.append(line)
        
        if len(h_lines) < 2 or len(v_lines) < 2:
            return None
        
        # Find extreme lines (top, bottom, left, right)
        top_y = min([min(line.y1, line.y2) for line in h_lines])
        bottom_y = max([max(line.y1, line.y2) for line in h_lines])
        left_x = min([min(line.x1, line.x2) for line in v_lines])
        right_x = max([max(line.x1, line.x2) for line in v_lines])
        
        # Create rectangular mask
        geometric_mask = np.zeros(shape, dtype=np.uint8)
        cv2.rectangle(geometric_mask, (left_x, top_y), (right_x, bottom_y), 255, -1)
        
        return geometric_mask > 0
    
    @staticmethod
    def refine_mask_with_lines(field_mask: np.ndarray, line_mask: np.ndarray,
                                lines: List[LineSegment], shape: Tuple[int, int]) -> np.ndarray:
        """Refine field mask using line positions - EXTEND TO GEOMETRIC BOUNDARIES"""
        
        if len(lines) == 0:
            print("  [WARNING] No lines detected, returning original mask")
            return field_mask
        
        # Try to create perfect geometric field mask
        geometric_mask = LineBasedRefiner.create_geometric_field_mask(lines, shape)
        
        if geometric_mask is not None:
            # Check overlap with original mask
            overlap = np.logical_and(field_mask, geometric_mask)
            overlap_ratio = overlap.sum() / field_mask.sum() if field_mask.sum() > 0 else 0
            
            if overlap_ratio > 0.5:
                print(f"  [GEOMETRIC] Using line-based rectangle ({overlap_ratio:.1%} overlap)")
                return geometric_mask
        
        # Fallback: Find bounding box of all lines
        all_x = []
        all_y = []
        for line in lines:
            all_x.extend([line.x1, line.x2])
            all_y.extend([line.y1, line.y2])
        
        if not all_x:
            return field_mask
        
        # Calculate bounds with small margin
        margin_x = int((max(all_x) - min(all_x)) * 0.03)
        margin_y = int((max(all_y) - min(all_y)) * 0.03)
        
        x_min = max(0, min(all_x) - margin_x)
        x_max = min(field_mask.shape[1], max(all_x) + margin_x)
        y_min = max(0, min(all_y) - margin_y)
        y_max = min(field_mask.shape[0], max(all_y) + margin_y)
        
        # Create line-bounded rectangular mask
        bounded_mask = np.zeros_like(field_mask)
        bounded_mask[y_min:y_max, x_min:x_max] = 1
        
        # Check coverage
        line_overlap = np.logical_and(field_mask, line_mask)
        line_coverage = line_overlap.sum() / field_mask.sum() if field_mask.sum() > 0 else 0
        
        if line_coverage > 0.10:
            print(f"  [BOUNDED] Using line bounding box ({line_coverage:.1%} coverage)")
            return bounded_mask.astype(bool)
        else:
            print(f"  [FALLBACK] Low line coverage ({line_coverage:.1%}), keeping oversegmented mask")
            return field_mask


class FinalFusion:
    """Fuse all results for final mask"""
    
    @staticmethod
    def fuse(stage_a_mask: np.ndarray, line_refined_mask: np.ndarray, 
             line_mask: np.ndarray, lines: List[LineSegment]) -> np.ndarray:
        """Intelligent fusion of all masks"""
        
        # If no lines detected, return Stage A mask
        if len(lines) < 3:
            print("  [FUSION] Few lines detected, using Stage A mask")
            return stage_a_mask
        
        # Calculate overlap
        overlap = np.logical_and(stage_a_mask, line_refined_mask)
        overlap_ratio = overlap.sum() / stage_a_mask.sum() if stage_a_mask.sum() > 0 else 0
        
        # If high overlap, use line-refined (more precise)
        if overlap_ratio > 0.75:
            print(f"  [FUSION] High overlap ({overlap_ratio:.1%}), using line-refined mask")
            fused = line_refined_mask
        else:
            # Moderate overlap - take union but keep only largest component
            print(f"  [FUSION] Moderate overlap ({overlap_ratio:.1%}), using validated union")
            fused = np.logical_or(stage_a_mask, line_refined_mask)
        
        # Final morphological refinement
        fused_uint8 = (fused * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        fused_uint8 = cv2.morphologyEx(fused_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Keep largest component
        contours, _ = cv2.findContours(fused_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            final = np.zeros_like(fused_uint8)
            cv2.drawContours(final, [largest], -1, 255, -1)
            
            # Smooth
            final = cv2.GaussianBlur(final, (5, 5), 0)
            return (final > 127).astype(bool)
        
        return fused.astype(bool)


def test_stage_b_line_detection():
    """Stage B: Line detection and refinement test"""
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    
    # Find latest Stage A output
    stage_a_base = project_root / "output/sam2/stage_a"
    stage_a_dirs = sorted(stage_a_base.glob("*"), reverse=True)
    
    if not stage_a_dirs:
        print("❌ No Stage A results found. Run test_stage_a_segmentation.py first!")
        return
    
    stage_a_dir = stage_a_dirs[0]
    print(f"[INFO] Using Stage A results from: {stage_a_dir.name}")
    
    # Output directory
    today = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = project_root / f"output/sam2/stage_b/{today}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Stage A masks
    mask_files = list(stage_a_dir.glob("*_stage_a_mask.npy"))
    
    if not mask_files:
        print("❌ No Stage A masks found!")
        return
    
    print(f"[INFO] Found {len(mask_files)} masks to process\n")
    
    # Initialize components
    white_detector = WhiteLineDetector()
    hough_detector = HoughLineDetector()
    line_refiner = LineBasedRefiner()
    fuser = FinalFusion()
    
    # Process each mask
    test_data_dir = project_root / "test_data/frames"
    
    for mask_idx, mask_path in enumerate(mask_files):
        print(f"\n{'='*70}")
        print(f"📸 Image {mask_idx + 1}/{len(mask_files)}: {mask_path.stem}")
        print(f"{'='*70}")
        
        # Load Stage A mask
        stage_a_mask = np.load(mask_path).astype(bool)
        
        # Find corresponding original image
        image_name = mask_path.stem.replace('_stage_a_mask', '')
        image_files = list(test_data_dir.glob(f"{image_name}.*"))
        
        if not image_files:
            print(f"❌ Original image not found for {image_name}")
            continue
        
        image = np.array(Image.open(image_files[0]).convert("RGB"))
        print(f"  Stage A mask area: {stage_a_mask.sum():,} pixels")
        
        # Step 5: White line detection
        print("\n[5/7] Detecting white lines...")
        white_mask = white_detector.detect_white_regions(image, stage_a_mask)
        edges = white_detector.enhance_lines(white_mask)
        print(f"  White pixels detected: {white_mask.sum():,}")
        print(f"  Edge pixels: {edges.sum():,}")
        
        # Step 6: Hough line detection
        print("\n[6/7] Hough line detection...")
        all_lines = hough_detector.detect_lines(edges, min_line_length=30)
        filtered_lines = hough_detector.filter_football_lines(all_lines)
        print(f"  Total lines detected: {len(all_lines)}")
        print(f"  Football lines (filtered): {len(filtered_lines)}")
        
        if filtered_lines:
            avg_length = np.mean([line.length for line in filtered_lines])
            print(f"  Average line length: {avg_length:.1f}px")
        
        # Step 7: Line-based geometric refinement
        print("\n[7/7] Geometric refinement with lines...")
        line_mask = line_refiner.create_line_mask(filtered_lines, stage_a_mask.shape)
        line_refined_mask = line_refiner.refine_mask_with_lines(
            stage_a_mask, line_mask, filtered_lines, stage_a_mask.shape
        )
        
        # Final fusion
        final_mask = fuser.fuse(stage_a_mask, line_refined_mask, line_mask, filtered_lines)
        
        print(f"  Final mask area: {final_mask.sum():,} pixels")
        print(f"  Change from Stage A: {(final_mask.sum() - stage_a_mask.sum()):+,} pixels")
        
        # Save results
        print("\n[SAVE] Saving Stage B results...")
        
        # Save PNG overlay
        final_png = (final_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / f"{image_name}_final_mask.png"), final_png)
        
        # Save visualization
        save_stage_b_visualization(
            image, stage_a_mask, white_mask, edges, 
            filtered_lines, line_refined_mask, final_mask,
            output_dir / f"{image_name}_stage_b.png"
        )
        
        print(f"  ✓ Final mask PNG saved")
        print(f"  ✓ Visualization saved")
    
    print(f"\n{'='*70}")
    print(f"✅ Stage B complete! Results: {output_dir}")
    print(f"{'='*70}")


def save_stage_b_visualization(image, stage_a_mask, white_mask, edges,
                                lines, line_refined, final_mask, save_path):
    """Create Stage B visualization showing all steps"""
    
    fig = plt.figure(figsize=(24, 12))
    
    # Row 1
    plt.subplot(2, 4, 1)
    plt.imshow(image)
    plt.title("1. Original Image", fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(image)
    plt.imshow(stage_a_mask, alpha=0.5, cmap='Greens')
    plt.title(f"2. Stage A Mask (Input)\nArea: {stage_a_mask.sum():,}px", fontsize=12)
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(image)
    plt.imshow(white_mask, alpha=0.7, cmap='hot')
    plt.title(f"5. White Line Detection (HSV)\n{white_mask.sum():,} white pixels", 
              fontsize=12)
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(edges, cmap='gray')
    plt.title(f"5b. Canny Edges\n{edges.sum():,} edge pixels", fontsize=12)
    plt.axis('off')
    
    # Row 2
    plt.subplot(2, 4, 5)
    line_overlay = image.copy()
    for line in lines:
        color = (0, 255, 0) if abs(line.angle) < 15 else (255, 0, 0)
        cv2.line(line_overlay, (line.x1, line.y1), (line.x2, line.y2), color, 2)
    plt.imshow(line_overlay)
    plt.title(f"6. Hough Lines Detected\n{len(lines)} lines found", fontsize=12)
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(image)
    plt.imshow(line_refined, alpha=0.5, cmap='plasma')
    plt.title(f"7. Line-Refined Mask\nArea: {line_refined.sum():,}px", fontsize=12)
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    # Show difference
    diff = np.zeros((*stage_a_mask.shape, 3), dtype=np.uint8)
    diff[stage_a_mask & ~final_mask] = [255, 0, 0]      # Removed: red
    diff[~stage_a_mask & final_mask] = [0, 255, 0]      # Added: green
    diff[stage_a_mask & final_mask] = [100, 100, 100]   # Kept: gray
    plt.imshow(diff)
    plt.title("7b. Mask Changes\nRed=Removed, Green=Added", fontsize=12)
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.imshow(image)
    plt.imshow(final_mask, alpha=0.6, cmap='plasma')
    plt.title(f"8. FINAL FUSED MASK\nArea: {final_mask.sum():,}px\n"
              f"Change: {(final_mask.sum() - stage_a_mask.sum()):+,}px",
              fontsize=12, fontweight='bold', color='green')
    plt.axis('off')
    
    plt.suptitle("STAGE B: Line Detection & Refinement Pipeline", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    test_stage_b_line_detection()
