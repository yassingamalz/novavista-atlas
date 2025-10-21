"""
Headless image test - no GUI windows, just saves results
Perfect for batch processing or remote servers
"""

import cv2
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from atlas.core import AtlasProcessor
from atlas.preprocessing.field_segmentation import FieldSegmenter
from atlas.detection.line_detector import LineDetector
from atlas.detection.circle_detector import CircleDetector
from atlas.preprocessing.enhancement import ImageEnhancer


def main():
    """Test with single image - no GUI."""
    
    if len(sys.argv) < 2:
        print("Usage: python test_image_headless.py <path_to_image>")
        print("\nExample:")
        print("  python test_image_headless.py test_data/frames/sample.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"[X] Error: Image not found at '{image_path}'")
        sys.exit(1)
    
    print("=" * 60)
    print("NovaVista Atlas - Headless Image Test")
    print("=" * 60)
    print(f"Input: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[X] Error: Cannot load image")
        sys.exit(1)
    
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")
    print("\nProcessing...")
    
    # Process
    processor = AtlasProcessor()
    result = processor.process_frame(image_path)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Status:           {result['field_detection']['status'].upper()}")
    print(f"Field Coverage:   {result['field_detection']['field_percentage']:.1f}%")
    print(f"Lines Detected:   {result['field_detection']['lines_detected']}")
    print(f"Circles Detected: {result['field_detection']['circles_detected']}")
    
    homography_status = "[OK] Calculated" if result['calibration']['homography_matrix'] else "[FAIL] Failed"
    print(f"Homography:       {homography_status}")
    print(f"Confidence:       {result['field_detection']['confidence']:.2f}")
    print(f"Processing Time:  {result['processing_metadata']['processing_time_ms']:.0f}ms")
    print("=" * 60)
    
    # Create visualization
    enhancer = ImageEnhancer()
    segmenter = FieldSegmenter()
    line_detector = LineDetector()
    circle_detector = CircleDetector()
    
    enhanced = enhancer.enhance(image)
    field_mask = segmenter.segment_field(enhanced)
    lines = line_detector.detect(enhanced, mask=field_mask)
    circles = circle_detector.detect(enhanced, mask=field_mask)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Full visualization
    output = image.copy()
    
    # Draw field mask overlay
    overlay = image.copy()
    overlay[field_mask > 0] = [0, 255, 0]
    output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
    
    # Draw lines
    for x1, y1, x2, y2 in lines:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # Draw circles
    for x, y, r in circles:
        cv2.circle(output, (x, y), r, (255, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (255, 255, 0), -1)
    
    # Add status text
    status_color = (0, 255, 0) if result['field_detection']['status'] == 'success' else (0, 165, 255)
    cv2.putText(output, f"Status: {result['field_detection']['status'].upper()}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(output, f"Field: {result['field_detection']['field_percentage']:.1f}%", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(output, f"Lines: {result['field_detection']['lines_detected']}", 
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(output, f"Circles: {result['field_detection']['circles_detected']}", 
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save full visualization
    output_path = output_dir / "test_result.jpg"
    cv2.imwrite(str(output_path), output)
    print(f"\n[OK] Full result saved to: {output_path}")
    
    # 2. Save field mask
    mask_path = output_dir / "field_mask.jpg"
    cv2.imwrite(str(mask_path), field_mask)
    print(f"[OK] Field mask saved to: {mask_path}")
    
    # 3. Save lines only
    lines_output = image.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(lines_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    lines_path = output_dir / "lines_only.jpg"
    cv2.imwrite(str(lines_path), lines_output)
    print(f"[OK] Lines visualization saved to: {lines_path}")
    
    # 4. Save circles only
    circles_output = image.copy()
    for x, y, r in circles:
        cv2.circle(circles_output, (x, y), r, (255, 0, 255), 2)
        cv2.circle(circles_output, (x, y), 2, (255, 0, 255), -1)
    circles_path = output_dir / "circles_only.jpg"
    cv2.imwrite(str(circles_path), circles_output)
    print(f"[OK] Circles visualization saved to: {circles_path}")
    
    # 5. Save enhanced image
    enhanced_path = output_dir / "enhanced.jpg"
    cv2.imwrite(str(enhanced_path), enhanced)
    print(f"[OK] Enhanced image saved to: {enhanced_path}")
    
    # 6. Save JSON result
    json_path = output_dir / "analysis.json"
    with open(json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        result_copy = result.copy()
        if result_copy['calibration']['homography_matrix'] is not None:
            result_copy['calibration']['homography_matrix'] = [
                [float(x) for x in row] 
                for row in result_copy['calibration']['homography_matrix']
            ]
        json.dump(result_copy, f, indent=2)
    print(f"[OK] Analysis JSON saved to: {json_path}")
    
    print("\n" + "=" * 60)
    print("All outputs saved to 'output/' directory:")
    print("  - test_result.jpg    : Full visualization")
    print("  - field_mask.jpg     : Field segmentation")
    print("  - lines_only.jpg     : Detected lines")
    print("  - circles_only.jpg   : Detected circles")
    print("  - enhanced.jpg       : Preprocessed image")
    print("  - analysis.json      : Complete analysis data")
    print("=" * 60)


if __name__ == "__main__":
    main()
