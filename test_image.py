"""
Test Atlas with a single image - Simple visual output
For end users who just want to see the result
"""

import cv2
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from atlas.core import AtlasProcessor


def main():
    """Test with single image."""
    
    # Get image path
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <path_to_image>")
        print("\nExample:")
        print("  python test_image.py test_data/frames/sample.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check file exists
    if not Path(image_path).exists():
        print(f"[X] Error: Image not found at '{image_path}'")
        sys.exit(1)
    
    print("=" * 60)
    print("NovaVista Atlas - Image Test")
    print("=" * 60)
    print(f"Input: {image_path}\n")
    
    # Load and show original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[X] Error: Cannot load image")
        sys.exit(1)
    
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")
    
    # Show original
    cv2.imshow("Original Image", image)
    print("\n[Press any key to start processing...]")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Process
    print("\nProcessing...")
    processor = AtlasProcessor()
    result = processor.process_frame(image_path)
    
    # Print results with ASCII-safe characters
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Status:           {result['field_detection']['status'].upper()}")
    print(f"Field Coverage:   {result['field_detection']['field_percentage']:.1f}%")
    print(f"Lines Detected:   {result['field_detection']['lines_detected']}")
    print(f"Circles Detected: {result['field_detection']['circles_detected']}")
    
    # ASCII-safe homography status
    homography_status = "[OK] Calculated" if result['calibration']['homography_matrix'] else "[FAIL] Failed"
    print(f"Homography:       {homography_status}")
    print(f"Confidence:       {result['field_detection']['confidence']:.2f}")
    print(f"Processing Time:  {result['processing_metadata']['processing_time_ms']:.0f}ms")
    print("=" * 60)
    
    # Create simple visualization
    from atlas.preprocessing.field_segmentation import FieldSegmenter
    from atlas.detection.line_detector import LineDetector
    from atlas.preprocessing.enhancement import ImageEnhancer
    
    enhancer = ImageEnhancer()
    segmenter = FieldSegmenter()
    line_detector = LineDetector()
    
    enhanced = enhancer.enhance(image)
    field_mask = segmenter.segment_field(enhanced)
    lines = line_detector.detect(enhanced, mask=field_mask)
    
    # Visualize
    output = image.copy()
    
    # Draw field mask overlay
    overlay = image.copy()
    overlay[field_mask > 0] = [0, 255, 0]
    output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
    
    # Draw lines
    for x1, y1, x2, y2 in lines:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # Add status text
    status_color = (0, 255, 0) if result['field_detection']['status'] == 'success' else (0, 165, 255)
    cv2.putText(output, f"Status: {result['field_detection']['status'].upper()}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(output, f"Field: {result['field_detection']['field_percentage']:.1f}%", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(output, f"Lines: {result['field_detection']['lines_detected']}", 
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show result
    print("\nDisplaying result... (Press any key to close)")
    cv2.imshow("Atlas Detection Result", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save output
    output_path = Path("output") / "test_result.jpg"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), output)
    print(f"\n[OK] Result saved to: {output_path}")


if __name__ == "__main__":
    main()
