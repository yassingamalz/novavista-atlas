"""Basic usage example for NovaVista Atlas."""

import cv2
from atlas.preprocessing.field_segmentation import FieldSegmenter
from atlas.detection.line_detector import LineDetector
from atlas.utils.visualization import draw_field_mask, draw_lines
from atlas.utils.io_handler import save_image


def main():
    """Run basic field detection pipeline."""
    # Load image
    image_path = "test_data/frames/sample_frame.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Segment field
    print("Segmenting field...")
    segmenter = FieldSegmenter()
    field_mask = segmenter.segment_field(image)
    
    # Detect lines
    print("Detecting lines...")
    line_detector = LineDetector()
    lines = line_detector.detect(image, mask=field_mask)
    print(f"Detected {len(lines)} lines")
    
    # Visualize results
    print("Creating visualization...")
    output = draw_field_mask(image, field_mask)
    output = draw_lines(output, lines)
    
    # Save output
    output_path = "output/basic_detection.jpg"
    save_image(output, output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
