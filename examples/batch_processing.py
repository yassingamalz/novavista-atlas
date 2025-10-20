"""Batch processing example for multiple frames."""

import cv2
from pathlib import Path
from atlas.preprocessing.field_segmentation import FieldSegmenter
from atlas.detection.line_detector import LineDetector
from atlas.calibration.homography import HomographyCalculator
from atlas.utils.io_handler import save_image, JSONWriter
from atlas.utils.logger import setup_logger


def process_frame(image, segmenter, line_detector):
    """Process a single frame."""
    field_mask = segmenter.segment_field(image)
    lines = line_detector.detect(image, mask=field_mask)
    return {
        'lines_detected': len(lines),
        'field_coverage': (field_mask > 0).sum() / field_mask.size
    }


def main():
    """Process multiple frames in batch."""
    logger = setup_logger('batch_processor')
    
    # Initialize components
    segmenter = FieldSegmenter()
    line_detector = LineDetector()
    
    # Get all frames
    frames_dir = Path("test_data/frames")
    frame_files = list(frames_dir.glob("*.jpg"))
    
    logger.info(f"Processing {len(frame_files)} frames...")
    
    results = []
    for i, frame_path in enumerate(frame_files):
        logger.info(f"Processing frame {i+1}/{len(frame_files)}: {frame_path.name}")
        
        image = cv2.imread(str(frame_path))
        if image is None:
            logger.warning(f"Could not load {frame_path}")
            continue
        
        result = process_frame(image, segmenter, line_detector)
        result['frame_name'] = frame_path.name
        results.append(result)
    
    # Save results
    JSONWriter.save_results(results, "output/batch_results.json")
    logger.info("Batch processing complete!")


if __name__ == "__main__":
    main()
