"""
Test Atlas with video - Simple visual output
For end users who just want to see the result
"""

import cv2
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from atlas.core import AtlasProcessor


def main():
    """Test with video."""
    
    # Get video path
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <path_to_video> [sample_rate]")
        print("\nExample:")
        print("  python test_video.py test_data/videos/match.mp4")
        print("  python test_video.py test_data/videos/match.mp4 30  # Process every 30th frame")
        sys.exit(1)
    
    video_path = sys.argv[1]
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    # Check file exists
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found at '{video_path}'")
        sys.exit(1)
    
    print("=" * 60)
    print("NovaVista Atlas - Video Test")
    print("=" * 60)
    print(f"Input: {video_path}")
    print(f"Sample Rate: Every {sample_rate} frames\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Duration: {duration:.1f}s")
    print(f"  - Will process: ~{total_frames // sample_rate} frames\n")
    
    # Initialize processor
    processor = AtlasProcessor()
    
    # Stats
    stats = {
        "processed": 0,
        "success": 0,
        "partial": 0,
        "failed": 0
    }
    
    print("Processing video...")
    print("Press 'q' to quit, 's' to skip to next frame, any other key to continue")
    print("-" * 60)
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_number % sample_rate == 0:
            # Process frame
            result = processor.process_frame(frame)
            stats["processed"] += 1
            
            status = result['field_detection']['status']
            stats[status] = stats.get(status, 0) + 1
            
            # Print progress
            print(f"Frame {frame_number:5d} | "
                  f"Status: {status:10s} | "
                  f"Field: {result['field_detection']['field_percentage']:5.1f}% | "
                  f"Lines: {result['field_detection']['lines_detected']:3d} | "
                  f"Time: {result['processing_metadata']['processing_time_ms']:6.0f}ms")
            
            # Visualize if successful
            if status in ['success', 'partial']:
                from atlas.preprocessing.field_segmentation import FieldSegmenter
                from atlas.detection.line_detector import LineDetector
                from atlas.preprocessing.enhancement import ImageEnhancer
                
                enhancer = ImageEnhancer()
                segmenter = FieldSegmenter()
                line_detector = LineDetector()
                
                enhanced = enhancer.enhance(frame)
                field_mask = segmenter.segment_field(enhanced)
                lines = line_detector.detect(enhanced, mask=field_mask)
                
                # Create visualization
                output = frame.copy()
                
                # Draw field mask
                overlay = frame.copy()
                overlay[field_mask > 0] = [0, 255, 0]
                output = cv2.addWeighted(output, 0.8, overlay, 0.2, 0)
                
                # Draw lines
                for x1, y1, x2, y2 in lines[:50]:  # Limit to 50 lines
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Add info
                cv2.putText(output, f"Frame: {frame_number}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output, f"Status: {status}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(output, f"Field: {result['field_detection']['field_percentage']:.1f}%", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show
                cv2.imshow("Atlas Video Processing", output)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("\nStopped by user")
                    break
                elif key == ord('s'):
                    cv2.waitKey(0)  # Wait for next key
        
        frame_number += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("-" * 60)
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Frames Processed:  {stats['processed']}")
    print(f"Success:           {stats['success']} ({stats['success']/stats['processed']*100:.1f}%)")
    print(f"Partial:           {stats['partial']} ({stats['partial']/stats['processed']*100:.1f}%)")
    print(f"Failed:            {stats['failed']} ({stats['failed']/stats['processed']*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
