"""
Test Atlas with video - Enhanced visualization with output saving
"""

import cv2
import sys
from pathlib import Path
import time

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from atlas.core import AtlasProcessor


def main():
    """Test with video."""
    
    # Get video path
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <path_to_video> [sample_rate] [--save]")
        print("\nExample:")
        print("  python test_video.py test_data/videos/match.mp4")
        print("  python test_video.py test_data/videos/match.mp4 30")
        print("  python test_video.py test_data/videos/match.mp4 30 --save")
        print("\nOptions:")
        print("  sample_rate: Process every Nth frame (default: 30)")
        print("  --save: Save output video to output/video_result.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    sample_rate = 30
    save_output = False
    
    # Parse arguments
    for arg in sys.argv[2:]:
        if arg == "--save":
            save_output = True
        else:
            try:
                sample_rate = int(arg)
            except ValueError:
                pass
    
    # Check file exists
    if not Path(video_path).exists():
        print(f"[X] Error: Video not found at '{video_path}'")
        sys.exit(1)
    
    print("=" * 60)
    print("NovaVista Atlas - Video Test")
    print("=" * 60)
    print(f"Input: {video_path}")
    print(f"Sample Rate: Every {sample_rate} frames")
    print(f"Save Output: {'Yes' if save_output else 'No'}\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[X] Error: Cannot open video")
        sys.exit(1)
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Duration: {duration:.1f}s")
    print(f"  - Will process: ~{total_frames // sample_rate} frames\n")
    
    # Setup output video writer if needed
    writer = None
    if save_output:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "video_result.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps/sample_rate, (width, height))
        print(f"[OK] Output will be saved to: {output_path}\n")
    
    # Initialize processor
    processor = AtlasProcessor()
    
    # Import visualization modules
    from atlas.preprocessing.field_segmentation import FieldSegmenter
    from atlas.detection.line_detector import LineDetector
    from atlas.detection.circle_detector import CircleDetector
    from atlas.preprocessing.enhancement import ImageEnhancer
    
    enhancer = ImageEnhancer()
    segmenter = FieldSegmenter()
    line_detector = LineDetector()
    circle_detector = CircleDetector()
    
    # Stats
    stats = {
        "processed": 0,
        "success": 0,
        "partial": 0,
        "failed": 0,
        "total_time": 0,
        "total_lines": 0,
        "total_circles": 0
    }
    
    print("Processing video...")
    print("\nControls:")
    print("  q - Quit")
    print("  SPACE - Pause/Resume")
    print("  s - Save current frame to output/")
    print("-" * 60)
    
    frame_number = 0
    paused = False
    start_time = time.time()
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_number % sample_rate == 0:
                # Process frame
                result = processor.process_frame(frame)
                stats["processed"] += 1
                stats["total_time"] += result['processing_metadata']['processing_time_ms']
                
                status = result['field_detection']['status']
                stats[status] = stats.get(status, 0) + 1
                stats["total_lines"] += result['field_detection']['lines_detected']
                stats["total_circles"] += result['field_detection']['circles_detected']
                
                # Print progress
                avg_time = stats["total_time"] / stats["processed"]
                print(f"Frame {frame_number:5d} | "
                      f"Status: {status:8s} | "
                      f"Field: {result['field_detection']['field_percentage']:5.1f}% | "
                      f"Lines: {result['field_detection']['lines_detected']:3d} | "
                      f"Circles: {result['field_detection']['circles_detected']:3d} | "
                      f"Time: {result['processing_metadata']['processing_time_ms']:6.0f}ms | "
                      f"Avg: {avg_time:6.0f}ms")
                
                # Create visualization
                output = frame.copy()
                
                # Process for visualization
                enhanced = enhancer.enhance(frame)
                field_mask = segmenter.segment_field(enhanced)
                lines = line_detector.detect(enhanced, mask=field_mask)
                circles = circle_detector.detect(enhanced, mask=field_mask)
                
                # Draw field mask overlay (semi-transparent green)
                overlay = frame.copy()
                overlay[field_mask > 0] = [0, 255, 0]
                output = cv2.addWeighted(output, 0.85, overlay, 0.15, 0)
                
                # Draw lines (yellow)
                for x1, y1, x2, y2 in lines[:50]:  # Limit to 50 lines to avoid clutter
                    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Draw circles (cyan)
                for x, y, r in circles[:20]:  # Limit to 20 circles
                    cv2.circle(output, (x, y), r, (255, 255, 0), 2)
                    cv2.circle(output, (x, y), 2, (255, 255, 0), -1)
                
                # Status color
                status_color = (0, 255, 0) if status == 'success' else (0, 165, 255) if status == 'partial' else (0, 0, 255)
                
                # Add info overlay - top left
                info_bg = output[0:150, 0:400].copy()
                info_bg[:] = [0, 0, 0]
                output[0:150, 0:400] = cv2.addWeighted(output[0:150, 0:400], 0.3, info_bg, 0.7, 0)
                
                cv2.putText(output, f"Frame: {frame_number}/{total_frames}", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(output, f"Status: {status.upper()}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(output, f"Field: {result['field_detection']['field_percentage']:.1f}%", 
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(output, f"Lines: {result['field_detection']['lines_detected']}", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(output, f"Circles: {result['field_detection']['circles_detected']}", 
                           (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Add legend - bottom left
                legend_y = height - 80
                cv2.putText(output, "Field Mask", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(output, "Lines", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(output, "Circles", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show
                cv2.imshow("Atlas Video Processing", output)
                
                # Save frame if writer is active
                if writer is not None:
                    writer.write(output)
            
            frame_number += 1
        
        # Handle keyboard input
        key = cv2.waitKey(1 if not paused else 0)
        
        if key == ord('q'):
            print("\n[!] Stopped by user")
            break
        elif key == ord(' '):  # Space to pause/resume
            paused = not paused
            print(f"\n[!] {'Paused' if paused else 'Resumed'}")
        elif key == ord('s'):  # Save current frame
            if frame_number > 0:
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                save_path = output_dir / f"frame_{frame_number}.jpg"
                cv2.imwrite(str(save_path), output)
                print(f"\n[OK] Frame saved to: {save_path}")
    
    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("-" * 60)
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total Time:        {elapsed_time:.1f}s")
    print(f"Frames Processed:  {stats['processed']}")
    print(f"Success:           {stats['success']} ({stats['success']/stats['processed']*100:.1f}%)")
    print(f"Partial:           {stats['partial']} ({stats['partial']/stats['processed']*100:.1f}%)")
    print(f"Failed:            {stats['failed']} ({stats['failed']/stats['processed']*100:.1f}%)")
    print(f"\nAverage Lines:     {stats['total_lines']/stats['processed']:.1f}")
    print(f"Average Circles:   {stats['total_circles']/stats['processed']:.1f}")
    print(f"Avg Process Time:  {stats['total_time']/stats['processed']:.0f}ms")
    print(f"Processing FPS:    {stats['processed']/elapsed_time:.2f}")
    print("=" * 60)
    
    if save_output:
        print(f"\n[OK] Output video saved to: output/video_result.mp4")


if __name__ == "__main__":
    main()
