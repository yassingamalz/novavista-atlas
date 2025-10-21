"""
Headless video processing - saves output without GUI
Perfect for batch processing or remote servers
"""

import cv2
import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent))

from atlas.core import AtlasProcessor


def main():
    """Process video without GUI."""
    
    if len(sys.argv) < 2:
        print("Usage: python test_video_headless.py <path_to_video> [sample_rate]")
        print("\nExample:")
        print("  python test_video_headless.py test_data/videos/match.mp4")
        print("  python test_video_headless.py test_data/videos/match.mp4 30")
        sys.exit(1)
    
    video_path = sys.argv[1]
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    if not Path(video_path).exists():
        print(f"[X] Error: Video not found at '{video_path}'")
        sys.exit(1)
    
    print("=" * 60)
    print("NovaVista Atlas - Headless Video Processing")
    print("=" * 60)
    print(f"Input: {video_path}")
    print(f"Sample Rate: Every {sample_rate} frames\n")
    
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
    
    # Setup output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "video_result.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps/sample_rate, (width, height))
    
    print(f"[OK] Output will be saved to: {output_path}")
    
    # Initialize processor
    processor = AtlasProcessor()
    
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
        "total_circles": 0,
        "frame_results": []
    }
    
    print("\nProcessing video...")
    print("-" * 60)
    
    frame_number = 0
    start_time = time.time()
    
    while True:
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
            
            # Store frame result
            stats["frame_results"].append({
                "frame": frame_number,
                "status": status,
                "field_percentage": result['field_detection']['field_percentage'],
                "lines": result['field_detection']['lines_detected'],
                "circles": result['field_detection']['circles_detected'],
                "processing_time": result['processing_metadata']['processing_time_ms']
            })
            
            # Print progress
            progress = (frame_number / total_frames) * 100
            avg_time = stats["total_time"] / stats["processed"]
            print(f"[{progress:5.1f}%] Frame {frame_number:5d} | "
                  f"Status: {status:8s} | "
                  f"Field: {result['field_detection']['field_percentage']:5.1f}% | "
                  f"Lines: {result['field_detection']['lines_detected']:3d} | "
                  f"Circles: {result['field_detection']['circles_detected']:3d} | "
                  f"Time: {result['processing_metadata']['processing_time_ms']:6.0f}ms")
            
            # Create visualization
            output = frame.copy()
            
            # Process for visualization
            enhanced = enhancer.enhance(frame)
            field_mask = segmenter.segment_field(enhanced)
            lines = line_detector.detect(enhanced, mask=field_mask)
            circles = circle_detector.detect(enhanced, mask=field_mask)
            
            # Draw field mask overlay
            overlay = frame.copy()
            overlay[field_mask > 0] = [0, 255, 0]
            output = cv2.addWeighted(output, 0.85, overlay, 0.15, 0)
            
            # Draw lines
            for x1, y1, x2, y2 in lines[:50]:
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw circles
            for x, y, r in circles[:20]:
                cv2.circle(output, (x, y), r, (255, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (255, 255, 0), -1)
            
            # Add info overlay
            status_color = (0, 255, 0) if status == 'success' else (0, 165, 255) if status == 'partial' else (0, 0, 255)
            
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
            
            # Write frame
            writer.write(output)
        
        frame_number += 1
    
    # Cleanup
    cap.release()
    writer.release()
    
    elapsed_time = time.time() - start_time
    
    # Save JSON results
    json_path = output_dir / "video_analysis.json"
    with open(json_path, 'w') as f:
        json.dump({
            "input_video": str(video_path),
            "processing_stats": {
                "total_time_seconds": elapsed_time,
                "frames_processed": stats['processed'],
                "success_rate": stats['success']/stats['processed']*100,
                "average_lines": stats['total_lines']/stats['processed'],
                "average_circles": stats['total_circles']/stats['processed'],
                "average_processing_time_ms": stats['total_time']/stats['processed']
            },
            "frame_by_frame": stats["frame_results"]
        }, f, indent=2)
    
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
    print(f"\n[OK] Output video saved to: {output_path}")
    print(f"[OK] Analysis JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
