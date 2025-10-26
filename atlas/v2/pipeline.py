"""
Sportlight Field Detection Pipeline
Main pipeline for field detection, calibration, and coordinate mapping

Uses Sportlight (SoccerNet Challenge 2023 1st Place):
- 73.22% Accuracy
- 75.59% Completeness  
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import time

from detection.sportlight_detector import FieldDetector
from calibration.sportlight_calibrator import Calibrator
from coordinates.mapper import Mapper


class Pipeline:
    """
    Complete field detection and calibration pipeline
    Using Sportlight solution (SoccerNet Challenge 2023 1st place)
    """
    
    def __init__(self, model_path: str = None):
        self.detector = FieldDetector(model_path)
        self.calibrator = Calibrator()
        self.mapper = None
        
        self.stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'successful_calibrations': 0,
            'total_processing_time': 0.0
        }
        
    def process_frame(self, frame: np.ndarray, visualize: bool = False) -> Optional[Dict]:
        start_time = time.time()
        self.stats['total_frames'] += 1
        
        # Detect field features
        detection = self.detector.detect(frame)
        
        if detection is None or not detection['success']:
            return None
        
        self.stats['successful_detections'] += 1
        
        # Calibrate camera
        calibration = self.calibrator.calibrate(
            detection['keypoints'],
            detection['lines'],
            frame.shape[:2]
        )
        
        if calibration is None or not calibration['valid']:
            return None
        
        self.stats['successful_calibrations'] += 1
        
        # Create coordinate mapper
        self.mapper = Mapper(calibration['homography'])
        
        processing_time = (time.time() - start_time) * 1000
        self.stats['total_processing_time'] += processing_time
        
        result = {
            'detection': detection,
            'calibration': calibration,
            'mapper': self.mapper,
            'success': True,
            'processing_time_ms': processing_time
        }
        
        if visualize:
            result['visualization'] = self.visualize_results(frame, result)
        
        return result
    
    def process_video(self, video_path: str, output_path: str = None, 
                     sample_rate: int = 1, visualize: bool = False):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if visualize and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                result = self.process_frame(frame, visualize=visualize)
                
                if result is not None:
                    results.append({
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / fps,
                        'confidence': result['detection']['confidence'],
                        'processing_time_ms': result['processing_time_ms']
                    })
                    
                    if writer and 'visualization' in result:
                        writer.write(result['visualization'])
            
            frame_idx += 1
        
        cap.release()
        if writer:
            writer.release()
        
        self.print_summary()
        return results
    
    def visualize_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        vis_frame = self.detector.visualize(frame, results['detection'])
        self._draw_info_overlay(vis_frame, results)
        return vis_frame
    
    def _draw_info_overlay(self, frame: np.ndarray, results: Dict):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        conf = results['detection']['confidence']
        proc_time = results['processing_time_ms']
        
        info_lines = [
            f"Sportlight Pipeline",
            f"Confidence: {conf:.2f}",
            f"Processing: {proc_time:.1f}ms",
            f"Status: {'SUCCESS' if results['success'] else 'FAILED'}"
        ]
        
        y_offset = 40
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
    
    def print_summary(self):
        stats = self.stats
        total = stats['total_frames']
        
        if total == 0:
            return
        
        det_rate = stats['successful_detections'] / total * 100
        cal_rate = stats['successful_calibrations'] / total * 100
        avg_time = stats['total_processing_time'] / total
        
        print("\n" + "="*60)
        print("Processing Summary")
        print("="*60)
        print(f"Total frames:              {total}")
        print(f"Successful detections:     {stats['successful_detections']} ({det_rate:.1f}%)")
        print(f"Successful calibrations:   {stats['successful_calibrations']} ({cal_rate:.1f}%)")
        print(f"Average processing time:   {avg_time:.1f}ms per frame")
        print(f"Estimated FPS:             {1000/avg_time:.1f} FPS")
        print("="*60 + "\n")
    
    def get_stats(self) -> Dict:
        return self.stats.copy()


if __name__ == "__main__":
    pipeline = Pipeline()
    print("Pipeline ready. Use pipeline.process_frame(frame) or pipeline.process_video(path)")
