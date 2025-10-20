"""I/O handling for video, images, and JSON output."""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class VideoReader:
    """Read video frames."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def read_frame(self, frame_number: Optional[int] = None) -> Optional[np.ndarray]:
        """Read a specific frame or next frame."""
        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release video capture."""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class JSONWriter:
    """Write detection results to JSON."""
    
    @staticmethod
    def save_results(output_dict: Dict, output_path: str, indent: int = 2):
        """Save results to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=indent)
    
    @staticmethod
    def load_results(input_path: str) -> Dict:
        """Load results from JSON file."""
        with open(input_path, 'r') as f:
            return json.load(f)


def save_image(image: np.ndarray, output_path: str):
    """Save image to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load image from file."""
    return cv2.imread(image_path)
