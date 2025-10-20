"""
Configuration management for Atlas
"""

DEFAULT_CONFIG = {
    "preprocessing": {
        "hsv_lower": [35, 40, 40],
        "hsv_upper": [85, 255, 255],
        "blur_kernel": 5,
        "morph_kernel": 5
    },
    "detection": {
        "canny_low": 50,
        "canny_high": 150,
        "hough_threshold": 50,
        "min_line_length": 100,
        "max_line_gap": 10
    },
    "calibration": {
        "ransac_threshold": 5.0,
        "ransac_iterations": 1000,
        "min_matches": 50
    },
    "pitch": {
        "length_meters": 105.0,
        "width_meters": 68.0
    }
}
