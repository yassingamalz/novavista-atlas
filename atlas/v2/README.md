# Atlas v2 - Sportlight Integration

Field detection and camera calibration using Sportlight (SoccerNet Challenge 2023 1st Place).

## Quick Start

```python
from atlas.v2 import Pipeline

# Initialize
pipeline = Pipeline()

# Process frame
import cv2
frame = cv2.imread('match_frame.jpg')
result = pipeline.process_frame(frame, visualize=True)

if result:
    mapper = result['mapper']
    # Convert image point to field coordinates
    field_x, field_y = mapper.image_to_field(640, 480)
    print(f"Position: {field_x:.1f}m, {field_y:.1f}m")
```

## Current Status

**Phase 0:** Structure created with placeholder implementations.

**Next:** Clone Sportlight repository and integrate real detection models.

See `STATUS.md` and `IMPLEMENTATION_PLAN_SPORTLIGHT.md` for details.

## Modules

- **detection**: Field keypoint & line detection (Sportlight-based)
- **calibration**: Automatic camera calibration (DLT algorithm)
- **coordinates**: Image ↔ field coordinate mapping
- **pipeline**: End-to-end processing pipeline
