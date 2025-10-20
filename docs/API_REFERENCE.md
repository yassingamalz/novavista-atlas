# NovaVista Atlas API Reference

## Core Components

### FieldSegmenter

```python
from atlas.preprocessing.field_segmentation import FieldSegmenter

segmenter = FieldSegmenter(
    lower_green=(35, 40, 40),
    upper_green=(85, 255, 255)
)

# Segment field from image
mask = segmenter.segment_field(image)
```

### LineDetector

```python
from atlas.detection.line_detector import LineDetector

detector = LineDetector(
    rho=1,
    theta=np.pi/180,
    threshold=100
)

# Detect lines
lines = detector.detect(image, mask=field_mask)
```

### HomographyCalculator

```python
from atlas.calibration.homography import HomographyCalculator

calculator = HomographyCalculator(
    ransac_threshold=5.0,
    max_iters=2000
)

# Calculate homography
H = calculator.calculate(src_points, dst_points)

# Transform points
transformed = calculator.transform_points(points, H)
```

### CoordinateTransformer

```python
from atlas.coordinates.transformer import CoordinateTransformer

transformer = CoordinateTransformer(homography_matrix)

# Convert pixel to world coordinates
world_coords = transformer.pixel_to_world((x, y))

# Convert world to pixel coordinates
pixel_coords = transformer.world_to_pixel((mx, my))
```

## Complete Pipeline Example

```python
from atlas.preprocessing.field_segmentation import FieldSegmenter
from atlas.detection.line_detector import LineDetector
from atlas.calibration.homography import HomographyCalculator
from atlas.coordinates.transformer import CoordinateTransformer

# Initialize components
segmenter = FieldSegmenter()
line_detector = LineDetector()
homography_calc = HomographyCalculator()

# Process frame
mask = segmenter.segment_field(frame)
lines = line_detector.detect(frame, mask)

# Calculate calibration
H = homography_calc.calculate(detected_points, template_points)
transformer = CoordinateTransformer(H)

# Use transformation
world_position = transformer.pixel_to_world((x_pixel, y_pixel))
```
