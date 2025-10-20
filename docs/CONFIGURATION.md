# Configuration Guide

## Configuration Files

Atlas uses YAML configuration files located in the `configs/` directory.

### Default Configuration

`configs/default_config.yaml` contains standard settings:

```yaml
preprocessing:
  gaussian_blur_kernel: 5
  hsv_green_lower: [35, 40, 40]
  hsv_green_upper: [85, 255, 255]
  morphology_kernel_size: 5

line_detection:
  threshold: 100
  min_line_length: 50
  max_line_gap: 10

pitch:
  length_meters: 105.0
  width_meters: 68.0
```

### Broadcast Configuration

Optimized for broadcast camera angles:

```yaml
broadcast_config:
  preprocessing:
    hsv_green_lower: [30, 35, 35]
    hsv_green_upper: [90, 255, 255]
    
  camera:
    view_type: "broadcast"
    expected_coverage: 0.85
```

### Tactical Configuration

Optimized for tactical/close-up views:

```yaml
tactical_config:
  camera:
    view_type: "tactical"
    expected_coverage: 0.5
    partial_pitch: true
```

## Loading Configuration

```python
import yaml

with open('configs/default_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use in components
segmenter = FieldSegmenter(
    lower_green=tuple(config['preprocessing']['hsv_green_lower']),
    upper_green=tuple(config['preprocessing']['hsv_green_upper'])
)
```

## Custom Configuration

Create your own configuration file:

```yaml
custom_config:
  stadium_name: "Example Stadium"
  
  preprocessing:
    # Adjust for specific lighting conditions
    hsv_green_lower: [32, 38, 38]
    hsv_green_upper: [88, 255, 255]
  
  pitch:
    # Non-standard pitch dimensions
    length_meters: 100.0
    width_meters: 64.0
```
