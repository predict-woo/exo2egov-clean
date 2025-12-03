# H2O Dataset Access Guide

This document explains how to access and use the H2O dataset for exocentric-to-egocentric view synthesis.

## Split Files

The dataset splits are defined in JSONL files located in the `splits_sampling/` directory:

- **Training set (multi-view)**: `splits_sampling/h2o_action_train_multi_exo.jsonl`
- **Test set (multi-view)**: `splits_sampling/h2o_action_test_multi_exo.jsonl`

### File Format

These files are in JSONL format (one JSON object per line). Each line represents a paired sample with 4 exocentric views (cam0-cam3) and 1 egocentric view (cam4):

```json
{"exo": ["subject1/h1/0/cam0/rgb/000000.png", "subject1/h1/0/cam1/rgb/000000.png", "subject1/h1/0/cam2/rgb/000000.png", "subject1/h1/0/cam3/rgb/000000.png"], "ego": "subject1/h1/0/cam4/rgb/000000.png"}
{"exo": ["subject1/h1/0/cam0/rgb/000005.png", "subject1/h1/0/cam1/rgb/000005.png", "subject1/h1/0/cam2/rgb/000005.png", "subject1/h1/0/cam3/rgb/000005.png"], "ego": "subject1/h1/0/cam4/rgb/000005.png"}
```

The dataset loader also supports the legacy single-view format (where `exo` is a string instead of a list), which will be replicated 4 times internally.

## Data Root

All image and camera data is stored under:

```
/cluster/project/cvg/data/H2O
```

To get the full path to an image, concatenate the root path with the relative path from the JSON file:

```python
root = "/cluster/project/cvg/data/H2O"
exo_path = os.path.join(root, sample["exo"])  # e.g., /cluster/project/cvg/data/H2O/subject1/h1/0/cam2/rgb/000000.png
ego_path = os.path.join(root, sample["ego"])  # e.g., /cluster/project/cvg/data/H2O/subject1/h1/0/cam4/rgb/000000.png
```

## Camera Intrinsics

Camera intrinsics are defined per camera and stored in a `cam_intrinsics.txt` file. The path can be inferred from the image path by navigating to the camera directory.

### Path Structure

For an image path like `subject1/h1/0/cam2/rgb/000000.png`, the intrinsics file is located at:

```
/cluster/project/cvg/data/H2O/subject1/h1/0/cam2/cam_intrinsics.txt
```

### Format

The intrinsics file contains 6 space-separated values:

```
fx fy cx cy width height
```

**Example:**
```
632.29296875 631.815673828125 640.5956219197251 367.3528703180782 1280 720
```

| Parameter | Description |
|-----------|-------------|
| `fx` | Focal length in x (pixels) |
| `fy` | Focal length in y (pixels) |
| `cx` | Principal point x coordinate |
| `cy` | Principal point y coordinate |
| `width` | Image width |
| `height` | Image height |

## Camera Extrinsics

Camera extrinsics (camera-to-world transformation) are stored per frame in individual text files.

### Path Structure

For an image path like `subject1/h1/0/cam2/rgb/000000.png`, the extrinsics file is located at:

```
/cluster/project/cvg/data/H2O/subject1/h1/0/cam2/cam_pose/000000.txt
```

The filename matches the frame number from the image path.

### Format

The extrinsics file contains 16 space-separated values representing a 4×4 transformation matrix in row-major order:

```
m00 m01 m02 m03 m10 m11 m12 m13 m20 m21 m22 m23 m30 m31 m32 m33
```

**Example:**
```
-0.9980796922 -0.0609574575 -0.0110313482 -0.0750193927 -0.0278716302 0.2828477634 0.9587601352 -0.7426801484 -0.0553233969 0.9572255435 -0.2840036253 0.5848526503 0.0000000000 0.0000000000 0.0000000000 1.0000000000
```

This corresponds to the matrix:

```
| m00  m01  m02  m03 |     | R R R tx |
| m10  m11  m12  m13 |  =  | R R R ty |
| m20  m21  m22  m23 |     | R R R tz |
| m30  m31  m32  m33 |     | 0 0 0 1  |
```

Where the 3×3 upper-left block is the rotation matrix and the rightmost column (m03, m13, m23) is the translation vector.

## Example: Loading a Sample

```python
import json
import numpy as np
from PIL import Image

ROOT = "/cluster/project/cvg/data/H2O"

def load_intrinsics(cam_dir):
    """Load camera intrinsics from cam_intrinsics.txt"""
    intrinsics_path = os.path.join(cam_dir, "cam_intrinsics.txt")
    with open(intrinsics_path, 'r') as f:
        values = f.read().strip().split()
    fx, fy, cx, cy, width, height = map(float, values)
    return {
        'fx': fx, 'fy': fy,
        'cx': cx, 'cy': cy,
        'width': int(width), 'height': int(height)
    }

def load_extrinsics(cam_dir, frame_id):
    """Load camera extrinsics (4x4 matrix) for a specific frame"""
    extrinsics_path = os.path.join(cam_dir, "cam_pose", f"{frame_id}.txt")
    with open(extrinsics_path, 'r') as f:
        values = list(map(float, f.read().strip().split()))
    return np.array(values).reshape(4, 4)

def parse_image_path(rel_path):
    """Extract camera directory and frame ID from relative image path"""
    # rel_path: subject1/h1/0/cam2/rgb/000000.png
    parts = rel_path.split('/')
    cam_dir = os.path.join(ROOT, *parts[:-2])  # subject1/h1/0/cam2
    frame_id = parts[-1].replace('.png', '')    # 000000
    return cam_dir, frame_id

# Load split file
with open("splits_sampling/h2o_action_train.json", 'r') as f:
    samples = [json.loads(line) for line in f]

# Load a sample
sample = samples[0]

# Load images
exo_image = Image.open(os.path.join(ROOT, sample["exo"]))
ego_image = Image.open(os.path.join(ROOT, sample["ego"]))

# Load camera parameters
exo_cam_dir, exo_frame = parse_image_path(sample["exo"])
ego_cam_dir, ego_frame = parse_image_path(sample["ego"])

exo_intrinsics = load_intrinsics(exo_cam_dir)
ego_intrinsics = load_intrinsics(ego_cam_dir)

exo_extrinsics = load_extrinsics(exo_cam_dir, exo_frame)
ego_extrinsics = load_extrinsics(ego_cam_dir, ego_frame)
```

## Directory Structure Summary

```
/cluster/project/cvg/data/H2O/
├── subject1/
│   ├── h1/
│   │   ├── 0/
│   │   │   ├── cam0/                    # Exocentric camera 0
│   │   │   │   ├── cam_intrinsics.txt   # Camera intrinsics
│   │   │   │   ├── cam_pose/
│   │   │   │   │   ├── 000000.txt       # Frame 0 extrinsics
│   │   │   │   │   ├── 000005.txt       # Frame 5 extrinsics
│   │   │   │   │   └── ...
│   │   │   │   └── rgb/
│   │   │   │       ├── 000000.png
│   │   │   │       ├── 000005.png
│   │   │   │       └── ...
│   │   │   ├── cam1/                    # Exocentric camera 1
│   │   │   │   └── ... (same structure)
│   │   │   ├── cam2/                    # Exocentric camera 2
│   │   │   │   └── ... (same structure)
│   │   │   ├── cam3/                    # Exocentric camera 3
│   │   │   │   └── ... (same structure)
│   │   │   ├── cam4/                    # Egocentric camera (head-mounted)
│   │   │   │   ├── cam_intrinsics.txt
│   │   │   │   ├── cam_pose/
│   │   │   │   │   └── ...
│   │   │   │   └── rgb/
│   │   │   │       └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

