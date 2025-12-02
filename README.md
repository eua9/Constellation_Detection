# Constellation Detection System

A classical computer vision pipeline for detecting and identifying constellations in astronomical images using star detection, normalization, and template matching.

## Project Overview

This system implements a three-stage pipeline:
1. **Star Detection**: Detects star-like blobs in grayscale astronomical images and extracts centroids
2. **Normalization**: Normalizes star point sets for translation, rotation, and scale invariance using PCA
3. **Matching**: Matches detected star patterns to known constellation templates using SSD (Sum of Squared Differences)

## Project Structure

```
Constellation_Detection/
├── star_detection.py    # Star detection module
├── normalization.py     # Point set normalization module
├── matching.py          # Template matching module
├── synthetic_data.py    # Synthetic data generation module
├── evaluate.py          # Evaluation metrics module
├── main.py              # Main entry point and demo script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

On macOS/Linux:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

Run a simple test to verify everything is working:

```bash
python -c "import numpy, cv2, skimage, matplotlib; print('All dependencies installed successfully!')"
```

## Usage

### Running the Demo

The main demo script demonstrates the complete pipeline:

```bash
python main.py
```

This will:
- Load sample constellation templates
- Generate a synthetic test image
- Detect stars in the image
- Normalize the detected stars
- Match against templates
- Display results and save a visualization

### Module Usage Examples

#### Star Detection

```python
from star_detection import detect_stars, visualize_detection
import cv2

# Load image
image = cv2.imread('star_field.png', cv2.IMREAD_GRAYSCALE)

# Detect stars
centroids = detect_stars(image, threshold=0.01)

# Visualize
vis_image = visualize_detection(image, centroids)
```

#### Normalization

```python
from normalization import normalize_star_set

# Normalize detected star centroids
normalized_points = normalize_star_set(centroids)
```

#### Template Matching

```python
from matching import match_constellation

# Match normalized query points to templates
best_match, score = match_constellation(
    normalized_points,
    templates_dict,
    method='ssd'
)
```

#### Synthetic Data Generation

```python
from synthetic_data import generate_synthetic_image, get_sample_templates

# Get templates
templates = get_sample_templates()

# Generate synthetic image with transformations
image, points = generate_synthetic_image(
    templates["Big Dipper"],
    image_size=(512, 512),
    rotation_angle=45.0,
    scale_factor=10.0,
    noise_level=1.0,
    random_seed=42
)
```

## Dependencies

- **numpy**: Numerical operations
- **scipy**: Scientific computing
- **opencv-python**: Image processing
- **scikit-image**: Blob detection and image analysis
- **matplotlib**: Visualization
- **pytest**: Testing framework (optional, for future tests)

## Features

- **Star Detection**: Robust blob detection using Laplacian of Gaussian
- **PCA-based Normalization**: Invariant to translation, rotation, and scale
- **Baseline SSD Matching**: Simple and effective template matching
- **Synthetic Data Generation**: Generate test images with configurable transformations
- **Evaluation Metrics**: Accuracy and confusion matrix computation

## Future Enhancements

- Advanced matching methods (Hausdorff distance, RANSAC)
- Support for real astronomical images
- Batch processing capabilities
- Extended template library

## License

This project is part of a Computer Vision course project (CS 4337).

## Author

Ryan Mitchell

