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
├── star_detection.py      # Star detection module (FR2)
├── normalization.py       # Point set normalization module (FR3)
├── matching.py           # Template matching module (FR5)
├── templates.py           # Template management module (FR4)
├── synthetic_data.py      # Synthetic data generation module (FR7)
├── evaluate.py           # Evaluation metrics module (FR8)
├── batch_run.py          # Batch processing module (FR1.2)
├── main.py               # Main entry point and demo script
├── config.json           # Configuration file for parameters
├── config_loader.py      # Configuration loader utility
├── templates_config.json # Template configuration
├── templates_data/       # Template JSON files
├── requirements.txt      # Python dependencies (NFR3)
└── README.md            # This file
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

The `requirements.txt` file contains exact version numbers for reproducibility (NFR3):
- numpy==2.0.2
- scipy==1.13.1
- opencv-python==4.12.0.88
- scikit-image==0.24.0
- matplotlib==3.9.4
- pytest==8.4.2

### 4. Verify Installation

Run a simple test to verify everything is working:

```bash
python -c "import numpy, cv2, skimage, matplotlib; print('All dependencies installed successfully!')"
```

## Configuration

The system uses a JSON configuration file (`config.json`) for managing parameters. This ensures reproducible experiments (NFR3). The configuration includes:

- **paths**: File and directory paths
- **detection**: Star detection parameters (thresholds, blob sizes)
- **matching**: Template matching parameters
- **synthetic_data**: Synthetic data generation parameters
- **evaluation**: Evaluation output settings
- **batch_processing**: Batch processing options

You can modify `config.json` to adjust parameters without changing code. The `config_loader.py` module provides utilities for loading configuration.

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

### Generating Synthetic Data

To generate a synthetic dataset for testing and evaluation:

```bash
python -c "
from synthetic_data import generate_dataset
from templates import load_templates

# Load templates
templates = load_templates('templates_config.json')

# Configuration for synthetic data generation
config = {
    'image_size': (512, 512),
    'star_radius': 2,
    'rotation_range': (0, 360),
    'scale_range': (0.5, 2.0),
    'translation_range': ((-50, 50), (-50, 50)),
    'noise_std': 1.0,
    'remove_prob': 0.1,
    'clutter_count': 5,
    'noise_settings': {'background_noise': 5.0}
}

# Generate 100 synthetic images
generate_dataset(100, templates, config, 'synthetic_dataset', random_seed=42)
"
```

Or use the test script:

```bash
python test_synthetic_data.py
```

This generates a small test dataset in `test_synthetic_output/` with 5 images.

### Running the Baseline Pipeline

The baseline pipeline (detection → normalization → SSD matching) can be run on a single image:

```bash
python main.py
```

Or programmatically:

```python
from main import run_pipeline
import cv2

# Load image
image = cv2.imread('star_field.png', cv2.IMREAD_GRAYSCALE)

# Run pipeline
best_match, score, centroids = run_pipeline(
    image,
    template_config_path='templates_config.json',
    detection_config={'intensity_threshold': 0.01}
)

print(f"Detected constellation: {best_match} (score: {score:.4f})")
```

### Running Evaluation

To evaluate the system on a labeled synthetic dataset:

```bash
python evaluate.py --dataset test_synthetic_output --output evaluation_results.json
```

This will:
- Process all images in the dataset directory
- Run the full pipeline on each image
- Compare predictions to ground truth labels
- Compute accuracy and confusion matrix
- Save results to JSON format

Additional options:
- `--templates`: Path to template config (default: templates_config.json)
- `--output-format`: Choose 'json' or 'csv' (default: json)
- `--detection-threshold`: Star detection sensitivity (default: 0.01)
- `--no-match-threshold`: Threshold for declaring no match

Example output:
```
Total images: 100
Valid predictions: 95
Correct predictions: 87
Overall Accuracy: 0.9158 (91.58%)

Confusion Matrix:
...
```

### Batch Processing

To process a directory of images in batch mode:

```bash
python batch_run.py --input test_synthetic_output --output batch_results
```

This will:
- Process all images in the input directory
- Run the full pipeline on each image
- Save predictions to JSON and CSV
- Generate optional overlay visualizations
- Report performance metrics (NFR1: < 5s per 512×512 image)

Additional options:
- `--templates`: Path to template config
- `--detection-threshold`: Star detection sensitivity
- `--no-overlays`: Skip saving overlay images
- `--no-match-threshold`: Threshold for declaring no match

## Module Usage Examples

### Star Detection

```python
from star_detection import detect_stars, visualize_detection
import cv2

# Load image
image = cv2.imread('star_field.png', cv2.IMREAD_GRAYSCALE)

# Detect stars with custom configuration
detection_config = {
    'intensity_threshold': 0.01,
    'min_sigma': 1.0,
    'max_sigma': 30.0
}
centroids = detect_stars(image, config=detection_config)

# Visualize detected stars
vis_image = visualize_detection(image, centroids)
cv2.imwrite('detected_stars.png', vis_image)
```

### Normalization

```python
from normalization import normalize_star_points

# Normalize detected star centroids
# This removes translation, rotation, and scale differences
normalized_points = normalize_star_points(centroids)
```

### Template Matching

```python
from matching import match_constellation_ssd
from templates import load_templates

# Load templates
templates = load_templates('templates_config.json')

# Match normalized query points to templates
best_match, score = match_constellation_ssd(
    normalized_points,
    templates,
    no_match_threshold=None
)

print(f"Best match: {best_match}, Score: {score:.4f}")
```

### Synthetic Data Generation

```python
from synthetic_data import generate_constellation_instance, render_star_image
from templates import load_templates
import numpy as np

# Load a template
templates = load_templates('templates_config.json')
template_points = templates['Big Dipper']

# Generate transformed instance
params = {
    'rotation_range': (0, 360),
    'scale_range': (0.5, 2.0),
    'translation_range': ((-50, 50), (-50, 50)),
    'noise_std': 1.0,
    'remove_prob': 0.1,
    'image_size': (512, 512),
    'random_seed': 42
}
transformed_points = generate_constellation_instance(template_points, params)

# Render as image
image = render_star_image(transformed_points, (512, 512), star_radius=2)
```

## Dependencies

- **numpy==2.0.2**: Numerical operations
- **scipy==1.13.1**: Scientific computing
- **opencv-python==4.12.0.88**: Image processing
- **scikit-image==0.24.0**: Blob detection and image analysis
- **matplotlib==3.9.4**: Visualization
- **pytest==8.4.2**: Testing framework

## Features

- **Modular Structure (NFR2)**: Clean separation of concerns across modules
- **Comprehensive Docstrings (NFR2)**: All public functions documented with inputs, outputs, and behavior
- **Reproducible Setup (NFR3)**: Exact dependency versions and random seed support
- **Star Detection**: Robust blob detection using Laplacian of Gaussian
- **PCA-based Normalization**: Invariant to translation, rotation, and scale
- **Baseline SSD Matching**: Simple and effective template matching
- **Synthetic Data Generation**: Generate test images with configurable transformations
- **Evaluation Metrics**: Accuracy and confusion matrix computation
- **Batch Processing**: Process directories of images with performance monitoring
- **Configuration Management**: JSON-based configuration for reproducible experiments

## Performance Requirements

- **NFR1**: Processing a 512×512 image should take less than 5 seconds on a standard laptop
- The batch processing script (`batch_run.py`) includes timing instrumentation and reports performance metrics

## Testing

Run the test suite:

```bash
python test_synthetic_data.py
python test_star_detection.py
python test_normalization.py
python test_templates.py
```

## License

This project is part of a Computer Vision course project (CS 4337).

## Author

Ryan Mitchell
