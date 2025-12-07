# Constellation Detection System

A classical computer vision pipeline for detecting and identifying constellations in astronomical images using star detection, normalization, and template matching.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**What this does:** Creates an isolated Python environment and installs all required libraries (numpy, opencv, scikit-image, matplotlib, etc.)

### 2. Verify Installation

```bash
# Test that all dependencies are installed correctly
python -c "import numpy, cv2, skimage, matplotlib; print('✓ All dependencies installed')"
```

**What this does:** Verifies that all required Python packages can be imported successfully.

---

## Running the Program

### Command 1: Run Demo (Single Image)

```bash
python main.py
```

**What this does:**
- Loads constellation templates (Big Dipper, Orion, Triangle, Line, L-Shape)
- Generates a synthetic test image with a constellation
- Detects stars in the image
- Normalizes the star pattern
- Matches it against templates using SSD algorithm
- Displays results and saves visualization to `demo_result.png`

**Expected output:** Console output showing detected constellation name, score, and a saved image file.

---

### Command 2: Generate Test Dataset

```bash
python test_synthetic_data.py
```

**What this does:**
- Generates 5 synthetic constellation images in `test_synthetic_output/`
- Each image has random rotation, scale, translation, and noise
- Saves images as PNG files and metadata as JSON files
- Creates ground truth labels for evaluation

**Output:** Creates `test_synthetic_output/` directory with:
- `constellation_0000.png` through `constellation_0004.png`
- Corresponding `constellation_XXXX_metadata.json` files with ground truth

---

### Command 3: Evaluate on Dataset (SSD Method)

```bash
python evaluate.py --dataset test_synthetic_output --output results.json
```

**What this does:**
- Processes all images in `test_synthetic_output/`
- Runs full pipeline (detection → normalization → matching) on each image
- Compares predictions to ground truth labels
- Computes accuracy and confusion matrix
- Saves results to `results.json`

**Expected output:**
```
Total images: 5
Valid predictions: 5
Correct predictions: 4
Overall Accuracy: 0.8000 (80.00%)

Confusion Matrix:
...
```

---

### Command 4: Compare All Matching Methods

```bash
python evaluate.py --dataset test_synthetic_output --method compare --output comparison.json
```

**What this does:**
- Evaluates the same dataset using three different matching algorithms:
  - **SSD** (Sum of Squared Differences) - baseline method
  - **Hausdorff Distance** - more robust to outliers
  - **RANSAC** - most robust, handles missing stars and clutter
- Compares accuracy and confusion matrices for all three methods
- Saves comparison results to `comparison.json`

**Expected output:** Side-by-side comparison table showing accuracy for each method and confusion matrices.

---

### Command 5: Batch Process Images

```bash
python batch_run.py --input test_synthetic_output --output batch_results
```

**What this does:**
- Processes all images in the input directory
- Runs the full pipeline on each image
- Saves predictions to `batch_results/batch_results.json` and `batch_results/batch_results.csv`
- Generates overlay images showing detected stars (saved in `batch_results/`)
- Reports performance metrics (processing time per image)

**Expected output:**
```
Performance Metrics:
  Total time: X.XXX seconds
  Average time per image: X.XXX seconds
  Status: ✓ PASS (under 5 seconds per 512×512 image)
```

---

## Command Options

### Evaluation Options

```bash
# Use Hausdorff matching method
python evaluate.py --dataset test_synthetic_output --method hausdorff

# Use RANSAC matching method
python evaluate.py --dataset test_synthetic_output --method ransac

# Adjust star detection sensitivity
python evaluate.py --dataset test_synthetic_output --detection-threshold 0.05

# Save results as CSV instead of JSON
python evaluate.py --dataset test_synthetic_output --output-format csv --output results.csv
```

### Batch Processing Options

```bash
# Skip generating overlay images
python batch_run.py --input test_synthetic_output --output batch_results --no-overlays

# Use custom template configuration
python batch_run.py --input test_synthetic_output --output batch_results --templates custom_templates.json
```

---

## Project Structure

```
Constellation_Detection/
├── main.py              # Demo script (Command 1)
├── test_synthetic_data.py  # Generate test data (Command 2)
├── evaluate.py          # Evaluation script (Commands 3-4)
├── batch_run.py         # Batch processing (Command 5)
├── star_detection.py     # Star detection module
├── normalization.py     # Pattern normalization module
├── matching.py          # Template matching (SSD, Hausdorff, RANSAC)
├── templates.py         # Template management
├── synthetic_data.py    # Synthetic data generation
├── templates_config.json # Template configuration
├── templates_data/      # Constellation template files
└── requirements.txt     # Python dependencies
```

---

## How It Works

1. **Star Detection**: Uses Laplacian of Gaussian blob detection to find stars in the image
2. **Normalization**: Normalizes star patterns using PCA to handle rotation, scale, and translation
3. **Template Matching**: Compares normalized pattern to known constellation templates using:
   - **SSD**: Simple distance-based matching (baseline)
   - **Hausdorff**: Maximum distance between point sets (more robust)
   - **RANSAC**: Robust estimation with transformation (handles outliers)

---

## Dependencies

- numpy==2.0.2
- scipy==1.13.1
- opencv-python==4.12.0.88
- scikit-image==0.24.0
- matplotlib==3.9.4
- pytest==8.4.2

---

## Example Workflow

```bash
# 1. Setup (one time)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate test data
python test_synthetic_data.py

# 3. Run evaluation
python evaluate.py --dataset test_synthetic_output --method compare --output results.json

# 4. View results
cat results.json
```

---

## Author

Ryan Mitchell  
CS 4337 - Intro to Computer Vision
