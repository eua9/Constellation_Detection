import cv2
import json
from pathlib import Path
from star_detection import detect_stars

# Check one image
image_file = 'test_synthetic_output/constellation_0000.png'
metadata_file = 'test_synthetic_output/constellation_0000_metadata.json'

with open(metadata_file, 'r') as f:
    metadata = json.load(f)

print(f"Ground truth: {metadata['constellation_name']}")
print(f"Expected stars (constellation + clutter): {metadata['num_stars']}")

image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
config = {'intensity_threshold': 0.05, 'min_sigma': 1.0, 'max_sigma': 15.0}
detected = detect_stars(image, config=config)

print(f"Detected stars: {len(detected)}")
print(f"Template 'Line' should have: 3 stars")
print(f"Clutter added: 5 stars")
print(f"Total expected: 8 stars")
