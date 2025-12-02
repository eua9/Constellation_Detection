"""
Main Entry Point for Constellation Detection System

This script demonstrates the basic pipeline:
1. Star detection
2. Normalization
3. Template matching
"""

import numpy as np
import matplotlib.pyplot as plt

from star_detection import detect_stars, visualize_detection
from normalization import normalize_star_set
from matching import match_constellation
from synthetic_data import generate_synthetic_image, get_sample_templates
from evaluate import evaluate_on_synthetic_data


def main():
    """
    Main demo function demonstrating the constellation detection pipeline.
    """
    print("=" * 60)
    print("Constellation Detection System - Demo")
    print("=" * 60)
    
    # Load sample templates
    templates = get_sample_templates()
    print(f"\nLoaded {len(templates)} templates: {list(templates.keys())}")
    
    # Normalize templates (they should be normalized for matching)
    normalized_templates = {}
    for name, points in templates.items():
        normalized_templates[name] = normalize_star_set(points)
    
    # Generate a test synthetic image
    print("\nGenerating synthetic test image...")
    test_template_name = "Big Dipper"
    test_template = templates[test_template_name]
    
    # Create synthetic image with some transformations
    synthetic_image, transformed_points = generate_synthetic_image(
        test_template,
        image_size=(512, 512),
        rotation_angle=45.0,
        scale_factor=10.0,
        translation=(20, -10),
        noise_level=1.0,
        random_seed=42
    )
    
    print(f"Generated synthetic image based on '{test_template_name}' template")
    print(f"Image size: {synthetic_image.shape}")
    print(f"Number of stars in template: {len(test_template)}")
    
    # Detect stars in the synthetic image
    print("\nDetecting stars in image...")
    detection_config = {'intensity_threshold': 0.01}
    detected_centroids = detect_stars(synthetic_image, config=detection_config)
    print(f"Detected {len(detected_centroids)} stars")
    
    # Normalize detected stars
    print("\nNormalizing detected stars...")
    normalized_query = normalize_star_set(detected_centroids)
    print(f"Normalized {len(normalized_query)} star positions")
    
    # Match to templates
    print("\nMatching to templates...")
    best_match, score = match_constellation(
        normalized_query,
        normalized_templates,
        method='ssd'
    )
    
    print(f"\nResults:")
    print(f"  Best match: {best_match}")
    print(f"  SSD score: {score:.4f}")
    print(f"  Ground truth: {test_template_name}")
    print(f"  Correct: {best_match == test_template_name}")
    
    # Visualize results
    print("\nCreating visualization...")
    vis_image = visualize_detection(synthetic_image, detected_centroids)
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(synthetic_image, cmap='gray')
    plt.title('Original Synthetic Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(vis_image)
    plt.title(f'Detected Stars (Match: {best_match})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_result.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'demo_result.png'")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

