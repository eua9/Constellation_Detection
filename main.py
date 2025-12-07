"""
Main Entry Point for Constellation Detection System

This script demonstrates the basic pipeline:
1. Star detection
2. Normalization
3. Template matching
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from star_detection import detect_stars, visualize_detection
from normalization import normalize_star_points, normalize_star_set
from matching import match_constellation_ssd, match_constellation
from templates import load_templates
from synthetic_data import generate_constellation_instance, render_star_image
from evaluate import evaluate_on_synthetic_data


def main():
    """
    Main demo function demonstrating the constellation detection pipeline.
    """
    print("=" * 60)
    print("Constellation Detection System - Demo")
    print("=" * 60)
    
    # Load templates
    templates = load_templates('templates_config.json')
    print(f"\nLoaded {len(templates)} templates: {list(templates.keys())}")
    
    # Templates are already normalized, so we can use them directly
    normalized_templates = templates
    
    # Generate a test synthetic image
    print("\nGenerating synthetic test image...")
    test_template_name = "Big Dipper"
    test_template = templates[test_template_name]
    
    # Create synthetic image with some transformations
    # First generate transformed coordinates
    params = {
        'rotation_range': None,  # Use specific angle instead
        'rotation_angle': 45.0,
        'scale_range': None,  # Use specific scale instead
        'scale_factor': 10.0,
        'translation_range': None,  # Use specific translation instead
        'translation': (20, -10),
        'noise_std': 1.0,
        'remove_prob': 0.0,  # Don't remove stars for demo
        'image_size': (512, 512),
        'random_seed': 42
    }
    transformed_points = generate_constellation_instance(test_template, params)
    
    # Then render the image
    synthetic_image = render_star_image(
        transformed_points,
        (512, 512),
        star_radius=2,
        noise_settings=None
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
    normalized_query = normalize_star_points(detected_centroids)
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


def run_pipeline(
    image: np.ndarray,
    template_config_path: str = "templates_config.json",
    detection_config: Optional[dict] = None,
    no_match_threshold: Optional[float] = None,
    visualize: bool = False
) -> Tuple[Optional[str], float, np.ndarray]:
    """
    Run the end-to-end constellation detection pipeline.
    
    Steps:
    1. Load image (passed as parameter)
    2. Detect stars (FR2)
    3. Normalize point set (FR3)
    4. Load templates (FR4)
    5. Run SSD-based matching (FR5)
    
    Args:
        image: Input grayscale or color image (numpy array)
        template_config_path: Path to template configuration file
        detection_config: Optional configuration dict for star detection
        no_match_threshold: Optional threshold to declare "no match" if all scores exceed it
        visualize: If True, return detected centroids for visualization
    
    Returns:
        Tuple of (best_match_name, score, detected_centroids)
        - best_match_name: Name of best matching constellation (None if no match)
        - score: SSD score of best match
        - detected_centroids: Array of detected star centroids
    """
    # Step 1: Image is already loaded (passed as parameter)
    
    # Step 2: Detect stars (FR2)
    if detection_config is None:
        detection_config = {'intensity_threshold': 0.01}
    
    detected_centroids = detect_stars(image, config=detection_config)
    
    if len(detected_centroids) == 0:
        print("Warning: No stars detected in image")
        return None, float('inf'), detected_centroids
    
    # Step 3: Normalize point set (FR3)
    normalized_query = normalize_star_points(detected_centroids)
    
    # Step 4: Load templates (FR4)
    templates = load_templates(template_config_path)
    
    if len(templates) == 0:
        raise ValueError(f"No templates found in {template_config_path}")
    
    # Step 5: Run SSD-based matching (FR5)
    best_match, score = match_constellation_ssd(
        normalized_query,
        templates,
        no_match_threshold=no_match_threshold
    )
    
    return best_match, score, detected_centroids


if __name__ == "__main__":
    main()

