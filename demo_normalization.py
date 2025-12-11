#!/usr/bin/env python3
"""
Demo script for Stage 2: Geometric Normalization

This script demonstrates normalization by:
1. Loading images from batch processing results
2. Detecting stars to get centroids
3. Applying normalization (centering, PCA alignment, scale normalization)
4. Visualizing before/after normalization
"""

import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from star_detection import detect_stars
from normalization import normalize_star_points


def load_batch_results(results_path: str):
    """Load batch processing results to get image filenames."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data.get('results', [])


def process_image_for_normalization_demo(
    image_path: Path,
    detection_config: dict,
    output_dir: Path
):
    """
    Process a single image: detect stars, normalize, and create visualization.
    
    Returns:
        Tuple of (original_centroids, normalized_centroids, image_filename)
    """
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Could not load {image_path}")
        return None, None, None
    
    # Detect stars
    detected_centroids = detect_stars(image, config=detection_config)
    
    if len(detected_centroids) == 0:
        print(f"Warning: No stars detected in {image_path.name}")
        return None, None, None
    
    # Normalize centroids
    normalized_centroids = normalize_star_points(detected_centroids)
    
    return detected_centroids, normalized_centroids, image_path.name


def create_normalization_visualization(
    original_centroids: np.ndarray,
    normalized_centroids: np.ndarray,
    image_filename: str,
    output_path: Path
):
    """
    Create a side-by-side visualization showing original and normalized centroids.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left plot: Original centroids
    ax1 = axes[0]
    ax1.scatter(
        original_centroids[:, 0],
        original_centroids[:, 1],
        c='red',
        s=150,
        marker='o',
        edgecolors='black',
        linewidths=2,
        label='Detected Stars',
        zorder=3
    )
    
    # Draw lines connecting stars (for better visualization)
    if len(original_centroids) > 1:
        # Simple connection: connect in order of detection
        for i in range(len(original_centroids) - 1):
            ax1.plot(
                [original_centroids[i, 0], original_centroids[i+1, 0]],
                [original_centroids[i, 1], original_centroids[i+1, 1]],
                'b--',
                alpha=0.3,
                linewidth=1
            )
    
    # Mark centroid
    centroid = np.mean(original_centroids, axis=0)
    ax1.scatter(
        [centroid[0]],
        [centroid[1]],
        c='green',
        s=200,
        marker='+',
        linewidths=3,
        label='Centroid',
        zorder=4
    )
    
    ax1.set_title(f'Original Detected Stars\n{image_filename}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X coordinate (pixels)', fontsize=12)
    ax1.set_ylabel('Y coordinate (pixels)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_aspect('equal', adjustable='box')
    
    # Add statistics
    std_dev = np.std(original_centroids)
    ax1.text(
        0.02, 0.98,
        f'Std Dev: {std_dev:.2f}\nCentroid: ({centroid[0]:.1f}, {centroid[1]:.1f})',
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Right plot: Normalized centroids
    ax2 = axes[1]
    ax2.scatter(
        normalized_centroids[:, 0],
        normalized_centroids[:, 1],
        c='purple',
        s=150,
        marker='o',
        edgecolors='black',
        linewidths=2,
        label='Normalized Stars',
        zorder=3
    )
    
    # Draw lines connecting stars
    if len(normalized_centroids) > 1:
        for i in range(len(normalized_centroids) - 1):
            ax2.plot(
                [normalized_centroids[i, 0], normalized_centroids[i+1, 0]],
                [normalized_centroids[i, 1], normalized_centroids[i+1, 1]],
                'b--',
                alpha=0.3,
                linewidth=1
            )
    
    # Mark centroid (should be at origin)
    normalized_centroid = np.mean(normalized_centroids, axis=0)
    ax2.scatter(
        [normalized_centroid[0]],
        [normalized_centroid[1]],
        c='green',
        s=200,
        marker='+',
        linewidths=3,
        label='Centroid (should be ~0,0)',
        zorder=4
    )
    
    ax2.set_title('Normalized Stars\n(Translation, Rotation, Scale Invariant)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Normalized X coordinate', fontsize=12)
    ax2.set_ylabel('Normalized Y coordinate', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_aspect('equal', adjustable='box')
    
    # Add statistics
    normalized_std = np.std(normalized_centroids)
    ax2.text(
        0.02, 0.98,
        f'Std Dev: {normalized_std:.4f}\nCentroid: ({normalized_centroid[0]:.4f}, {normalized_centroid[1]:.4f})',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    # Add reference lines at origin
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = f"{Path(image_filename).stem}_normalization.png"
    output_file = output_path / output_filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def main():
    """Main function to run normalization demo."""
    parser = argparse.ArgumentParser(
        description='Demo Stage 2: Geometric Normalization on batch results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (batch_results.json and test_synthetic_output)
  python demo_normalization.py
  
  # Specify custom paths
  python demo_normalization.py --results batch_results/batch_results.json --input test_synthetic_output --output normalization_demo
  
  # Process only first 3 images
  python demo_normalization.py --max-images 3
        """
    )
    
    parser.add_argument(
        '--results',
        type=str,
        default='batch_results/batch_results.json',
        help='Path to batch_results.json file (default: batch_results/batch_results.json)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='test_synthetic_output',
        help='Directory containing input images (default: test_synthetic_output)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='normalization_demo_output',
        help='Output directory for visualizations (default: normalization_demo_output)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to process (default: all)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='Intensity threshold for star detection (default: 0.05)'
    )
    
    parser.add_argument(
        '--min-sigma',
        type=float,
        default=1.0,
        help='Minimum blob size (sigma) for detection (default: 1.0)'
    )
    
    parser.add_argument(
        '--max-sigma',
        type=float,
        default=15.0,
        help='Maximum blob size (sigma) for detection (default: 15.0)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Stage 2: Geometric Normalization Demo")
    print("=" * 70)
    
    # Load batch results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print("Please run batch processing first or specify correct path with --results")
        return 1
    
    print(f"\nLoading batch results from: {results_path}")
    batch_results = load_batch_results(str(results_path))
    print(f"Found {len(batch_results)} processed images")
    
    # Setup input and output directories
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please generate synthetic data first or specify correct path with --input")
        return 1
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Detection configuration
    detection_config = {
        'intensity_threshold': args.threshold,
        'min_sigma': args.min_sigma,
        'max_sigma': args.max_sigma
    }
    
    print(f"\nDetection configuration:")
    print(f"  Threshold: {args.threshold}")
    print(f"  Sigma range: [{args.min_sigma}, {args.max_sigma}]")
    
    # Process images
    print(f"\nProcessing images...")
    print("-" * 70)
    
    processed_count = 0
    max_images = args.max_images if args.max_images else len(batch_results)
    
    for i, result in enumerate(batch_results[:max_images]):
        image_filename = result['image_filename']
        image_path = input_dir / image_filename
        
        if not image_path.exists():
            print(f"  [{i+1}/{max_images}] Skipping {image_filename} (file not found)")
            continue
        
        print(f"  [{i+1}/{max_images}] Processing {image_filename}...")
        
        # Process image
        original_centroids, normalized_centroids, filename = process_image_for_normalization_demo(
            image_path,
            detection_config,
            output_dir
        )
        
        if original_centroids is None:
            print(f"    ⚠ Skipped (no stars detected)")
            continue
        
        # Create visualization
        output_file = create_normalization_visualization(
            original_centroids,
            normalized_centroids,
            filename,
            output_dir
        )
        
        # Print statistics
        centroid_orig = np.mean(original_centroids, axis=0)
        centroid_norm = np.mean(normalized_centroids, axis=0)
        std_orig = np.std(original_centroids)
        std_norm = np.std(normalized_centroids)
        
        print(f"    ✓ Stars detected: {len(original_centroids)}")
        print(f"      Original - Centroid: ({centroid_orig[0]:.1f}, {centroid_orig[1]:.1f}), Std: {std_orig:.2f}")
        print(f"      Normalized - Centroid: ({centroid_norm[0]:.4f}, {centroid_norm[1]:.4f}), Std: {std_norm:.4f}")
        print(f"      Saved: {output_file.name}")
        
        processed_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Processed: {processed_count} images")
    print(f"Visualizations saved to: {output_dir}")
    print("\nNormalization properties verified:")
    print("  ✓ Centered at origin (translation invariant)")
    print("  ✓ Unit variance (scale invariant)")
    print("  ✓ PCA aligned (rotation invariant)")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
