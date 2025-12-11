#!/usr/bin/env python3
"""
Demo script for Stage 3: Template Matching Algorithms

This script demonstrates all three matching algorithms (SSD, Hausdorff, RANSAC) by:
1. Loading images and detecting stars
2. Normalizing the detected stars
3. Running each matching algorithm separately
4. Visualizing query points vs matched template for each method
"""

import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from star_detection import detect_stars
from normalization import normalize_star_points
from matching import (
    match_constellation_ssd,
    match_constellation_hausdorff,
    match_constellation_ransac,
    compute_ransac_score,
    apply_similarity_transform
)
from templates import load_templates


def process_image_for_matching_demo(image_path: Path, detection_config: dict):
    """
    Process an image: detect stars and normalize.
    
    Returns:
        Tuple of (original_centroids, normalized_centroids, image)
    """
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    
    # Detect stars
    detected_centroids = detect_stars(image, config=detection_config)
    
    if len(detected_centroids) == 0:
        return None, None, None
    
    # Normalize centroids
    normalized_centroids = normalize_star_points(detected_centroids)
    
    return detected_centroids, normalized_centroids, image


def create_matching_visualization(
    query_points: np.ndarray,
    matched_template_name: str,
    matched_template_points: np.ndarray,
    ssd_result: tuple,
    hausdorff_result: tuple,
    ransac_result: tuple,
    image_filename: str,
    output_path: Path
):
    """
    Create a comprehensive visualization showing all three matching methods.
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Create 3 subplots for the three methods
    ax1 = plt.subplot(1, 3, 1)  # SSD
    ax2 = plt.subplot(1, 3, 2)  # Hausdorff
    ax3 = plt.subplot(1, 3, 3)  # RANSAC
    
    # Common settings
    query_color = 'red'
    template_color = 'blue'
    query_marker = 'o'
    template_marker = 's'
    query_size = 150
    template_size = 120
    
    # Plot 1: SSD Method
    ax1.scatter(
        query_points[:, 0],
        query_points[:, 1],
        c=query_color,
        s=query_size,
        marker=query_marker,
        edgecolors='black',
        linewidths=2,
        label='Query Points',
        zorder=3,
        alpha=0.8
    )
    
    ax1.scatter(
        matched_template_points[:, 0],
        matched_template_points[:, 1],
        c=template_color,
        s=template_size,
        marker=template_marker,
        edgecolors='black',
        linewidths=2,
        label=f'Template: {matched_template_name}',
        zorder=2,
        alpha=0.8
    )
    
    # Draw lines from query to nearest template point (for visualization)
    for q_point in query_points:
        distances = np.linalg.norm(matched_template_points - q_point, axis=1)
        nearest_idx = np.argmin(distances)
        ax1.plot(
            [q_point[0], matched_template_points[nearest_idx, 0]],
            [q_point[1], matched_template_points[nearest_idx, 1]],
            'gray',
            linestyle='--',
            alpha=0.3,
            linewidth=1,
            zorder=1
        )
    
    ax1.set_title(f'SSD Method\nMatch: {ssd_result[0]}\nScore: {ssd_result[1]:.4f}', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Normalized X', fontsize=10)
    ax1.set_ylabel('Normalized Y', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    ax1.set_aspect('equal', adjustable='box')
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Plot 2: Hausdorff Method
    ax2.scatter(
        query_points[:, 0],
        query_points[:, 1],
        c=query_color,
        s=query_size,
        marker=query_marker,
        edgecolors='black',
        linewidths=2,
        label='Query Points',
        zorder=3,
        alpha=0.8
    )
    
    ax2.scatter(
        matched_template_points[:, 0],
        matched_template_points[:, 1],
        c=template_color,
        s=template_size,
        marker=template_marker,
        edgecolors='black',
        linewidths=2,
        label=f'Template: {matched_template_name}',
        zorder=2,
        alpha=0.8
    )
    
    # Draw lines from query to nearest template point
    for q_point in query_points:
        distances = np.linalg.norm(matched_template_points - q_point, axis=1)
        nearest_idx = np.argmin(distances)
        ax2.plot(
            [q_point[0], matched_template_points[nearest_idx, 0]],
            [q_point[1], matched_template_points[nearest_idx, 1]],
            'gray',
            linestyle='--',
            alpha=0.3,
            linewidth=1,
            zorder=1
        )
    
    ax2.set_title(f'Hausdorff Method\nMatch: {hausdorff_result[0]}\nScore: {hausdorff_result[1]:.4f}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Normalized X', fontsize=10)
    ax2.set_ylabel('Normalized Y', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    ax2.set_aspect('equal', adjustable='box')
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Plot 3: RANSAC Method
    ax3.scatter(
        query_points[:, 0],
        query_points[:, 1],
        c=query_color,
        s=query_size,
        marker=query_marker,
        edgecolors='black',
        linewidths=2,
        label='Query Points',
        zorder=3,
        alpha=0.8
    )
    
    ax3.scatter(
        matched_template_points[:, 0],
        matched_template_points[:, 1],
        c=template_color,
        s=template_size,
        marker=template_marker,
        edgecolors='black',
        linewidths=2,
        label=f'Template: {matched_template_name}',
        zorder=2,
        alpha=0.8
    )
    
    # For RANSAC, show inliers (points that match well)
    if ransac_result[0] is not None:
        # Compute RANSAC score to get inlier information
        score, num_inliers, transform = compute_ransac_score(
            query_points,
            matched_template_points,
            inlier_threshold=0.3,
            max_iterations=2000,
            min_inliers=2
        )
        
        # Apply transformation and find inliers
        transformed_query = apply_similarity_transform(
            query_points, transform[0], transform[1], transform[2]
        )
        distances = np.sqrt(
            ((transformed_query[:, np.newaxis, :] - 
              matched_template_points[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        min_distances = np.min(distances, axis=1)
        inlier_mask = min_distances <= 0.3
        
        # Highlight inliers
        if np.any(inlier_mask):
            ax3.scatter(
                query_points[inlier_mask, 0],
                query_points[inlier_mask, 1],
                c='green',
                s=query_size * 1.2,
                marker='*',
                edgecolors='black',
                linewidths=2,
                label=f'Inliers ({np.sum(inlier_mask)})',
                zorder=4,
                alpha=0.9
            )
        
        # Draw lines for inliers
        for i, q_point in enumerate(query_points):
            if inlier_mask[i]:
                distances = np.linalg.norm(matched_template_points - transformed_query[i], axis=1)
                nearest_idx = np.argmin(distances)
                ax3.plot(
                    [q_point[0], matched_template_points[nearest_idx, 0]],
                    [q_point[1], matched_template_points[nearest_idx, 1]],
                    'green',
                    linestyle='-',
                    alpha=0.5,
                    linewidth=2,
                    zorder=1
                )
    
    ax3.set_title(f'RANSAC Method\nMatch: {ransac_result[0]}\nScore: {ransac_result[1]:.4f}', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Normalized X', fontsize=10)
    ax3.set_ylabel('Normalized Y', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=9)
    ax3.set_aspect('equal', adjustable='box')
    ax3.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Add overall title
    fig.suptitle(f'Matching Algorithms Comparison: {image_filename}', 
                  fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = f"{Path(image_filename).stem}_matching_comparison.png"
    output_file = output_path / output_filename
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def main():
    """Main function to run matching algorithms demo."""
    parser = argparse.ArgumentParser(
        description='Demo Stage 3: Template Matching Algorithms (SSD, Hausdorff, RANSAC)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python demo_matching.py
  
  # Specify custom paths
  python demo_matching.py --input test_synthetic_output --output matching_demo_output
  
  # Process only first 3 images
  python demo_matching.py --max-images 3
  
  # Use specific detection threshold
  python demo_matching.py --threshold 0.05
        """
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
        default='matching_demo_output',
        help='Output directory for visualizations (default: matching_demo_output)'
    )
    
    parser.add_argument(
        '--templates',
        type=str,
        default='templates_config.json',
        help='Path to template configuration file (default: templates_config.json)'
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
    
    parser.add_argument(
        '--ransac-threshold',
        type=float,
        default=0.3,
        help='Inlier threshold for RANSAC (default: 0.3)'
    )
    
    parser.add_argument(
        '--ransac-iterations',
        type=int,
        default=2000,
        help='Max iterations for RANSAC (default: 2000)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Stage 3: Template Matching Algorithms Demo")
    print("=" * 70)
    
    # Load templates
    print(f"\nLoading templates from: {args.templates}")
    try:
        templates = load_templates(args.templates)
        print(f"Loaded {len(templates)} templates: {list(templates.keys())}")
    except Exception as e:
        print(f"Error loading templates: {e}")
        return 1
    
    # Setup input and output directories
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions and f.is_file()
    ])
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {input_dir}")
        return 1
    
    print(f"Found {len(image_files)} images")
    
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
    max_images = args.max_images if args.max_images else len(image_files)
    
    for i, image_file in enumerate(image_files[:max_images]):
        print(f"\n[{i+1}/{max_images}] Processing {image_file.name}...")
        
        # Process image
        original_centroids, normalized_centroids, image = process_image_for_matching_demo(
            image_file,
            detection_config
        )
        
        if normalized_centroids is None:
            print(f"  ⚠ Skipped (no stars detected)")
            continue
        
        print(f"  ✓ Detected {len(normalized_centroids)} stars")
        
        # Run all three matching algorithms
        print(f"  Running matching algorithms...")
        
        # SSD
        ssd_result = match_constellation_ssd(normalized_centroids, templates)
        print(f"    SSD:      Match = {ssd_result[0]}, Score = {ssd_result[1]:.4f}")
        
        # Hausdorff
        hausdorff_result = match_constellation_hausdorff(
            normalized_centroids, 
            templates,
            percentile=90.0
        )
        print(f"    Hausdorff: Match = {hausdorff_result[0]}, Score = {hausdorff_result[1]:.4f}")
        
        # RANSAC
        ransac_result = match_constellation_ransac(
            normalized_centroids,
            templates,
            inlier_threshold=args.ransac_threshold,
            max_iterations=args.ransac_iterations,
            min_inliers=2
        )
        print(f"    RANSAC:   Match = {ransac_result[0]}, Score = {ransac_result[1]:.4f}")
        
        # Get the matched template (use SSD result as reference, or best match)
        matched_template_name = ssd_result[0]
        if matched_template_name is None:
            matched_template_name = hausdorff_result[0] or ransac_result[0]
        
        if matched_template_name is None:
            print(f"  ⚠ No match found, skipping visualization")
            continue
        
        matched_template_points = templates[matched_template_name]
        
        # Create visualization
        output_file = create_matching_visualization(
            normalized_centroids,
            matched_template_name,
            matched_template_points,
            ssd_result,
            hausdorff_result,
            ransac_result,
            image_file.name,
            output_dir
        )
        
        print(f"  ✓ Visualization saved: {output_file.name}")
        
        processed_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Processed: {processed_count} images")
    print(f"Visualizations saved to: {output_dir}")
    print("\nMatching algorithms compared:")
    print("  ✓ SSD (Sum of Squared Differences) - Baseline method")
    print("  ✓ Hausdorff Distance - Robust to outliers")
    print("  ✓ RANSAC - Most robust, handles missing stars")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
