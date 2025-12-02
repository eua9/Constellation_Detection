#!/usr/bin/env python3
"""
Demo script for star detection with optional visualization mode.

This script demonstrates the star detection pipeline and provides
a CLI flag to enable visualization of detected stars.
"""

import argparse
import numpy as np
from star_detection import detect_stars, visualize_detections
from synthetic_data import generate_synthetic_image, get_sample_templates


def main():
    """
    Main function to run star detection demo.
    """
    parser = argparse.ArgumentParser(
        description='Star Detection Demo - Detect stars in synthetic images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run detection without visualization
  python demo_detection.py
  
  # Run detection with visualization
  python demo_detection.py --visualize
  
  # Save visualization to file
  python demo_detection.py --visualize --save-vis output.png
  
  # Use custom detection threshold
  python demo_detection.py --threshold 0.05 --visualize
        """
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization mode to overlay detected stars on image'
    )
    
    parser.add_argument(
        '--save-vis',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to save visualization image (only used with --visualize)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='Intensity threshold for blob detection (default: 0.01)'
    )
    
    parser.add_argument(
        '--min-sigma',
        type=float,
        default=0.5,
        help='Minimum blob size (sigma) (default: 0.5)'
    )
    
    parser.add_argument(
        '--max-sigma',
        type=float,
        default=5.0,
        help='Maximum blob size (sigma) (default: 5.0)'
    )
    
    parser.add_argument(
        '--template',
        type=str,
        default='Big Dipper',
        choices=['Big Dipper', 'Orion'],
        help='Constellation template to use for synthetic image (default: Big Dipper)'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=('WIDTH', 'HEIGHT'),
        help='Size of synthetic image (default: 512 512)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=10.0,
        help='Scale factor for synthetic image (default: 10.0)'
    )
    
    parser.add_argument(
        '--rotation',
        type=float,
        default=0.0,
        help='Rotation angle in degrees (default: 0.0)'
    )
    
    parser.add_argument(
        '--noise',
        type=float,
        default=0.0,
        help='Noise level for star positions (default: 0.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("Star Detection Demo")
    print("=" * 60)
    
    # Load template
    templates = get_sample_templates()
    if args.template not in templates:
        print(f"Error: Template '{args.template}' not found.")
        print(f"Available templates: {list(templates.keys())}")
        return 1
    
    template = templates[args.template]
    print(f"\nUsing template: {args.template}")
    print(f"Template has {len(template)} stars")
    
    # Generate synthetic image
    print(f"\nGenerating synthetic image...")
    print(f"  Size: {args.image_size[0]}x{args.image_size[1]}")
    print(f"  Scale: {args.scale}")
    print(f"  Rotation: {args.rotation}°")
    print(f"  Noise level: {args.noise}")
    
    synthetic_image, transformed_points = generate_synthetic_image(
        template,
        image_size=tuple(args.image_size),
        rotation_angle=args.rotation,
        scale_factor=args.scale,
        translation=(0, 0),
        noise_level=args.noise,
        random_seed=args.seed
    )
    
    print(f"Generated image shape: {synthetic_image.shape}")
    
    # Configure detection parameters
    detection_config = {
        'intensity_threshold': args.threshold,
        'min_sigma': args.min_sigma,
        'max_sigma': args.max_sigma
    }
    
    print(f"\nDetecting stars...")
    print(f"  Threshold: {args.threshold}")
    print(f"  Sigma range: [{args.min_sigma}, {args.max_sigma}]")
    
    # Detect stars
    detected_centroids = detect_stars(synthetic_image, config=detection_config)
    
    print(f"\nResults:")
    print(f"  Stars detected: {len(detected_centroids)}")
    print(f"  Expected stars: {len(template)}")
    
    if len(detected_centroids) > 0:
        print(f"\n  First few detections:")
        for i, (x, y) in enumerate(detected_centroids[:5]):
            print(f"    Star {i+1}: ({x:.2f}, {y:.2f})")
        if len(detected_centroids) > 5:
            print(f"    ... and {len(detected_centroids) - 5} more")
    else:
        print("\n  Warning: No stars detected!")
        print("  Try adjusting --threshold or --min-sigma parameters")
    
    # Visualization
    if args.visualize:
        print(f"\nCreating visualization...")
        save_path = args.save_vis if args.save_vis else None
        
        visualize_detections(
            synthetic_image,
            detected_centroids,
            save_path=save_path
        )
        
        if save_path:
            print(f"Visualization saved to: {save_path}")
        else:
            print("Visualization displayed (close window to continue)")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

