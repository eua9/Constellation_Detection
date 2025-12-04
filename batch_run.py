"""
Batch Processing Module

This module provides batch processing functionality for running the constellation
detection pipeline on a directory of images, with timing instrumentation and
performance monitoring (NFR1).
"""

import numpy as np
import cv2
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import csv

from star_detection import detect_stars, visualize_detection
from normalization import normalize_star_points
from matching import match_constellation_ssd
from templates import load_templates


def process_batch(
    input_dir: str,
    output_dir: str,
    template_config_path: str = "templates_config.json",
    detection_config: Optional[dict] = None,
    save_overlays: bool = True,
    no_match_threshold: Optional[float] = None
) -> Dict:
    """
    Process a directory of images through the full pipeline (FR1.2).
    
    Runs the full pipeline on each image, saves predictions and optional overlays
    to an output directory, and tracks timing for performance monitoring.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory where results will be saved
        template_config_path: Path to template configuration file
        detection_config: Optional configuration dict for star detection
        save_overlays: If True, save visualization overlays with detected stars
        no_match_threshold: Optional threshold to declare "no match" if all scores exceed it
    
    Returns:
        Dictionary containing:
            - 'results': List of per-image results with timing
            - 'summary': Summary statistics (total time, average time, etc.)
            - 'predictions': List of predictions per image
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files (common formats)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions and f.is_file()
    ]
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {input_dir}")
    
    print(f"Found {len(image_files)} images to process")
    
    # Load templates once (optimization)
    print(f"Loading templates from {template_config_path}...")
    templates = load_templates(template_config_path)
    print(f"Loaded {len(templates)} templates")
    
    # Set default detection config if not provided
    if detection_config is None:
        detection_config = {'intensity_threshold': 0.01}
    
    # Process each image
    results = []
    total_time = 0.0
    processing_times = []
    
    for i, image_file in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_file.name}")
        
        # Load image
        image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  Warning: Could not load {image_file.name}, skipping")
            continue
        
        # Record image size for performance tracking
        image_size = image.shape[:2]
        
        # Time the full pipeline
        start_time = time.perf_counter()
        
        # Run pipeline (optimized: templates already loaded)
        try:
            # Step 1: Detect stars
            detected_centroids = detect_stars(image, config=detection_config)
            
            if len(detected_centroids) == 0:
                best_match = None
                score = float('inf')
            else:
                # Step 2: Normalize point set
                normalized_query = normalize_star_points(detected_centroids)
                
                # Step 3: Match to templates (already loaded)
                best_match, score = match_constellation_ssd(
                    normalized_query,
                    templates,
                    no_match_threshold=no_match_threshold
                )
        except Exception as e:
            print(f"  Error processing {image_file.name}: {e}")
            best_match = None
            score = float('inf')
            detected_centroids = np.array([])
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        total_time += processing_time
        processing_times.append(processing_time)
        
        # Save overlay if requested
        overlay_path = None
        if save_overlays and len(detected_centroids) > 0:
            overlay_image = visualize_detection(image, detected_centroids)
            overlay_filename = f"{image_file.stem}_overlay.png"
            overlay_path = output_path / overlay_filename
            cv2.imwrite(str(overlay_path), overlay_image)
        
        # Save prediction result
        result = {
            'image_filename': image_file.name,
            'predicted_constellation': best_match,
            'score': float(score) if score != float('inf') else None,
            'num_stars_detected': len(detected_centroids),
            'image_size': image_size,
            'processing_time_seconds': processing_time,
            'overlay_filename': overlay_path.name if overlay_path else None
        }
        results.append(result)
        
        # Print result
        print(f"  Match: {best_match}, Score: {score:.4f if score != float('inf') else 'inf'}, "
              f"Time: {processing_time:.3f}s, Stars: {len(detected_centroids)}")
    
    # Compute summary statistics
    if len(processing_times) > 0:
        avg_time = total_time / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
    else:
        avg_time = 0.0
        min_time = 0.0
        max_time = 0.0
    
    summary = {
        'total_images': len(image_files),
        'processed_images': len(results),
        'total_time_seconds': total_time,
        'average_time_seconds': avg_time,
        'min_time_seconds': min_time,
        'max_time_seconds': max_time,
        'images_per_second': len(results) / total_time if total_time > 0 else 0.0
    }
    
    # Save results to JSON
    results_file = output_path / 'batch_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'results': results
        }, f, indent=2)
    
    # Save results to CSV for easy analysis
    csv_file = output_path / 'batch_results.csv'
    with open(csv_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    return {
        'results': results,
        'summary': summary
    }


def print_performance_summary(summary: Dict):
    """
    Print a formatted performance summary.
    
    Args:
        summary: Summary dictionary from process_batch
    """
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    print(f"Total images: {summary['total_images']}")
    print(f"Processed images: {summary['processed_images']}")
    print(f"\nPerformance Metrics:")
    print(f"  Total time: {summary['total_time_seconds']:.3f} seconds")
    print(f"  Average time per image: {summary['average_time_seconds']:.3f} seconds")
    print(f"  Min time: {summary['min_time_seconds']:.3f} seconds")
    print(f"  Max time: {summary['max_time_seconds']:.3f} seconds")
    print(f"  Throughput: {summary['images_per_second']:.2f} images/second")
    
    # Check NFR1 requirement (512x512 image in under 5 seconds)
    avg_time = summary['average_time_seconds']
    nfr1_met = avg_time < 5.0
    print(f"\nNFR1 Performance Requirement:")
    print(f"  Target: < 5 seconds per 512×512 image")
    print(f"  Average: {avg_time:.3f} seconds")
    print(f"  Status: {'✓ PASS' if nfr1_met else '✗ FAIL'}")
    print("=" * 60)


def main():
    """
    Main entry point for batch processing with CLI support.
    """
    parser = argparse.ArgumentParser(
        description='Batch process images through constellation detection pipeline'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing images to process'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for results and overlays'
    )
    parser.add_argument(
        '--templates',
        type=str,
        default='templates_config.json',
        help='Path to template configuration file (default: templates_config.json)'
    )
    parser.add_argument(
        '--detection-threshold',
        type=float,
        default=0.01,
        help='Intensity threshold for star detection (default: 0.01)'
    )
    parser.add_argument(
        '--no-overlays',
        action='store_true',
        help='Skip saving overlay visualization images'
    )
    parser.add_argument(
        '--no-match-threshold',
        type=float,
        default=None,
        help='SSD threshold for declaring no match (default: None)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Constellation Detection - Batch Processing")
    print("=" * 60)
    
    # Prepare configuration
    detection_config = {
        'intensity_threshold': args.detection_threshold
    }
    
    # Run batch processing
    try:
        results = process_batch(
            input_dir=args.input,
            output_dir=args.output,
            template_config_path=args.templates,
            detection_config=detection_config,
            save_overlays=not args.no_overlays,
            no_match_threshold=args.no_match_threshold
        )
        
        # Print summary
        print_performance_summary(results['summary'])
        
        print(f"\nResults saved to:")
        print(f"  - {Path(args.output) / 'batch_results.json'}")
        print(f"  - {Path(args.output) / 'batch_results.csv'}")
        if not args.no_overlays:
            print(f"  - Overlay images in {args.output}/")
        
        return 0
        
    except Exception as e:
        print(f"\nError during batch processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

