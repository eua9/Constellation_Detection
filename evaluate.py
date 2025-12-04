"""
Evaluation Module

This module provides evaluation metrics and utilities for assessing
constellation detection performance, including accuracy, confusion matrices, etc.
"""

import numpy as np
import json
import cv2
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import csv

from star_detection import detect_stars
from normalization import normalize_star_points
from matching import match_constellation_ssd, match_constellation_hausdorff, match_constellation
from templates import load_templates


def compute_accuracy(
    predictions: List[str],
    ground_truth: List[str]
) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: List of predicted constellation names
        ground_truth: List of ground truth constellation names

    Returns:
        Accuracy as a float between 0 and 1
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
    return correct / len(predictions)


def compute_confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    class_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute confusion matrix for constellation classification.

    Args:
        predictions: List of predicted constellation names
        ground_truth: List of ground truth constellation names
        class_names: Optional list of all class names (if None, inferred from data)

    Returns:
        Tuple of (confusion_matrix, class_names)
        - confusion_matrix: 2D numpy array where [i, j] is count of class i predicted as class j
        - class_names: List of class names in order
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    # Get unique class names
    if class_names is None:
        all_classes = set(predictions) | set(ground_truth)
        class_names = sorted(list(all_classes))
    
    # Create mapping from class name to index
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    n_classes = len(class_names)
    
    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix
    for pred, gt in zip(predictions, ground_truth):
        if pred in class_to_idx and gt in class_to_idx:
            pred_idx = class_to_idx[pred]
            gt_idx = class_to_idx[gt]
            cm[gt_idx, pred_idx] += 1
    
    return cm, class_names


def print_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix"
):
    """
    Print a formatted confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        title: Title for the confusion matrix
    """
    print(f"\n{title}")
    print("=" * (len(title) + 2))
    
    # Header row
    print(f"{'':>15}", end="")
    for name in class_names:
        print(f"{name[:10]:>12}", end="")
    print()
    
    # Data rows
    for i, name in enumerate(class_names):
        print(f"{name[:14]:>15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>12}", end="")
        print()
    
    print()


def evaluate_dataset(
    dataset_dir: str,
    templates: Dict[str, np.ndarray],
    config: Optional[Dict] = None
) -> Dict:
    """
    Evaluate the full pipeline on a labeled synthetic dataset.
    
    Iterates over all images and labels, runs the full pipeline
    (detection → normalization → SSD matching), and records
    predicted vs. true constellation labels.
    
    Args:
        dataset_dir: Directory containing synthetic dataset images and metadata
        templates: Dictionary mapping constellation names to normalized template point arrays
        config: Optional configuration dictionary with:
            - 'detection_config': Configuration for star detection
            - 'no_match_threshold': Optional threshold for declaring no match
    
    Returns:
        Dictionary containing:
            - 'predictions': List of predicted constellation names
            - 'ground_truth': List of true constellation names
            - 'scores': List of SSD scores
            - 'accuracy': Overall accuracy
            - 'confusion_matrix': Confusion matrix array
            - 'class_names': List of class names
            - 'results': List of (predicted, ground_truth, score) tuples
    """
    if config is None:
        config = {}
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    # Find all image files
    image_files = sorted(list(dataset_path.glob('constellation_*.png')))
    
    if len(image_files) == 0:
        raise ValueError(f"No constellation images found in {dataset_dir}")
    
    print(f"Found {len(image_files)} images in dataset")
    
    # Extract detection config
    detection_config = config.get('detection_config', {'intensity_threshold': 0.01})
    no_match_threshold = config.get('no_match_threshold', None)
    matching_method = config.get('matching_method', 'ssd')
    
    predictions = []
    ground_truth = []
    scores = []
    results = []
    
    # Process each image
    for i, image_file in enumerate(image_files):
        # Load corresponding metadata file
        metadata_file = dataset_path / f"{image_file.stem}_metadata.json"
        
        if not metadata_file.exists():
            print(f"Warning: No metadata found for {image_file.name}, skipping")
            continue
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        true_label = metadata.get('constellation_name')
        if true_label is None:
            print(f"Warning: No constellation_name in metadata for {image_file.name}, skipping")
            continue
        
        # Load image
        image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not load image {image_file.name}, skipping")
            continue
        
        # Run full pipeline: detection → normalization → matching
        # Step 1: Detect stars
        detected_centroids = detect_stars(image, config=detection_config)
        
        if len(detected_centroids) == 0:
            print(f"Warning: No stars detected in {image_file.name}, skipping")
            predictions.append(None)
            ground_truth.append(true_label)
            scores.append(float('inf'))
            results.append((None, true_label, float('inf')))
            continue
        
        # Step 2: Normalize detected stars
        normalized_query = normalize_star_points(detected_centroids)
        
        # Step 3: Match to templates using selected method
        if matching_method == 'ssd':
            predicted_label, score = match_constellation_ssd(
                normalized_query,
                templates,
                no_match_threshold=no_match_threshold
            )
        elif matching_method == 'hausdorff':
            predicted_label, score = match_constellation_hausdorff(
                normalized_query,
                templates,
                no_match_threshold=no_match_threshold
            )
        else:
            # Use generic match_constellation function
            predicted_label, score = match_constellation(
                normalized_query,
                templates,
                method=matching_method
            )
        
        # Record results
        predictions.append(predicted_label)
        ground_truth.append(true_label)
        scores.append(score)
        results.append((predicted_label, true_label, score))
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images...")
    
    # Filter out None predictions for accuracy calculation
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    if len(valid_indices) == 0:
        print("Warning: No valid predictions found")
        return {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'scores': scores,
            'accuracy': 0.0,
            'confusion_matrix': None,
            'class_names': [],
            'results': results
        }
    
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_ground_truth = [ground_truth[i] for i in valid_indices]
    
    # Compute metrics
    accuracy = compute_accuracy(valid_predictions, valid_ground_truth)
    cm, class_names = compute_confusion_matrix(valid_predictions, valid_ground_truth)
    
    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'scores': scores,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'class_names': class_names,
        'results': results
    }


def compare_methods(
    dataset_dir: str,
    templates: Dict[str, np.ndarray],
    config: Dict,
    output_path: Optional[str] = None
) -> Dict:
    """
    Compare SSD and Hausdorff matching methods on the same dataset.
    
    Runs evaluation with both methods and compares accuracy and confusion matrices.
    
    Args:
        dataset_dir: Directory containing synthetic dataset images and metadata
        templates: Dictionary mapping constellation names to normalized template point arrays
        config: Configuration dictionary (detection_config, no_match_threshold)
        output_path: Optional path to save comparison results
    
    Returns:
        Dictionary containing comparison results for both methods
    """
    print("\n" + "=" * 60)
    print("Method Comparison: SSD vs. Hausdorff")
    print("=" * 60)
    
    # Evaluate with SSD
    print("\n[1/2] Evaluating with SSD method...")
    config_ssd = config.copy()
    config_ssd['matching_method'] = 'ssd'
    metrics_ssd = evaluate_dataset(dataset_dir, templates, config_ssd)
    
    # Evaluate with Hausdorff
    print("\n[2/2] Evaluating with Hausdorff method...")
    config_hausdorff = config.copy()
    config_hausdorff['matching_method'] = 'hausdorff'
    metrics_hausdorff = evaluate_dataset(dataset_dir, templates, config_hausdorff)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'SSD':<15} {'Hausdorff':<15} {'Difference':<15}")
    print("-" * 70)
    
    acc_ssd = metrics_ssd['accuracy']
    acc_hausdorff = metrics_hausdorff['accuracy']
    acc_diff = acc_hausdorff - acc_ssd
    
    print(f"{'Accuracy':<25} {acc_ssd:<15.4f} {acc_hausdorff:<15.4f} {acc_diff:+.4f}")
    
    num_valid_ssd = sum(1 for p in metrics_ssd['predictions'] if p is not None)
    num_valid_hausdorff = sum(1 for p in metrics_hausdorff['predictions'] if p is not None)
    
    print(f"{'Valid Predictions':<25} {num_valid_ssd:<15} {num_valid_hausdorff:<15} {num_valid_hausdorff - num_valid_ssd:+d}")
    
    num_correct_ssd = sum(1 for p, gt in zip(metrics_ssd['predictions'], metrics_ssd['ground_truth']) 
                         if p is not None and p == gt)
    num_correct_hausdorff = sum(1 for p, gt in zip(metrics_hausdorff['predictions'], metrics_hausdorff['ground_truth']) 
                                if p is not None and p == gt)
    
    print(f"{'Correct Predictions':<25} {num_correct_ssd:<15} {num_correct_hausdorff:<15} {num_correct_hausdorff - num_correct_ssd:+d}")
    
    # Print confusion matrices
    print("\n" + "=" * 60)
    print("SSD Method - Confusion Matrix")
    print("=" * 60)
    if metrics_ssd['confusion_matrix'] is not None:
        print_confusion_matrix(metrics_ssd['confusion_matrix'], metrics_ssd['class_names'], "SSD Method")
    
    print("\n" + "=" * 60)
    print("Hausdorff Method - Confusion Matrix")
    print("=" * 60)
    if metrics_hausdorff['confusion_matrix'] is not None:
        print_confusion_matrix(metrics_hausdorff['confusion_matrix'], metrics_hausdorff['class_names'], "Hausdorff Method")
    
    # Save comparison results
    comparison_results = {
        'ssd': {
            'accuracy': acc_ssd,
            'num_valid': num_valid_ssd,
            'num_correct': num_correct_ssd,
            'confusion_matrix': metrics_ssd['confusion_matrix'].tolist() if metrics_ssd['confusion_matrix'] is not None else None,
            'class_names': metrics_ssd['class_names']
        },
        'hausdorff': {
            'accuracy': acc_hausdorff,
            'num_valid': num_valid_hausdorff,
            'num_correct': num_correct_hausdorff,
            'confusion_matrix': metrics_hausdorff['confusion_matrix'].tolist() if metrics_hausdorff['confusion_matrix'] is not None else None,
            'class_names': metrics_hausdorff['class_names']
        },
        'comparison': {
            'accuracy_difference': acc_diff,
            'best_method': 'hausdorff' if acc_hausdorff > acc_ssd else 'ssd'
        }
    }
    
    if output_path:
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\nComparison results saved to {output_file}")
    
    return comparison_results


def save_metrics(metrics: Dict, output_path: str, format: str = 'json'):
    """
    Save evaluation metrics to a file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_path: Path where metrics should be saved
        format: Output format - 'json' or 'csv'
    """
    output_file = Path(output_path)
    
    if format == 'json':
        # Prepare JSON-serializable data
        json_data = {
            'accuracy': metrics['accuracy'],
            'num_images': len(metrics['ground_truth']),
            'num_valid_predictions': sum(1 for p in metrics['predictions'] if p is not None),
            'class_names': metrics['class_names'],
            'confusion_matrix': metrics['confusion_matrix'].tolist() if metrics['confusion_matrix'] is not None else None,
            'per_image_results': [
                {
                    'predicted': pred,
                    'ground_truth': gt,
                    'score': score
                }
                for pred, gt, score in metrics['results']
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Metrics saved to {output_file} (JSON format)")
    
    elif format == 'csv':
        # Save confusion matrix as CSV
        if metrics['confusion_matrix'] is not None:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header row
                writer.writerow([''] + metrics['class_names'])
                # Data rows
                for i, class_name in enumerate(metrics['class_names']):
                    row = [class_name] + metrics['confusion_matrix'][i].tolist()
                    writer.writerow(row)
            
            print(f"Confusion matrix saved to {output_file} (CSV format)")
        else:
            print("Warning: No confusion matrix to save")
    
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'")


def evaluate_on_synthetic_data(
    results: List[Tuple[str, str, float]],
    print_results: bool = True
) -> Dict[str, float]:
    """
    Evaluate performance on synthetic dataset.

    Args:
        results: List of (predicted_name, ground_truth_name, score) tuples
        print_results: Whether to print evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    predictions = [r[0] for r in results]
    ground_truth = [r[1] for r in results]
    
    accuracy = compute_accuracy(predictions, ground_truth)
    
    cm, class_names = compute_confusion_matrix(predictions, ground_truth)
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    
    if print_results:
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print_confusion_matrix(cm, class_names)
    
    return metrics


def main():
    """
    Main entry point for evaluation script with CLI support.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate constellation detection pipeline on synthetic dataset'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to synthetic dataset directory containing images and metadata'
    )
    parser.add_argument(
        '--templates',
        type=str,
        default='templates_config.json',
        help='Path to template configuration file (default: templates_config.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation metrics (JSON format). If not specified, only prints results.'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['json', 'csv'],
        default='json',
        help='Output format for metrics file (default: json)'
    )
    parser.add_argument(
        '--detection-threshold',
        type=float,
        default=0.01,
        help='Intensity threshold for star detection (default: 0.01)'
    )
    parser.add_argument(
        '--no-match-threshold',
        type=float,
        default=None,
        help='Threshold for declaring no match (default: None)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['ssd', 'hausdorff', 'compare'],
        default='ssd',
        help='Matching method: ssd (baseline), hausdorff (extended), or compare (both) (default: ssd)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Constellation Detection Evaluation")
    print("=" * 60)
    
    # Load templates
    print(f"\nLoading templates from {args.templates}...")
    try:
        templates = load_templates(args.templates)
        print(f"Loaded {len(templates)} templates: {list(templates.keys())}")
    except Exception as e:
        print(f"Error loading templates: {e}")
        return 1
    
    # Prepare configuration
    config = {
        'detection_config': {
            'intensity_threshold': args.detection_threshold
        },
        'no_match_threshold': args.no_match_threshold,
        'matching_method': args.method
    }
    
    # Run evaluation
    if args.method == 'compare':
        # Compare both methods
        print(f"\nComparing SSD and Hausdorff methods on dataset: {args.dataset}")
        try:
            compare_methods(args.dataset, templates, config, args.output)
        except Exception as e:
            print(f"Error during comparison: {e}")
            import traceback
            traceback.print_exc()
            return 1
        return 0
    
    print(f"\nEvaluating dataset: {args.dataset} (method: {args.method})")
    try:
        metrics = evaluate_dataset(args.dataset, templates, config)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    num_total = len(metrics['ground_truth'])
    num_valid = sum(1 for p in metrics['predictions'] if p is not None)
    num_correct = sum(1 for p, gt in zip(metrics['predictions'], metrics['ground_truth']) 
                     if p is not None and p == gt)
    
    print(f"\nTotal images: {num_total}")
    print(f"Valid predictions: {num_valid}")
    print(f"Correct predictions: {num_correct}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    if metrics['confusion_matrix'] is not None:
        print_confusion_matrix(metrics['confusion_matrix'], metrics['class_names'])
    
    # Save metrics if output path specified
    if args.output:
        save_metrics(metrics, args.output, format=args.output_format)
    
    print("=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
