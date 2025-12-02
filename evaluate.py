"""
Evaluation Module

This module provides evaluation metrics and utilities for assessing
constellation detection performance, including accuracy, confusion matrices, etc.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


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

