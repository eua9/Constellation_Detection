"""
Matching Module

This module handles matching detected star patterns to known constellation templates.
Implements baseline SSD (Sum of Squared Differences) matching and provides
structure for extended matching methods.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def compute_ssd(query_points: np.ndarray, template_points: np.ndarray) -> float:
    """
    Compute Sum of Squared Differences between two point sets.

    This is a baseline matching algorithm. Both point sets should be normalized
    before calling this function.

    Args:
        query_points: Normalized query points of shape (n_query, 2)
        template_points: Normalized template points of shape (n_template, 2)

    Returns:
        SSD score (lower is better match)
    """
    if len(query_points) == 0 or len(template_points) == 0:
        return float('inf')
    
    # Handle different sizes by using the minimum size
    min_size = min(len(query_points), len(template_points))
    
    # Sort points for consistent comparison (by distance from origin)
    query_sorted = sorted(query_points, key=lambda p: p[0]**2 + p[1]**2)
    template_sorted = sorted(template_points, key=lambda p: p[0]**2 + p[1]**2)
    
    # Compute SSD for the first min_size points
    query_subset = np.array(query_sorted[:min_size])
    template_subset = np.array(template_sorted[:min_size])
    
    # Compute squared differences
    diff = query_subset - template_subset
    ssd = np.sum(diff ** 2)
    
    return ssd


def match_constellation(
    query_points: np.ndarray,
    templates: Dict[str, np.ndarray],
    method: str = 'ssd'
) -> Tuple[str, float]:
    """
    Match a normalized query point set to the best-matching template.

    Args:
        query_points: Normalized query points of shape (n_query, 2)
        templates: Dictionary mapping constellation names to normalized template points
        method: Matching method ('ssd' for baseline, or future methods)

    Returns:
        Tuple of (best_matching_constellation_name, score)
    """
    if len(templates) == 0:
        raise ValueError("No templates provided for matching")
    
    best_match = None
    best_score = float('inf')
    
    for constellation_name, template_points in templates.items():
        if method == 'ssd':
            score = compute_ssd(query_points, template_points)
        else:
            raise ValueError(f"Unknown matching method: {method}")
        
        if score < best_score:
            best_score = score
            best_match = constellation_name
    
    return best_match, best_score


def compute_hausdorff_distance(
    query_points: np.ndarray,
    template_points: np.ndarray
) -> float:
    """
    Compute Hausdorff distance between two point sets.
    
    This is an extended matching method (stretch goal).

    Args:
        query_points: Normalized query points of shape (n_query, 2)
        template_points: Normalized template points of shape (n_template, 2)

    Returns:
        Hausdorff distance (lower is better match)
    """
    if len(query_points) == 0 or len(template_points) == 0:
        return float('inf')
    
    # Compute pairwise distances
    distances = np.sqrt(((query_points[:, np.newaxis, :] - 
                         template_points[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Hausdorff distance is the maximum of:
    # - max over query points of min distance to template
    # - max over template points of min distance to query
    h1 = np.max(np.min(distances, axis=1))
    h2 = np.max(np.min(distances, axis=0))
    
    return max(h1, h2)

