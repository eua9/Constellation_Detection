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

    Handles differing cardinality using a nearest neighbor matching strategy:
    - Finds the closest template point for each query point
    - Computes SSD using the matched pairs
    - This allows matching even when point sets have different sizes

    Args:
        query_points: Normalized query points of shape (n_query, 2)
        template_points: Normalized template points of shape (n_template, 2)

    Returns:
        SSD score (lower is better match)
    """
    if len(query_points) == 0 or len(template_points) == 0:
        return float('inf')
    
    # Strategy: For each query point, find nearest template point
    # Compute squared distances between all pairs
    # Shape: (n_query, n_template)
    distances_squared = np.sum(
        (query_points[:, np.newaxis, :] - template_points[np.newaxis, :, :]) ** 2,
        axis=2
    )
    
    # For each query point, find the nearest template point
    min_distances_squared = np.min(distances_squared, axis=1)
    
    # SSD is the sum of squared distances from query points to their nearest template points
    ssd = np.sum(min_distances_squared)
    
    # Normalize by number of query points to make score comparable across different sizes
    ssd = ssd / len(query_points)
    
    return ssd


def match_constellation_ssd(
    query_points: np.ndarray,
    templates: Dict[str, np.ndarray],
    no_match_threshold: Optional[float] = None
) -> Tuple[Optional[str], float]:
    """
    Match a normalized query point set to the best-matching template using SSD.

    Args:
        query_points: Normalized query points of shape (n_query, 2)
        templates: Dictionary mapping constellation names to normalized template points
        no_match_threshold: Optional threshold to declare "no match". If all SSD scores
                          exceed this threshold, returns (None, best_score).

    Returns:
        Tuple of (best_matching_constellation_name, score).
        If no_match_threshold is set and all scores exceed it, returns (None, best_score).
    """
    if len(templates) == 0:
        raise ValueError("No templates provided for matching")
    
    if len(query_points) == 0:
        return None, float('inf')
    
    best_match = None
    best_score = float('inf')
    
    # Compute SSD for each template
    for constellation_name, template_points in templates.items():
        score = compute_ssd(query_points, template_points)
        
        if score < best_score:
            best_score = score
            best_match = constellation_name
    
    # Check if best score exceeds threshold
    if no_match_threshold is not None and best_score > no_match_threshold:
        return None, best_score
    
    return best_match, best_score


def match_constellation(
    query_points: np.ndarray,
    templates: Dict[str, np.ndarray],
    method: str = 'ssd'
) -> Tuple[Optional[str], float]:
    """
    Match a normalized query point set to the best-matching template.

    Args:
        query_points: Normalized query points of shape (n_query, 2)
        templates: Dictionary mapping constellation names to normalized template points
        method: Matching method - 'ssd' (baseline) or 'hausdorff' (extended method)

    Returns:
        Tuple of (best_matching_constellation_name, score)
    """
    if len(templates) == 0:
        raise ValueError("No templates provided for matching")
    
    if method == 'ssd':
        return match_constellation_ssd(query_points, templates)
    elif method == 'hausdorff':
        return match_constellation_hausdorff(query_points, templates)
    else:
        raise ValueError(f"Unknown matching method: {method}. Use 'ssd' or 'hausdorff'")


def compute_hausdorff_distance(
    query_points: np.ndarray,
    template_points: np.ndarray
) -> float:
    """
    Compute Hausdorff distance between two point sets.
    
    The Hausdorff distance is the maximum of:
    - Directed Hausdorff from query to template: max over query points of min distance to template
    - Directed Hausdorff from template to query: max over template points of min distance to query
    
    This is a symmetric distance metric that measures how far two point sets are from each other.
    Lower values indicate better matches.

    Args:
        query_points: Normalized query points of shape (n_query, 2)
        template_points: Normalized template points of shape (n_template, 2)

    Returns:
        Hausdorff distance (lower is better match). Returns inf if either set is empty.
    """
    if len(query_points) == 0 or len(template_points) == 0:
        return float('inf')
    
    # Compute pairwise distances
    # Shape: (n_query, n_template)
    distances = np.sqrt(((query_points[:, np.newaxis, :] - 
                         template_points[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Directed Hausdorff from query to template:
    # For each query point, find minimum distance to any template point, then take max
    h_query_to_template = np.max(np.min(distances, axis=1))
    
    # Directed Hausdorff from template to query:
    # For each template point, find minimum distance to any query point, then take max
    h_template_to_query = np.max(np.min(distances, axis=0))
    
    # Symmetric Hausdorff distance is the maximum of the two directed distances
    return max(h_query_to_template, h_template_to_query)


def match_constellation_hausdorff(
    query_points: np.ndarray,
    templates: Dict[str, np.ndarray],
    no_match_threshold: Optional[float] = None
) -> Tuple[Optional[str], float]:
    """
    Match a normalized query point set to the best-matching template using Hausdorff distance.
    
    This is an extended matching method (FR6.1) that uses Hausdorff distance instead of SSD.
    Hausdorff distance is more robust to outliers and can handle point sets of different sizes.

    Args:
        query_points: Normalized query points of shape (n_query, 2)
        templates: Dictionary mapping constellation names to normalized template points
        no_match_threshold: Optional threshold to declare "no match". If all Hausdorff distances
                          exceed this threshold, returns (None, best_score).

    Returns:
        Tuple of (best_matching_constellation_name, score).
        If no_match_threshold is set and all scores exceed it, returns (None, best_score).
    """
    if len(templates) == 0:
        raise ValueError("No templates provided for matching")
    
    if len(query_points) == 0:
        return None, float('inf')
    
    best_match = None
    best_score = float('inf')
    
    # Compute Hausdorff distance for each template
    for constellation_name, template_points in templates.items():
        score = compute_hausdorff_distance(query_points, template_points)
        
        if score < best_score:
            best_score = score
            best_match = constellation_name
    
    # Check if best score exceeds threshold
    if no_match_threshold is not None and best_score > no_match_threshold:
        return None, best_score
    
    return best_match, best_score

