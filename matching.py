"""
Matching Module

This module handles matching detected star patterns to known constellation templates.
Implements baseline SSD (Sum of Squared Differences) matching and provides
structure for extended matching methods.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import random


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
        method: Matching method - 'ssd' (baseline), 'hausdorff' (extended), or 'ransac' (extended)

    Returns:
        Tuple of (best_matching_constellation_name, score)
    """
    if len(templates) == 0:
        raise ValueError("No templates provided for matching")
    
    if method == 'ssd':
        return match_constellation_ssd(query_points, templates)
    elif method == 'hausdorff':
        return match_constellation_hausdorff(query_points, templates)
    elif method == 'ransac':
        return match_constellation_ransac(query_points, templates)
    else:
        raise ValueError(f"Unknown matching method: {method}. Use 'ssd', 'hausdorff', or 'ransac'")


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


def estimate_similarity_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Estimate similarity transformation (rotation, scale, translation) from point correspondences.
    
    A similarity transformation preserves angles and ratios of distances.
    It can be represented as: x' = s * R * x + t
    where s is scale, R is rotation matrix, and t is translation.
    
    For 2 points, we can solve for the 4 parameters (rotation angle, scale, tx, ty).
    
    Args:
        src_points: Source points of shape (2, 2) - two (x, y) points
        dst_points: Destination points of shape (2, 2) - two (x, y) points
    
    Returns:
        Tuple of (rotation_matrix, scale, translation_vector)
        - rotation_matrix: 2x2 rotation matrix
        - scale: Scale factor
        - translation_vector: Translation vector [tx, ty]
    """
    if len(src_points) != 2 or len(dst_points) != 2:
        raise ValueError("Need exactly 2 point correspondences for similarity transform")
    
    # Compute vectors from first point to second point
    src_vec = src_points[1] - src_points[0]
    dst_vec = dst_points[1] - dst_points[0]
    
    # Compute scale as ratio of vector lengths
    src_len = np.linalg.norm(src_vec)
    dst_len = np.linalg.norm(dst_vec)
    
    if src_len < 1e-10:  # Degenerate case: points are too close
        # Fall back to translation only
        scale = 1.0
        rotation_matrix = np.eye(2)
        translation = dst_points[0] - src_points[0]
        return rotation_matrix, scale, translation
    
    scale = dst_len / src_len
    
    # Compute rotation angle
    # Angle between vectors
    cos_angle = np.dot(src_vec, dst_vec) / (src_len * dst_len)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    
    # Cross product to determine sign of angle
    cross = src_vec[0] * dst_vec[1] - src_vec[1] * dst_vec[0]
    sin_angle = cross / (src_len * dst_len)
    
    # Build rotation matrix
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])
    
    # Compute translation: t = dst[0] - s * R * src[0]
    translation = dst_points[0] - scale * (rotation_matrix @ src_points[0])
    
    return rotation_matrix, scale, translation


def apply_similarity_transform(
    points: np.ndarray,
    rotation_matrix: np.ndarray,
    scale: float,
    translation: np.ndarray
) -> np.ndarray:
    """
    Apply similarity transformation to points.
    
    Args:
        points: Points of shape (n, 2)
        rotation_matrix: 2x2 rotation matrix
        scale: Scale factor
        translation: Translation vector [tx, ty]
    
    Returns:
        Transformed points of shape (n, 2)
    """
    return (scale * (points @ rotation_matrix.T)) + translation


def compute_ransac_score(
    query_points: np.ndarray,
    template_points: np.ndarray,
    inlier_threshold: float = 0.1,
    max_iterations: int = 1000,
    min_inliers: int = 3,
    random_seed: Optional[int] = None
) -> Tuple[float, int, Tuple]:
    """
    Compute RANSAC-based score between two point sets.
    
    Uses RANSAC to find the best similarity transformation and counts inliers.
    The score is based on average inlier error (lower is better, for consistency
    with other matching methods).
    
    Args:
        query_points: Normalized query points of shape (n_query, 2)
        template_points: Normalized template points of shape (n_template, 2)
        inlier_threshold: Distance threshold for considering a point an inlier
        max_iterations: Maximum number of RANSAC iterations
        min_inliers: Minimum number of inliers required for a valid match
        random_seed: Optional random seed for reproducibility
    
    Returns:
        Tuple of (score, num_inliers, best_transform_params)
        - score: Average inlier error (lower is better)
        - num_inliers: Number of inliers found
        - best_transform_params: (rotation_matrix, scale, translation) of best transform
    """
    if len(query_points) == 0 or len(template_points) == 0:
        return float('inf'), 0, (np.eye(2), 1.0, np.array([0.0, 0.0]))
    
    if len(query_points) < 2 or len(template_points) < 2:
        # Not enough points for RANSAC
        return float('inf'), 0, (np.eye(2), 1.0, np.array([0.0, 0.0]))
    
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    best_num_inliers = 0
    best_transform = (np.eye(2), 1.0, np.array([0.0, 0.0]))
    
    n_query = len(query_points)
    n_template = len(template_points)
    
    # RANSAC iterations
    for iteration in range(max_iterations):
        # Randomly sample 2 correspondences
        # Sample 2 query points and 2 template points
        query_indices = np.random.choice(n_query, size=min(2, n_query), replace=False)
        template_indices = np.random.choice(n_template, size=min(2, n_template), replace=False)
        
        src_sample = query_points[query_indices]
        dst_sample = template_points[template_indices]
        
        try:
            # Estimate transformation from sample
            rotation_matrix, scale, translation = estimate_similarity_transform(
                src_sample, dst_sample
            )
            
            # Apply transformation to all query points
            transformed_query = apply_similarity_transform(
                query_points, rotation_matrix, scale, translation
            )
            
            # Count inliers: query points that are close to some template point
            # For each transformed query point, find distance to nearest template point
            distances = np.sqrt(
                ((transformed_query[:, np.newaxis, :] - 
                  template_points[np.newaxis, :, :]) ** 2).sum(axis=2)
            )
            min_distances = np.min(distances, axis=1)
            
            # Count inliers (points within threshold)
            num_inliers = np.sum(min_distances <= inlier_threshold)
            
            # Update best if this is better
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_transform = (rotation_matrix, scale, translation)
                
                # Early termination if we have enough inliers
                if best_num_inliers >= min(n_query, n_template) * 0.8:
                    break
        
        except Exception:
            # Skip this iteration if transformation estimation fails
            continue
    
    # Score is average inlier error (lower is better)
    if best_num_inliers >= min_inliers:
        # Apply best transform and compute average inlier error
        transformed_query = apply_similarity_transform(
            query_points, best_transform[0], best_transform[1], best_transform[2]
        )
        distances = np.sqrt(
            ((transformed_query[:, np.newaxis, :] - 
              template_points[np.newaxis, :, :]) ** 2).sum(axis=2)
        )
        min_distances = np.min(distances, axis=1)
        inlier_distances = min_distances[min_distances <= inlier_threshold]
        
        if len(inlier_distances) > 0:
            avg_inlier_error = np.mean(inlier_distances)
            score = avg_inlier_error
        else:
            score = float('inf')
    else:
        score = float('inf')
    
    return score, best_num_inliers, best_transform


def match_constellation_ransac(
    query_points: np.ndarray,
    templates: Dict[str, np.ndarray],
    no_match_threshold: Optional[float] = None,
    inlier_threshold: float = 0.1,
    max_iterations: int = 1000,
    min_inliers: int = 3,
    random_seed: Optional[int] = None
) -> Tuple[Optional[str], float]:
    """
    Match a normalized query point set to the best-matching template using RANSAC.
    
    This is an extended matching method (FR6.2) that uses RANSAC to robustly handle
    outliers (extra stars, missed detections). RANSAC estimates a similarity transformation
    and counts inliers, making it more robust than SSD or Hausdorff distance.
    
    Args:
        query_points: Normalized query points of shape (n_query, 2)
        templates: Dictionary mapping constellation names to normalized template points
        no_match_threshold: Optional threshold to declare "no match". If all RANSAC scores
                          exceed this threshold, returns (None, best_score).
        inlier_threshold: Distance threshold for considering a point an inlier
        max_iterations: Maximum number of RANSAC iterations per template
        min_inliers: Minimum number of inliers required for a valid match
        random_seed: Optional random seed for reproducibility
    
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
    best_num_inliers = 0
    
    # Compute RANSAC score for each template
    for constellation_name, template_points in templates.items():
        score, num_inliers, _ = compute_ransac_score(
            query_points,
            template_points,
            inlier_threshold=inlier_threshold,
            max_iterations=max_iterations,
            min_inliers=min_inliers,
            random_seed=random_seed
        )
        
        # Prefer matches with more inliers, then lower error
        # Score is negative num_inliers (so more inliers = lower score = better)
        # Then add normalized error as tiebreaker
        if num_inliers >= min_inliers:
            # Combined score: prioritize inlier count, then error
            # Lower score is better
            combined_score = -num_inliers * 1000 + score
            
            if combined_score < best_score or (num_inliers > best_num_inliers and score < float('inf')):
                best_score = combined_score
                best_match = constellation_name
                best_num_inliers = num_inliers
    
    # Check if best score exceeds threshold
    if no_match_threshold is not None and best_score > no_match_threshold:
        return None, best_score
    
    return best_match, best_score

