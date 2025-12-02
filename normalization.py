"""
Normalization Module

This module handles normalization of star point sets for translation,
rotation, and scale invariance using PCA and centering.
"""

import numpy as np
from typing import List, Tuple, Optional


def center_points(points: np.ndarray) -> np.ndarray:
    """
    Center points by subtracting the mean (centroid).
    
    This removes translation differences by moving the centroid to the origin.
    
    Args:
        points: Array of shape (n_points, 2) with (x, y) coordinates
    
    Returns:
        Centered points array of shape (n_points, 2) with zero mean
    """
    if len(points) == 0:
        return points
    
    points = np.array(points)
    
    # Compute centroid (mean)
    centroid = np.mean(points, axis=0)
    
    # Subtract mean to center at origin
    centered = points - centroid
    
    return centered


def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Normalize a set of 2D points for translation, rotation, and scale.

    The normalization process:
    1. Centers the points by subtracting the centroid
    2. Scales to unit variance
    3. Uses PCA to align to a canonical orientation

    Args:
        points: Array of shape (n_points, 2) with (x, y) coordinates

    Returns:
        Normalized points array of shape (n_points, 2)
    """
    if len(points) == 0:
        return points
    
    points = np.array(points)
    
    # Step 1: Center the points (translation normalization)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Step 2: Scale normalization - compute scale based on standard deviation
    scale = np.std(centered)
    if scale < 1e-10:  # Avoid division by zero
        return centered
    
    scaled = centered / scale
    
    # Step 3: Rotation normalization using PCA
    # Compute covariance matrix
    cov_matrix = np.cov(scaled.T)
    
    # Get principal components
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending order)
    idx = eigenvals.argsort()[::-1]
    eigenvecs = eigenvecs[:, idx]
    
    # Project points onto principal components
    normalized = scaled @ eigenvecs
    
    return normalized


def normalize_star_set(centroids: List[Tuple[float, float]]) -> np.ndarray:
    """
    Normalize a list of star centroids.

    Args:
        centroids: List of (x, y) coordinates

    Returns:
        Normalized points array of shape (n_stars, 2)
    """
    if len(centroids) == 0:
        return np.array([])
    
    points = np.array(centroids)
    return normalize_points(points)

