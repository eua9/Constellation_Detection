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


def normalize_scale(points: np.ndarray, method: str = 'variance') -> np.ndarray:
    """
    Normalize scale of points to unit variance or unit max distance.
    
    This removes scale differences by scaling points so they have:
    - Unit variance (standard deviation = 1), or
    - Unit max distance (max distance from origin = 1)
    
    Args:
        points: Array of shape (n_points, 2) with (x, y) coordinates
        method: Scaling method - 'variance' (default) or 'max_distance'
    
    Returns:
        Scaled points array of shape (n_points, 2)
    """
    if len(points) == 0:
        return points
    
    points = np.array(points)
    
    if method == 'variance':
        # Scale to unit variance (standard deviation = 1)
        scale = np.std(points)
        if scale < 1e-10:  # Avoid division by zero
            return points
        scaled = points / scale
    
    elif method == 'max_distance':
        # Scale so that maximum distance from origin is 1
        distances = np.linalg.norm(points, axis=1)
        max_dist = np.max(distances)
        if max_dist < 1e-10:  # Avoid division by zero
            return points
        scaled = points / max_dist
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'variance' or 'max_distance'")
    
    return scaled


def pca_align(points: np.ndarray) -> np.ndarray:
    """
    Align points using PCA to ensure consistent orientation.
    
    This removes rotation differences by:
    1. Computing principal components via PCA
    2. Aligning the first principal component to the positive x-axis
    3. Ensuring consistent orientation (positive determinant)
    
    Args:
        points: Array of shape (n_points, 2) with (x, y) coordinates
    
    Returns:
        Aligned points array of shape (n_points, 2) with consistent orientation
    """
    if len(points) == 0:
        return points
    
    points = np.array(points)
    
    # Handle degenerate case (all points at origin or collinear)
    if np.allclose(points, 0) or len(points) < 2:
        return points
    
    # Compute covariance matrix
    cov_matrix = np.cov(points.T)
    
    # Handle case where variance is zero (all points are the same)
    if np.allclose(cov_matrix, 0):
        return points
    
    # Get principal components via eigenvalue decomposition
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending order)
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Ensure consistent orientation:
    # Make sure the first principal component points in the positive x direction
    # (or positive y if x-component is very small)
    if eigenvecs[0, 0] < 0:
        eigenvecs[:, 0] = -eigenvecs[:, 0]
    
    # Ensure right-handed coordinate system (determinant = 1)
    if np.linalg.det(eigenvecs) < 0:
        eigenvecs[:, 1] = -eigenvecs[:, 1]
    
    # Project points onto principal components
    aligned = points @ eigenvecs
    
    return aligned


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


def normalize_star_points(points: np.ndarray, scale_method: str = 'variance') -> np.ndarray:
    """
    Normalize star point coordinates by chaining centering, PCA rotation, and scale normalization.
    
    This function removes translation, rotation, and scale differences from a point set
    by applying the normalization steps in sequence:
    1. Center points (remove translation)
    2. PCA align (remove rotation, ensure consistent orientation)
    3. Normalize scale (remove scale differences)
    
    Args:
        points: Array of shape (n_points, 2) with (x, y) coordinates
        scale_method: Scaling method - 'variance' (default) or 'max_distance'
    
    Returns:
        Normalized points array of shape (n_points, 2) that is translation,
        rotation, and scale invariant
    """
    if len(points) == 0:
        return points
    
    points = np.array(points)
    
    # Step 1: Center points (remove translation)
    centered = center_points(points)
    
    # Step 2: PCA align (remove rotation, ensure consistent orientation)
    aligned = pca_align(centered)
    
    # Step 3: Normalize scale (remove scale differences)
    normalized = normalize_scale(aligned, method=scale_method)
    
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

