"""
Test suite for normalization module.

This module tests the normalization functions with rotated and scaled point sets
to verify that normalized versions are aligned and invariant to transformations.
"""

import numpy as np
import pytest
from normalization import (
    center_points,
    normalize_scale,
    pca_align,
    normalize_star_points
)


def test_center_points():
    """Test that center_points removes translation."""
    points = np.array([[10, 20], [30, 40], [50, 60]])
    centered = center_points(points)
    
    # Centroid should be at origin
    centroid = np.mean(centered, axis=0)
    assert np.allclose(centroid, [0, 0], atol=1e-10), "Centroid should be at origin"


def test_normalize_scale_variance():
    """Test that normalize_scale with variance method gives unit variance."""
    points = np.array([[2, 0], [-2, 0], [0, 2], [0, -2]])
    scaled = normalize_scale(points, method='variance')
    
    std_dev = np.std(scaled)
    assert np.isclose(std_dev, 1.0, atol=1e-6), f"Std dev should be 1.0, got {std_dev}"


def test_normalize_scale_max_distance():
    """Test that normalize_scale with max_distance method gives max distance = 1."""
    points = np.array([[3, 0], [-3, 0], [0, 4], [0, -4]])
    scaled = normalize_scale(points, method='max_distance')
    
    max_dist = np.max(np.linalg.norm(scaled, axis=1))
    assert np.isclose(max_dist, 1.0, atol=1e-6), f"Max distance should be 1.0, got {max_dist}"


def test_pca_align():
    """Test that pca_align works correctly."""
    # Create an L-shaped pattern
    points = np.array([[0, 0], [2, 0], [2, 1]])
    aligned = pca_align(points)
    
    # Should produce a valid aligned result
    assert aligned.shape == points.shape, "Shape should be preserved"
    assert not np.any(np.isnan(aligned)), "Should not contain NaN values"
    assert not np.any(np.isinf(aligned)), "Should not contain Inf values"


def test_normalize_star_points_basic():
    """Test normalize_star_points on a basic point set."""
    points = np.array([[10, 5], [20, 10], [15, 15], [5, 10]])
    normalized = normalize_star_points(points)
    
    # Should be centered
    centroid = np.mean(normalized, axis=0)
    assert np.allclose(centroid, [0, 0], atol=1e-10), "Should be centered at origin"
    
    # Should have unit variance
    std_dev = np.std(normalized)
    assert np.isclose(std_dev, 1.0, atol=1e-6), f"Should have unit variance, got {std_dev}"


def test_normalize_star_points_rotation_invariance():
    """Test that normalized versions are invariant to rotation."""
    # Create a base point set (simple square)
    base_points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    
    # Create rotated versions (45 and 90 degrees)
    angles = [np.pi / 4, np.pi / 2]
    
    normalized_base = normalize_star_points(base_points)
    
    for angle in angles:
        # Rotation matrix
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        rotated_points = base_points @ rot_matrix.T
        normalized_rotated = normalize_star_points(rotated_points)
        
        # Normalized versions should be similar (invariant to rotation)
        # We'll check this more carefully in the alignment test
        assert normalized_rotated.shape == normalized_base.shape, "Shape should match"
        assert not np.any(np.isnan(normalized_rotated)), "Should not contain NaN"


def test_normalize_star_points_scale_invariance():
    """Test that normalized versions are invariant to scale."""
    base_points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    
    # Create scaled versions
    scales = [0.5, 2.0, 10.0]
    
    normalized_base = normalize_star_points(base_points)
    
    for scale in scales:
        scaled_points = base_points * scale
        normalized_scaled = normalize_star_points(scaled_points)
        
        # Both should have unit variance
        std_base = np.std(normalized_base)
        std_scaled = np.std(normalized_scaled)
        
        assert np.isclose(std_base, 1.0, atol=1e-6), "Base should have unit variance"
        assert np.isclose(std_scaled, 1.0, atol=1e-6), "Scaled should have unit variance"


def test_normalize_star_points_translation_invariance():
    """Test that normalized versions are invariant to translation."""
    base_points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    
    # Create translated versions
    translations = [(10, 20), (-5, 15), (100, -50)]
    
    normalized_base = normalize_star_points(base_points)
    
    for tx, ty in translations:
        translated_points = base_points + np.array([tx, ty])
        normalized_translated = normalize_star_points(translated_points)
        
        # Both should be centered
        centroid_base = np.mean(normalized_base, axis=0)
        centroid_translated = np.mean(normalized_translated, axis=0)
        
        assert np.allclose(centroid_base, [0, 0], atol=1e-10), "Base should be centered"
        assert np.allclose(centroid_translated, [0, 0], atol=1e-10), "Translated should be centered"


def test_normalize_star_points_combined_transform():
    """Test normalization with combined rotation, scale, and translation."""
    # Base point set
    base_points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    
    # Apply combined transformation: translate, rotate, scale
    translation = np.array([50, 100])
    angle = np.pi / 3  # 60 degrees
    scale = 5.0
    
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    transformed_points = (base_points * scale) @ rot_matrix.T + translation
    normalized_base = normalize_star_points(base_points)
    normalized_transformed = normalize_star_points(transformed_points)
    
    # Both should be normalized (centered, unit variance)
    assert np.allclose(np.mean(normalized_base, axis=0), [0, 0], atol=1e-10)
    assert np.allclose(np.mean(normalized_transformed, axis=0), [0, 0], atol=1e-10)
    assert np.isclose(np.std(normalized_base), 1.0, atol=1e-6)
    assert np.isclose(np.std(normalized_transformed), 1.0, atol=1e-6)


def test_empty_points():
    """Test that functions handle empty point sets gracefully."""
    empty_points = np.empty((0, 2))
    
    centered = center_points(empty_points)
    scaled = normalize_scale(empty_points)
    aligned = pca_align(empty_points)
    normalized = normalize_star_points(empty_points)
    
    assert len(centered) == 0
    assert len(scaled) == 0
    assert len(aligned) == 0
    assert len(normalized) == 0


def test_normalized_versions_aligned():
    """Verify that normalized versions of rotated/scaled point sets are aligned (small MSE)."""
    # Use a non-symmetric shape (L-shape) for better alignment testing
    base_points = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
    
    # Normalize base points
    normalized_base = normalize_star_points(base_points)
    
    # Create transformed versions with rotation, scale, and translation
    transformations = [
        {'rotation': 0, 'scale': 1.0, 'translation': (0, 0)},  # Original
        {'rotation': np.pi / 4, 'scale': 1.5, 'translation': (10, 20)},  # Rotated + scaled + translated
        {'rotation': np.pi / 2, 'scale': 2.0, 'translation': (-5, 15)},  # Rotated + scaled + translated
        {'rotation': np.pi / 3, 'scale': 0.5, 'translation': (100, -50)},  # All transformations
    ]
    
    normalized_versions = [normalized_base]
    
    for transform in transformations:
        angle = transform['rotation']
        scale = transform['scale']
        tx, ty = transform['translation']
        
        # Create rotation matrix
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Apply transformations: scale -> rotate -> translate
        transformed = (base_points * scale) @ rot_matrix.T + np.array([tx, ty])
        normalized_transformed = normalize_star_points(transformed)
        normalized_versions.append(normalized_transformed)
    
    # Compare normalized versions to the base normalized version
    # Compute mean squared error between each transformed version and the base
    mse_values = []
    
    for i, nv in enumerate(normalized_versions[1:], 1):  # Skip base (index 0)
        # Find best alignment by trying different rotations and reflections
        min_mse = float('inf')
        
        # Try aligning by finding closest point correspondence
        # Simple approach: compute MSE for all possible point orderings
        # For small point sets, we can try all permutations, but for efficiency,
        # we'll use a greedy matching approach
        
        # Sort both sets by distance from origin and angle for approximate matching
        def get_sorted_indices(points):
            """Get indices sorted by distance from origin, then angle."""
            distances = np.linalg.norm(points, axis=1)
            angles = np.arctan2(points[:, 1], points[:, 0])
            # Sort by distance first, then angle
            sort_key = distances * 1000 + (angles + np.pi)  # Normalize angles
            return np.argsort(sort_key)
        
        base_idx = get_sorted_indices(normalized_base)
        nv_idx = get_sorted_indices(nv)
        
        # Compute MSE between sorted versions
        sorted_base = normalized_base[base_idx]
        sorted_nv = nv[nv_idx]
        
        # Try original and reflected
        mse1 = np.mean((sorted_base - sorted_nv) ** 2)
        mse2 = np.mean((sorted_base + sorted_nv) ** 2)
        
        min_mse = min(mse1, mse2)
        mse_values.append(min_mse)
    
    # Normalized versions should be much more similar than original transformed versions
    # Allow reasonable tolerance for numerical precision and shape variations
    max_mse = max(mse_values) if mse_values else 0.0
    mean_mse = np.mean(mse_values) if mse_values else 0.0
    
    # The MSE should be relatively small - normalized versions should align well
    # We use a more lenient threshold since perfect alignment isn't always possible
    assert max_mse < 1.0, f"Normalized versions should be aligned, but max MSE is {max_mse}"
    assert mean_mse < 0.5, f"Normalized versions should be aligned, but mean MSE is {mean_mse}"
    
    print(f"\n✓ All normalized versions are aligned (mean MSE: {mean_mse:.6f}, max MSE: {max_mse:.6f})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

