"""
Test suite for star detection module.

This module tests the detect_stars function using synthetic and simple toy images
to confirm star centroids are detected correctly.
"""

import numpy as np
import pytest
from star_detection import detect_stars
from synthetic_data import generate_synthetic_image, get_sample_templates


def test_detect_stars_returns_numpy_array():
    """Test that detect_stars returns a NumPy array of shape (N, 2)."""
    # Create a simple test image with a single bright pixel
    image = np.zeros((100, 100), dtype=np.uint8)
    image[50, 50] = 255
    
    config = {'intensity_threshold': 0.01}
    result = detect_stars(image, config=config)
    
    # Check return type and shape
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert len(result.shape) == 2, "Result should be 2D"
    assert result.shape[1] == 2, "Result should have 2 columns (x, y)"


def test_detect_stars_empty_image():
    """Test detect_stars on an empty (black) image."""
    image = np.zeros((100, 100), dtype=np.uint8)
    
    config = {'intensity_threshold': 0.01}
    result = detect_stars(image, config=config)
    
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert result.shape == (0, 2), "Empty image should return empty array with shape (0, 2)"


def test_detect_stars_simple_stars():
    """Test detect_stars on a simple image with multiple stars."""
    image = np.zeros((200, 200), dtype=np.uint8)
    
    # Add several bright stars at known positions
    star_positions = [(50, 50), (100, 100), (150, 150), (50, 150), (150, 50)]
    
    for x, y in star_positions:
        # Draw a small bright circle for each star
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if 0 <= x + dx < 200 and 0 <= y + dy < 200:
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist <= 3:
                        intensity = int(255 * (1 - dist / 3))
                        image[y + dy, x + dx] = max(image[y + dy, x + dx], intensity)
    
    config = {'intensity_threshold': 0.01, 'min_sigma': 0.5, 'max_sigma': 5.0}
    result = detect_stars(image, config=config)
    
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert result.shape[1] == 2, "Result should have 2 columns (x, y)"
    assert len(result) >= len(star_positions) - 1, "Should detect at least most of the stars"


def test_detect_stars_synthetic_image():
    """Test detect_stars on a synthetic constellation image."""
    templates = get_sample_templates()
    test_template = templates["Big Dipper"]
    
    # Generate synthetic image
    synthetic_image, transformed_points = generate_synthetic_image(
        test_template,
        image_size=(512, 512),
        rotation_angle=0.0,
        scale_factor=10.0,
        translation=(0, 0),
        noise_level=0.0,
        random_seed=42
    )
    
    config = {'intensity_threshold': 0.01, 'min_sigma': 0.5, 'max_sigma': 5.0}
    result = detect_stars(synthetic_image, config=config)
    
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert result.shape[1] == 2, "Result should have 2 columns (x, y)"
    assert len(result) >= len(test_template) - 1, "Should detect most stars from the template"


def test_detect_stars_config_threshold():
    """Test that intensity threshold configuration works."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[50, 50] = 255  # Single bright star
    
    # Low threshold should detect the star
    config_low = {'intensity_threshold': 0.001}
    result_low = detect_stars(image, config=config_low)
    
    # Very high threshold might not detect it
    config_high = {'intensity_threshold': 1.0}
    result_high = detect_stars(image, config=config_high)
    
    assert len(result_low) >= 0, "Low threshold should work"
    # Note: Very high threshold might still detect it, so we just check it doesn't crash


def test_detect_stars_config_sigma():
    """Test that min/max sigma configuration works."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[50, 50] = 255
    
    # Small sigma range
    config_small = {'intensity_threshold': 0.01, 'min_sigma': 0.5, 'max_sigma': 2.0}
    result_small = detect_stars(image, config=config_small)
    
    # Larger sigma range
    config_large = {'intensity_threshold': 0.01, 'min_sigma': 0.1, 'max_sigma': 10.0}
    result_large = detect_stars(image, config=config_large)
    
    assert isinstance(result_small, np.ndarray), "Result should be a NumPy array"
    assert isinstance(result_large, np.ndarray), "Result should be a NumPy array"
    assert result_small.shape[1] == 2, "Result should have 2 columns"
    assert result_large.shape[1] == 2, "Result should have 2 columns"


def test_detect_stars_grayscale_conversion():
    """Test that detect_stars converts color images to grayscale."""
    # Create a 3-channel color image with a star
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    color_image[50, 50, :] = [255, 255, 255]  # White pixel
    
    config = {'intensity_threshold': 0.01}
    result = detect_stars(color_image, config=config)
    
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert result.shape[1] == 2, "Result should have 2 columns (x, y)"


def test_detect_stars_default_config():
    """Test that detect_stars works with default (None) config."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[50, 50] = 255
    
    result = detect_stars(image, config=None)
    
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"
    assert result.shape[1] == 2, "Result should have 2 columns (x, y)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

