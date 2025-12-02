"""
Synthetic Data Generation Module

This module generates synthetic star field images based on known constellation
templates with configurable transformations (rotation, scale, translation, noise).
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def generate_synthetic_image(
    template_points: np.ndarray,
    image_size: Tuple[int, int] = (512, 512),
    rotation_angle: float = 0.0,
    scale_factor: float = 1.0,
    translation: Tuple[float, float] = (0.0, 0.0),
    noise_level: float = 0.0,
    add_clutter: bool = False,
    clutter_stars: int = 0,
    remove_stars: Optional[List[int]] = None,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic star field image from a template.

    Args:
        template_points: Array of (x, y) coordinates defining the constellation template
        image_size: (width, height) of the output image
        rotation_angle: Rotation angle in degrees
        scale_factor: Scale factor to apply
        translation: (dx, dy) translation offset
        noise_level: Standard deviation of Gaussian noise to add to star positions
        add_clutter: Whether to add random clutter stars
        clutter_stars: Number of clutter stars to add
        remove_stars: List of indices of stars to remove (for occlusion simulation)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (image, transformed_points)
        - image: Synthetic star field image (grayscale)
        - transformed_points: The transformed star coordinates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Start with template points
    points = template_points.copy()
    
    # Apply transformations
    # 1. Scale
    points = points * scale_factor
    
    # 2. Rotation
    if rotation_angle != 0:
        angle_rad = np.deg2rad(rotation_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        points = points @ rotation_matrix.T
    
    # 3. Translation
    points = points + np.array(translation)
    
    # 4. Center in image
    img_center = np.array([image_size[0] / 2, image_size[1] / 2])
    points = points + img_center
    
    # 5. Add noise to positions
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, points.shape)
        points = points + noise
    
    # 6. Remove specified stars (occlusion)
    if remove_stars is not None:
        points = np.delete(points, remove_stars, axis=0)
    
    # Create blank image
    image = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    # Draw stars as small bright circles
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            cv2.circle(image, (x, y), 2, 255, -1)
            cv2.circle(image, (x, y), 4, 200, 1)
    
    # Add clutter stars
    if add_clutter and clutter_stars > 0:
        clutter_x = np.random.randint(0, image_size[0], clutter_stars)
        clutter_y = np.random.randint(0, image_size[1], clutter_stars)
        for x, y in zip(clutter_x, clutter_y):
            cv2.circle(image, (x, y), 2, 255, -1)
            cv2.circle(image, (x, y), 4, 200, 1)
    
    return image, points


def create_template(
    star_positions: List[Tuple[float, float]],
    name: str = "unknown"
) -> np.ndarray:
    """
    Create a constellation template from star positions.

    Args:
        star_positions: List of (x, y) coordinates relative to constellation center
        name: Name of the constellation template

    Returns:
        Template points array of shape (n_stars, 2)
    """
    return np.array(star_positions)


def get_sample_templates() -> dict:
    """
    Get sample constellation templates for testing.

    Returns:
        Dictionary mapping constellation names to template point arrays
    """
    templates = {}
    
    # Simple "Big Dipper" pattern (7 stars forming a dipper shape)
    big_dipper = np.array([
        [0, 0],      # Handle end
        [5, 5],      # Handle middle
        [10, 10],    # Handle base
        [15, 12],    # Bowl corner
        [20, 10],    # Bowl middle
        [25, 8],     # Bowl corner
        [30, 5]      # Bowl tip
    ])
    templates["Big Dipper"] = big_dipper
    
    # Simple "Orion" pattern (3 stars in a line with 2 off to the side)
    orion = np.array([
        [-5, 0],     # Left shoulder
        [0, -5],     # Head
        [5, 0],      # Right shoulder
        [0, 5],      # Belt center
        [-5, 10],    # Left foot
        [5, 10]      # Right foot
    ])
    templates["Orion"] = orion
    
    return templates

