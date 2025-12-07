"""
Synthetic Data Generation Module

This module generates synthetic star field images based on known constellation
templates with configurable transformations (rotation, scale, translation, noise).
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


def generate_constellation_instance(
    template_points: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Generate transformed constellation coordinates from a template.
    
    Applies random rotation, scale, translation, positional noise, and optionally
    removes some stars and adds random clutter stars.
    
    Args:
        template_points: Array of shape (n_stars, 2) with normalized (x, y) coordinates
        params: Dictionary containing transformation parameters:
            - 'rotation_range': Tuple (min, max) rotation angle in degrees, or None
            - 'scale_range': Tuple (min, max) scale factor, or None
            - 'translation_range': Tuple ((dx_min, dx_max), (dy_min, dy_max)), or None
            - 'noise_std': Standard deviation of Gaussian noise for positions
            - 'remove_prob': Probability of removing each star (0.0 to 1.0)
            - 'clutter_count': Number of random clutter stars to add
            - 'image_size': Tuple (width, height) for centering
            - 'random_seed': Optional random seed for reproducibility
    
    Returns:
        Array of shape (n_stars, 2) with transformed coordinates in image space
    """
    # Set random seed if provided
    random_seed = params.get('random_seed')
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Start with template points copy
    points = template_points.copy().astype(np.float64)
    
    # Apply transformations in order: scale -> rotation -> translation -> centering -> noise
    
    # 1. Scale
    scale_range = params.get('scale_range')
    if scale_range is not None:
        scale_min, scale_max = scale_range
        scale_factor = np.random.uniform(scale_min, scale_max)
        points = points * scale_factor
    else:
        scale_factor = params.get('scale_factor', 1.0)
        points = points * scale_factor
    
    # 2. Rotation
    rotation_range = params.get('rotation_range')
    if rotation_range is not None:
        angle_min, angle_max = rotation_range
        rotation_angle = np.random.uniform(angle_min, angle_max)
    else:
        rotation_angle = params.get('rotation_angle', 0.0)
    
    if rotation_angle != 0:
        angle_rad = np.deg2rad(rotation_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        points = points @ rotation_matrix.T
    
    # 3. Translation
    translation_range = params.get('translation_range')
    if translation_range is not None:
        (dx_min, dx_max), (dy_min, dy_max) = translation_range
        dx = np.random.uniform(dx_min, dx_max)
        dy = np.random.uniform(dy_min, dy_max)
        translation = np.array([dx, dy])
    else:
        translation = np.array(params.get('translation', [0.0, 0.0]))
    
    points = points + translation
    
    # 4. Center in image
    image_size = params.get('image_size', (512, 512))
    img_center = np.array([image_size[0] / 2.0, image_size[1] / 2.0])
    points = points + img_center
    
    # 5. Add positional noise
    noise_std = params.get('noise_std', 0.0)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, points.shape)
        points = points + noise
    
    # 6. Remove stars (occlusion simulation)
    remove_prob = params.get('remove_prob', 0.0)
    if remove_prob > 0:
        # Randomly remove stars based on probability
        keep_mask = np.random.random(len(points)) > remove_prob
        points = points[keep_mask]
    
    return points


def render_star_image(
    points: np.ndarray,
    image_size: Tuple[int, int],
    star_radius: int = 2,
    noise_settings: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Render a grayscale image from star coordinates.
    
    Args:
        points: Array of shape (n_stars, 2) with (x, y) coordinates in image space
        image_size: Tuple (width, height) of the output image
        star_radius: Radius of each star in pixels (default: 2)
        noise_settings: Optional dictionary with image noise settings:
            - 'background_noise': Standard deviation of background Gaussian noise
            - 'random_seed': Optional random seed for reproducibility
    
    Returns:
        Grayscale image as numpy array of shape (height, width) with dtype uint8
    """
    # Set random seed if provided
    if noise_settings is not None:
        random_seed = noise_settings.get('random_seed')
        if random_seed is not None:
            np.random.seed(random_seed)
    
    # Create blank image
    image = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    # Draw stars as bright circles
    for point in points:
        x, y = int(round(point[0])), int(round(point[1]))
        # Only draw if within image bounds
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            # Draw filled circle (bright center)
            cv2.circle(image, (x, y), star_radius, 255, -1)
            # Draw outer ring for more realistic star appearance
            if star_radius > 1:
                cv2.circle(image, (x, y), star_radius + 1, 200, 1)
    
    # Add background noise if specified
    if noise_settings is not None:
        background_noise = noise_settings.get('background_noise', 0.0)
        if background_noise > 0:
            noise = np.random.normal(0, background_noise, image.shape)
            image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    return image


def generate_dataset(
    num_images: int,
    templates: Dict[str, np.ndarray],
    config: Dict[str, Any],
    output_dir: str,
    random_seed: Optional[int] = None
) -> None:
    """
    Generate a synthetic dataset of constellation images.
    
    For each image:
    1. Randomly samples a template
    2. Generates transformed coordinates using generate_constellation_instance
    3. Renders the image using render_star_image
    4. Saves the image and metadata (constellation name, true coordinates)
    
    Args:
        num_images: Number of images to generate
        templates: Dictionary mapping constellation names to template point arrays
        config: Configuration dictionary with parameters for generation:
            - 'image_size': Tuple (width, height)
            - 'star_radius': Radius of stars in pixels
            - 'rotation_range': Tuple (min, max) rotation in degrees
            - 'scale_range': Tuple (min, max) scale factor
            - 'translation_range': Tuple ((dx_min, dx_max), (dy_min, dy_max))
            - 'noise_std': Standard deviation of positional noise
            - 'remove_prob': Probability of removing each star
            - 'clutter_count': Number of clutter stars to add
            - 'noise_settings': Dictionary for render_star_image noise settings
        output_dir: Directory path where images and metadata will be saved
        random_seed: Optional random seed for reproducibility (NFR3)
    """
    if len(templates) == 0:
        raise ValueError("Cannot generate dataset: no templates provided")
    
    # Set random seed if provided (only once at the beginning)
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract configuration parameters
    image_size = config.get('image_size', (512, 512))
    star_radius = config.get('star_radius', 2)
    noise_settings = config.get('noise_settings', {})
    
    # Prepare template names list for random sampling
    template_names = list(templates.keys())
    
    # Generate each image
    for i in range(num_images):
        # Sample a random template (random state advances naturally)
        template_name = np.random.choice(template_names)
        template_points = templates[template_name]
        
        # Prepare parameters for generate_constellation_instance
        # Don't pass random_seed - let the random state advance naturally
        # This ensures each image gets different random transformations
        instance_params = {
            'rotation_range': config.get('rotation_range'),
            'scale_range': config.get('scale_range'),
            'translation_range': config.get('translation_range'),
            'noise_std': config.get('noise_std', 0.0),
            'remove_prob': config.get('remove_prob', 0.0),
            'image_size': image_size,
            'random_seed': None  # Don't reset seed - let random state advance
        }
        
        # Generate transformed coordinates
        transformed_points = generate_constellation_instance(template_points, instance_params)
        
        # Add clutter stars if specified
        clutter_count = config.get('clutter_count', 0)
        if clutter_count > 0:
            clutter_x = np.random.uniform(0, image_size[0], clutter_count)
            clutter_y = np.random.uniform(0, image_size[1], clutter_count)
            clutter_points = np.column_stack([clutter_x, clutter_y])
            transformed_points = np.vstack([transformed_points, clutter_points])
        
        # Render the image
        image = render_star_image(
            transformed_points,
            image_size,
            star_radius,
            noise_settings
        )
        
        # Save image
        image_filename = f"constellation_{i:04d}.png"
        image_path = output_path / image_filename
        cv2.imwrite(str(image_path), image)
        
        # Save metadata
        metadata = {
            'image_filename': image_filename,
            'constellation_name': template_name,
            'num_stars': len(transformed_points),
            'true_coordinates': transformed_points.tolist(),
            'image_size': image_size,
            'generation_params': {
                'rotation_range': config.get('rotation_range'),
                'scale_range': config.get('scale_range'),
                'translation_range': config.get('translation_range'),
                'noise_std': config.get('noise_std', 0.0),
                'remove_prob': config.get('remove_prob', 0.0),
                'clutter_count': clutter_count
            }
        }
        
        metadata_filename = f"constellation_{i:04d}_metadata.json"
        metadata_path = output_path / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Generated {num_images} synthetic constellation images in {output_dir}")
    print(f"Templates used: {set(template_names)}")
