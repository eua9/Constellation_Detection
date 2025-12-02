"""
Star Detection Module

This module handles the detection of star-like blobs in astronomical images
and extraction of star centroids as 2D coordinates.
"""

import numpy as np
import cv2
from skimage.feature import blob_log
from typing import List, Tuple, Optional, Dict, Any


def detect_stars(
    image: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Detect star-like blobs in a grayscale astronomical image and output their 
    centroids as 2D coordinates.

    Args:
        image: Input image (grayscale or color numpy array)
        config: Configuration dictionary with the following optional keys:
            - 'intensity_threshold' (float): Threshold for blob detection. 
              Default: 0.1
            - 'min_sigma' (float): Minimum blob size (standard deviation). 
              Default: 1.0
            - 'max_sigma' (float): Maximum blob size (standard deviation). 
              Default: 30.0
            - 'num_sigma' (int): Number of intermediate sigma values. 
              Default: 10
            - 'min_area' (float): Minimum blob area in pixels (optional).
            - 'max_area' (float): Maximum blob area in pixels (optional).

    Returns:
        NumPy array of shape (N, 2) containing star centroids as (x, y) coordinates
    """
    # Set default configuration
    if config is None:
        config = {}
    
    # Extract configuration parameters with defaults
    intensity_threshold = config.get('intensity_threshold', 0.1)
    min_sigma = config.get('min_sigma', 1.0)
    max_sigma = config.get('max_sigma', 30.0)
    num_sigma = config.get('num_sigma', 10)
    min_area = config.get('min_area', None)
    max_area = config.get('max_area', None)
    
    # a. Convert image to grayscale (if needed)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize image if needed
    if image.dtype != np.uint8:
        image = (image - image.min()) / (image.max() - image.min() + 1e-10)
        image = (image * 255).astype(np.uint8)
    
    # b. Apply thresholding / blob detection using scikit-image blob_log
    blobs = blob_log(
        image,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=intensity_threshold
    )
    
    # Filter blobs by sigma (blob size)
    if len(blobs) > 0:
        # Filter by min_sigma
        valid_mask = blobs[:, 2] >= min_sigma
        # Filter by max_sigma
        valid_mask = valid_mask & (blobs[:, 2] <= max_sigma)
        blobs = blobs[valid_mask]
    
    # Filter by area if specified
    if len(blobs) > 0 and (min_area is not None or max_area is not None):
        # Approximate area as pi * (2 * sigma)^2 for circular blobs
        areas = np.pi * (2 * blobs[:, 2]) ** 2
        valid_mask = np.ones(len(blobs), dtype=bool)
        
        if min_area is not None:
            valid_mask = valid_mask & (areas >= min_area)
        if max_area is not None:
            valid_mask = valid_mask & (areas <= max_area)
        
        blobs = blobs[valid_mask]
    
    # c. Extract blob centers as (x, y) coordinates
    if len(blobs) > 0:
        # blob_log returns [y, x, sigma], so we extract [x, y]
        centroids = np.column_stack([blobs[:, 1], blobs[:, 0]])
    else:
        # Return empty array with correct shape if no blobs found
        centroids = np.empty((0, 2), dtype=np.float64)
    
    return centroids


def visualize_detection(
    image: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """
    Overlay detected stars on the input image for visualization.

    Args:
        image: Input grayscale or color image
        centroids: NumPy array of shape (N, 2) containing (x, y) star centroid coordinates

    Returns:
        Visualization image with detected stars marked
    """
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Handle both NumPy arrays and lists/tuples for backward compatibility
    if isinstance(centroids, np.ndarray):
        if centroids.shape[1] == 2:
            for i in range(len(centroids)):
                x, y = centroids[i, 0], centroids[i, 1]
                cv2.circle(vis_image, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 0), 1)
    else:
        # Fallback for list/tuple format
        for x, y in centroids:
            cv2.circle(vis_image, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 0), 1)
    
    return vis_image

