"""
Star detection module for constellation detection.

This module implements star detection using thresholding and blob detection
to identify star positions in preprocessed astronomical images.
"""

import numpy as np
import cv2
from typing import Optional


class StarDetector:
    """
    Detect star positions in preprocessed images using blob detection.
    
    Uses OpenCV's SimpleBlobDetector to identify star-like features after
    thresholding. Returns star centroids as (x, y) coordinates.
    
    Parameters
    ----------
    threshold : int, default=200
        Threshold value for binarization (0-255). Pixels above this value
        are considered potential stars.
    min_area : float, default=5.0
        Minimum blob area in pixels. Smaller blobs are filtered out.
    max_area : float, optional
        Maximum blob area in pixels. If None, no maximum is applied.
    filter_by_circularity : bool, default=False
        Whether to filter blobs by circularity.
    min_circularity : float, default=0.1
        Minimum circularity (0-1). 1.0 is a perfect circle.
    filter_by_convexity : bool, default=False
        Whether to filter blobs by convexity.
    min_convexity : float, default=0.87
        Minimum convexity (0-1). 1.0 is fully convex.
    filter_by_inertia : bool, default=False
        Whether to filter blobs by inertia ratio.
    min_inertia_ratio : float, default=0.01
        Minimum inertia ratio (0-1). Higher values prefer more elongated shapes.
    """
    
    def __init__(
        self,
        threshold: int = 200,
        min_area: float = 5.0,
        max_area: Optional[float] = None,
        filter_by_circularity: bool = False,
        min_circularity: float = 0.1,
        filter_by_convexity: bool = False,
        min_convexity: float = 0.87,
        filter_by_inertia: bool = False,
        min_inertia_ratio: float = 0.01,
    ):
        self.threshold = threshold
        
        # Configure blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Area filtering
        params.filterByArea = True
        params.minArea = min_area
        if max_area is not None:
            params.maxArea = max_area
        else:
            params.maxArea = float('inf')
        
        # Color filtering (detect white blobs on black background)
        params.filterByColor = True
        params.blobColor = 255
        
        # Circularity filtering
        params.filterByCircularity = filter_by_circularity
        params.minCircularity = min_circularity
        
        # Convexity filtering
        params.filterByConvexity = filter_by_convexity
        params.minConvexity = min_convexity
        
        # Inertia filtering
        params.filterByInertia = filter_by_inertia
        params.minInertiaRatio = min_inertia_ratio
        
        # Create detector
        self.detector = cv2.SimpleBlobDetector_create(params)
    
    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """
        Binarize image using threshold.
        
        Parameters
        ----------
        img : np.ndarray
            Input image (should be uint8, grayscale).
        
        Returns
        -------
        np.ndarray
            Binary image (0 or 255).
        """
        _, binary = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Detect star positions in a preprocessed image.
        
        Parameters
        ----------
        img : np.ndarray
            Preprocessed image (should be uint8, grayscale).
            Typically the output from Preprocessor.run().
        
        Returns
        -------
        np.ndarray
            Star centroids as array of shape (N, 2) with (x, y) coordinates.
            Returns empty array of shape (0, 2) if no stars are detected.
        """
        # Ensure image is uint8
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Handle multi-channel images (convert to grayscale if needed)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binarize image
        binary = self._binarize(img)
        
        # Detect blobs
        keypoints = self.detector.detect(binary)
        
        # Extract centroids
        if len(keypoints) == 0:
            return np.empty((0, 2), dtype=np.float32)
        
        centroids = np.array(
            [[kp.pt[0], kp.pt[1]] for kp in keypoints],
            dtype=np.float32
        )
        
        return centroids

