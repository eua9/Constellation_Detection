"""
Image preprocessing for robust star detection.

This module provides preprocessing functionality to normalize intensity,
denoise images, and optionally threshold background to prepare images
for star detection algorithms.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class Preprocessor:
    """
    Preprocess astronomical images for star detection.
    
    Performs normalization, denoising, and optional thresholding to prepare
    images for robust star detection algorithms.
    
    Parameters
    ----------
    blur_kernel : tuple of int, default=(3, 3)
        Kernel size for Gaussian blur (width, height). Must be odd numbers.
    gaussian_sigma : float, default=0
        Standard deviation for Gaussian blur. If 0, computed from kernel size.
    threshold_method : str, optional
        Thresholding method: 'simple', 'adaptive', 'otsu', or None.
        If None, no thresholding is applied.
    threshold_value : float, optional
        Threshold value for simple thresholding. Required if threshold_method='simple'.
    max_value : float, default=255
        Maximum value for thresholding output.
    """
    
    def __init__(
        self,
        blur_kernel: Tuple[int, int] = (3, 3),
        gaussian_sigma: float = 0,
        threshold_method: Optional[str] = None,
        threshold_value: Optional[float] = None,
        max_value: float = 255.0,
    ):
        self.blur_kernel = blur_kernel
        self.gaussian_sigma = gaussian_sigma
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.max_value = max_value
        
        # Validate kernel size (must be odd)
        if blur_kernel[0] % 2 == 0 or blur_kernel[1] % 2 == 0:
            raise ValueError("blur_kernel dimensions must be odd numbers")
        
        # Validate threshold method
        if threshold_method is not None:
            valid_methods = ['simple', 'adaptive', 'otsu']
            if threshold_method not in valid_methods:
                raise ValueError(
                    f"threshold_method must be one of {valid_methods}, got {threshold_method}"
                )
            
            if threshold_method == 'simple' and threshold_value is None:
                raise ValueError(
                    "threshold_value must be provided when threshold_method='simple'"
                )
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity to [0, 255] range.
        
        Parameters
        ----------
        img : np.ndarray
            Input image (any dtype, any range).
        
        Returns
        -------
        np.ndarray
            Normalized image as uint8 in [0, 255] range.
        """
        # Handle multi-channel images (convert to grayscale if needed)
        if len(img.shape) == 3:
            # If color image, convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img
        else:
            img_gray = img
        
        # Normalize to [0, 255]
        norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)
    
    def denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to denoise the image.
        
        Parameters
        ----------
        img : np.ndarray
            Input image (should be uint8 after normalization).
        
        Returns
        -------
        np.ndarray
            Denoised image.
        """
        # If sigma is 0, compute from kernel size
        sigma = self.gaussian_sigma
        if sigma == 0:
            sigma = 0.3 * ((min(self.blur_kernel) - 1) * 0.5 - 1) + 0.8
        
        return cv2.GaussianBlur(img, self.blur_kernel, sigma)
    
    def threshold(
        self,
        img: np.ndarray,
        method: Optional[str] = None,
        threshold_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply thresholding to remove background.
        
        Parameters
        ----------
        img : np.ndarray
            Input image (should be uint8, grayscale).
        method : str, optional
            Thresholding method. If None, uses self.threshold_method.
        threshold_value : float, optional
            Threshold value for simple thresholding. If None, uses self.threshold_value.
        
        Returns
        -------
        np.ndarray
            Thresholded binary image.
        """
        method = method or self.threshold_method
        threshold_value = threshold_value or self.threshold_value
        
        if method is None:
            return img
        
        if method == 'simple':
            if threshold_value is None:
                raise ValueError("threshold_value must be provided for simple thresholding")
            _, thresh = cv2.threshold(
                img, threshold_value, self.max_value, cv2.THRESH_BINARY
            )
            return thresh
        
        elif method == 'adaptive':
            # Adaptive thresholding using mean
            thresh = cv2.adaptiveThreshold(
                img,
                self.max_value,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                11,  # block size
                2,   # C constant
            )
            return thresh
        
        elif method == 'otsu':
            # Otsu's method for automatic threshold selection
            _, thresh = cv2.threshold(
                img, 0, self.max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return thresh
        
        else:
            raise ValueError(f"Unknown threshold method: {method}")
    
    def run(
        self,
        img: np.ndarray,
        apply_threshold: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Run the complete preprocessing pipeline.
        
        Parameters
        ----------
        img : np.ndarray
            Input image (any dtype, any range).
        apply_threshold : bool, optional
            Whether to apply thresholding. If None, uses self.threshold_method to decide.
        
        Returns
        -------
        np.ndarray
            Preprocessed image ready for star detection.
        """
        # Normalize intensity
        img = self.normalize(img)
        
        # Denoise
        img = self.denoise(img)
        
        # Optionally threshold
        if apply_threshold is None:
            apply_threshold = self.threshold_method is not None
        
        if apply_threshold:
            img = self.threshold(img)
        
        return img

