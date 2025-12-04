"""
Configuration Loader Module

This module provides utilities for loading and managing configuration parameters
from JSON configuration files, ensuring reproducible setup (NFR3).
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration parameters from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file. Default: "config.json"
    
    Returns:
        Dictionary containing configuration parameters organized by category:
        - 'paths': File and directory paths
        - 'detection': Star detection parameters
        - 'matching': Template matching parameters
        - 'synthetic_data': Synthetic data generation parameters
        - 'evaluation': Evaluation parameters
        - 'batch_processing': Batch processing parameters
    
    Raises:
        FileNotFoundError: If the configuration file does not exist
        json.JSONDecodeError: If the configuration file is not valid JSON
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config


def get_detection_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract star detection configuration from the main config dictionary.
    
    Args:
        config: Optional configuration dictionary. If None, loads from default config.json
    
    Returns:
        Dictionary with star detection parameters:
        - 'intensity_threshold': Threshold for blob detection
        - 'min_sigma': Minimum blob size
        - 'max_sigma': Maximum blob size
        - 'num_sigma': Number of intermediate sigma values
    """
    if config is None:
        config = load_config()
    
    return config.get('detection', {
        'intensity_threshold': 0.01,
        'min_sigma': 1.0,
        'max_sigma': 30.0,
        'num_sigma': 10
    })


def get_matching_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract template matching configuration from the main config dictionary.
    
    Args:
        config: Optional configuration dictionary. If None, loads from default config.json
    
    Returns:
        Dictionary with matching parameters:
        - 'no_match_threshold': Optional threshold for declaring no match
        - 'method': Matching method ('ssd' for baseline)
    """
    if config is None:
        config = load_config()
    
    return config.get('matching', {
        'no_match_threshold': None,
        'method': 'ssd'
    })


def get_synthetic_data_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract synthetic data generation configuration from the main config dictionary.
    
    Args:
        config: Optional configuration dictionary. If None, loads from default config.json
    
    Returns:
        Dictionary with synthetic data generation parameters including:
        - 'image_size': Output image dimensions [width, height]
        - 'star_radius': Radius of stars in pixels
        - 'rotation_range': [min, max] rotation angles in degrees
        - 'scale_range': [min, max] scale factors
        - 'translation_range': [[dx_min, dx_max], [dy_min, dy_max]]
        - 'noise_std': Standard deviation of positional noise
        - 'remove_prob': Probability of removing each star
        - 'clutter_count': Number of clutter stars to add
        - 'background_noise': Standard deviation of background noise
    """
    if config is None:
        config = load_config()
    
    return config.get('synthetic_data', {})

