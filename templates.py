"""
Template Storage Module

This module handles loading and saving constellation templates as normalized point sets.
Templates are stored in JSON format with constellation names and normalized coordinates.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from normalization import normalize_star_points


def save_template(
    name: str,
    points: np.ndarray,
    path: str,
    normalize: bool = True
) -> None:
    """
    Save a constellation template to a JSON file.
    
    Templates are stored in JSON format with:
    - constellation_name: Name of the constellation
    - normalized_coordinates: List of [x, y] coordinate pairs (normalized)
    - num_stars: Number of stars in the template
    
    Args:
        name: Name of the constellation
        points: Array of shape (n_points, 2) with (x, y) coordinates
        path: File path where the template should be saved
        normalize: If True, normalize points before saving (default: True)
    """
    # Ensure points is a numpy array
    points = np.array(points)
    
    if len(points) == 0:
        raise ValueError("Cannot save template with no points")
    
    # Normalize points if requested
    if normalize:
        normalized_points = normalize_star_points(points)
    else:
        normalized_points = points
    
    # Create template dictionary
    template_data = {
        "constellation_name": name,
        "num_stars": len(normalized_points),
        "normalized_coordinates": normalized_points.tolist()
    }
    
    # Ensure directory exists
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to JSON file
    with open(path, 'w') as f:
        json.dump(template_data, f, indent=2)
    
    print(f"Saved template '{name}' with {len(normalized_points)} stars to {path}")


def load_template(path: str) -> Tuple[str, np.ndarray]:
    """
    Load a single constellation template from a JSON file.
    
    Args:
        path: File path to the template JSON file
    
    Returns:
        Tuple of (constellation_name, normalized_points_array)
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Template file not found: {path}")
    
    # Read JSON file
    with open(path, 'r') as f:
        template_data = json.load(f)
    
    # Extract data
    name = template_data.get("constellation_name", "unknown")
    coordinates = template_data.get("normalized_coordinates", [])
    
    # Convert to numpy array
    points = np.array(coordinates, dtype=np.float64)
    
    if len(points) == 0:
        raise ValueError(f"Template '{name}' contains no coordinates")
    
    return name, points


def load_templates(config_path: str) -> Dict[str, np.ndarray]:
    """
    Load all templates listed in the configuration file into memory.
    
    Configuration file format (JSON):
    {
        "templates": [
            {
                "name": "Constellation Name",
                "path": "path/to/template.json"
            },
            ...
        ]
    }
    
    Args:
        config_path: Path to the configuration file listing templates
    
    Returns:
        Dictionary mapping constellation names to normalized point arrays
    """
    config_path_obj = Path(config_path)
    
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    templates = {}
    template_list = config.get("templates", [])
    
    for template_info in template_list:
        template_name = template_info.get("name")
        template_path = template_info.get("path")
        
        if not template_name or not template_path:
            print(f"Warning: Skipping template with missing name or path: {template_info}")
            continue
        
        try:
            # Resolve relative paths relative to config file location
            if not Path(template_path).is_absolute():
                template_path = config_path_obj.parent / template_path
                template_path = str(template_path.resolve())
            
            loaded_name, loaded_points = load_template(template_path)
            
            # Use name from config if different (allow override)
            if template_name != loaded_name:
                print(f"Warning: Template name mismatch. Config: '{template_name}', File: '{loaded_name}'. Using config name.")
            
            templates[template_name] = loaded_points
            print(f"Loaded template '{template_name}' from {template_path} ({len(loaded_points)} stars)")
            
        except Exception as e:
            print(f"Error loading template '{template_name}' from {template_path}: {e}")
            continue
    
    print(f"\nLoaded {len(templates)} templates: {list(templates.keys())}")
    return templates


def create_template_config(
    templates: Dict[str, str],
    output_path: str
) -> None:
    """
    Create a template configuration file from a dictionary of name->path mappings.
    
    Args:
        templates: Dictionary mapping template names to file paths
        output_path: Path where the configuration file should be saved
    """
    config_data = {
        "templates": [
            {"name": name, "path": path}
            for name, path in templates.items()
        ]
    }
    
    # Ensure directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration file
    with open(output_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Created template configuration file: {output_path}")

