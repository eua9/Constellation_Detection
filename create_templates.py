#!/usr/bin/env python3
"""
Script to create initial toy constellation templates for testing.

This script generates a few simple constellation templates and saves them
in the templates_data/ directory, then updates the configuration file.
"""

import numpy as np
from pathlib import Path
from templates import save_template, create_template_config
from synthetic_data import get_sample_templates


def create_initial_templates():
    """
    Create initial toy templates for testing.
    """
    print("=" * 60)
    print("Creating Initial Constellation Templates")
    print("=" * 60)
    
    # Create templates directory
    templates_dir = Path("templates_data")
    templates_dir.mkdir(exist_ok=True)
    print(f"\nTemplates directory: {templates_dir.absolute()}")
    
    # Get sample templates from synthetic_data
    sample_templates = get_sample_templates()
    
    # Template paths dictionary (name -> path)
    template_paths = {}
    
    # Save each template
    print("\nCreating templates...")
    for name, points in sample_templates.items():
        template_path = templates_dir / f"{name.lower().replace(' ', '_')}.json"
        template_path_str = str(template_path)
        
        # Save template (will be normalized automatically)
        save_template(name, points, template_path_str, normalize=True)
        
        # Store path for config
        template_paths[name] = f"templates_data/{template_path.name}"
    
    # Create additional simple test templates
    print("\nCreating additional test templates...")
    
    # Simple triangle template
    triangle_points = np.array([
        [0, 2],      # Top
        [-1, 0],     # Bottom left
        [1, 0]       # Bottom right
    ])
    triangle_path = templates_dir / "triangle.json"
    save_template("Triangle", triangle_points, str(triangle_path), normalize=True)
    template_paths["Triangle"] = "templates_data/triangle.json"
    
    # Simple line template (3 stars in a row)
    line_points = np.array([
        [-2, 0],
        [0, 0],
        [2, 0]
    ])
    line_path = templates_dir / "line.json"
    save_template("Line", line_points, str(line_path), normalize=True)
    template_paths["Line"] = "templates_data/line.json"
    
    # L-shaped template
    l_shape_points = np.array([
        [0, 0],
        [0, 2],
        [1, 2],
        [1, 3]
    ])
    l_shape_path = templates_dir / "l_shape.json"
    save_template("L-Shape", l_shape_points, str(l_shape_path), normalize=True)
    template_paths["L-Shape"] = "templates_data/l_shape.json"
    
    # Create or update configuration file
    config_path = "templates_config.json"
    create_template_config(template_paths, config_path)
    
    print("\n" + "=" * 60)
    print(f"Created {len(template_paths)} templates:")
    for name in template_paths.keys():
        print(f"  - {name}")
    print(f"\nConfiguration file: {config_path}")
    print("=" * 60)


if __name__ == "__main__":
    create_initial_templates()

