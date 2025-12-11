"""
Test script for synthetic_data.py module
"""

import numpy as np
from synthetic_data import (
    generate_constellation_instance,
    render_star_image,
    generate_dataset
)
from templates import load_templates
import os
from pathlib import Path


def test_generate_constellation_instance():
    """Test generate_constellation_instance function"""
    print("Testing generate_constellation_instance...")
    
    # Create a simple template (triangle)
    template = np.array([
        [0, 0],
        [10, 0],
        [5, 10]
    ])
    
    # Test parameters
    params = {
        'rotation_range': (0, 360),
        'scale_range': (0.5, 2.0),
        'translation_range': ((-50, 50), (-50, 50)),
        'noise_std': 1.0,
        'remove_prob': 0.1,
        'image_size': (512, 512),
        'random_seed': 42
    }
    
    # Generate instance
    transformed = generate_constellation_instance(template, params)
    
    assert transformed.shape[1] == 2, "Output should have 2 columns (x, y)"
    assert len(transformed) <= len(template), "Should have same or fewer points after removal"
    print(f"  ✓ Generated {len(transformed)} points from {len(template)} template points")
    print(f"  ✓ Point range: x=[{transformed[:, 0].min():.1f}, {transformed[:, 0].max():.1f}], "
          f"y=[{transformed[:, 1].min():.1f}, {transformed[:, 1].max():.1f}]")


def test_render_star_image():
    """Test render_star_image function"""
    print("\nTesting render_star_image...")
    
    # Create some test points
    points = np.array([
        [100, 100],
        [200, 150],
        [300, 200]
    ])
    
    image_size = (512, 512)
    star_radius = 3
    
    # Render image
    image = render_star_image(points, image_size, star_radius)
    
    assert image.shape == (512, 512), f"Image shape should be (512, 512), got {image.shape}"
    assert image.dtype == np.uint8, f"Image dtype should be uint8, got {image.dtype}"
    assert image.max() <= 255, "Image values should be <= 255"
    print(f"  ✓ Rendered image with shape {image.shape}, dtype {image.dtype}")
    print(f"  ✓ Image value range: [{image.min()}, {image.max()}]")


def test_generate_dataset():
    """Test generate_dataset function"""
    print("\nTesting generate_dataset...")
    
    # Load templates
    try:
        templates = load_templates('templates_config.json')
        print(f"  ✓ Loaded {len(templates)} templates")
    except Exception as e:
        print(f"  ⚠ Could not load templates: {e}")
        # Create a simple test template
        templates = {
            'Test Triangle': np.array([
                [0, 0],
                [10, 0],
                [5, 10]
            ])
        }
        print(f"  ✓ Using test template")
    
    # Configuration
    config = {
        'image_size': (512, 512),
        'star_radius': 2,
        'rotation_range': (0, 360),
        'scale_range': (0.5, 2.0),
        'translation_range': ((-50, 50), (-50, 50)),
        'noise_std': 1.0,
        'remove_prob': 0.1,
        'clutter_count': 5,
        'noise_settings': {
            'background_noise': 5.0
        }
    }
    
    # Test output directory
    output_dir = 'test_synthetic_output'
    
    # Generate small dataset
    num_images = 5
    generate_dataset(num_images, templates, config, output_dir, random_seed=42)
    
    # Verify files were created
    output_path = Path(output_dir)
    image_files = list(output_path.glob('constellation_*.png'))
    metadata_files = list(output_path.glob('constellation_*_metadata.json'))
    
    assert len(image_files) == num_images, f"Expected {num_images} images, found {len(image_files)}"
    assert len(metadata_files) == num_images, f"Expected {num_images} metadata files, found {len(metadata_files)}"
    
    print(f"  ✓ Generated {len(image_files)} images and {len(metadata_files)} metadata files")
    
    # Check one metadata file
    import json
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
        assert 'constellation_name' in metadata
        assert 'true_coordinates' in metadata
        print(f"  ✓ Metadata structure is correct")
        print(f"    Example: {metadata['constellation_name']} with {metadata['num_stars']} stars")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing synthetic_data.py module")
    print("=" * 60)
    
    try:
        test_generate_constellation_instance()
        test_render_star_image()
        test_generate_dataset()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

