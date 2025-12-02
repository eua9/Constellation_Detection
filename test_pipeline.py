#!/usr/bin/env python3
"""
Test suite for the end-to-end constellation detection pipeline.

Tests the pipeline on synthetic images where ground truth constellation is known.
"""

import numpy as np
from main import run_pipeline
from synthetic_data import generate_synthetic_image, get_sample_templates


def test_pipeline_big_dipper():
    """Test pipeline on Big Dipper synthetic image."""
    templates = get_sample_templates()
    test_template = templates["Big Dipper"]
    
    # Generate synthetic image with transformations
    synthetic_image, _ = generate_synthetic_image(
        test_template,
        image_size=(512, 512),
        rotation_angle=0.0,
        scale_factor=10.0,
        translation=(0, 0),
        noise_level=0.0,
        random_seed=42
    )
    
    match, score, centroids = run_pipeline(
        synthetic_image,
        template_config_path="templates_config.json",
        detection_config={'intensity_threshold': 0.01}
    )
    
    assert match == "Big Dipper", f"Expected 'Big Dipper', got '{match}'"
    assert score < 1.0, f"Score should be reasonable, got {score}"
    assert len(centroids) > 0, "Should detect at least some stars"
    
    print(f"✓ Big Dipper test passed: Match={match}, Score={score:.4f}, Stars={len(centroids)}")


def test_pipeline_orion():
    """Test pipeline on Orion synthetic image."""
    templates = get_sample_templates()
    test_template = templates["Orion"]
    
    synthetic_image, _ = generate_synthetic_image(
        test_template,
        image_size=(512, 512),
        rotation_angle=0.0,
        scale_factor=10.0,
        translation=(0, 0),
        noise_level=0.0,
        random_seed=42
    )
    
    match, score, centroids = run_pipeline(
        synthetic_image,
        template_config_path="templates_config.json",
        detection_config={'intensity_threshold': 0.01}
    )
    
    assert match == "Orion", f"Expected 'Orion', got '{match}'"
    assert score < 1.0, f"Score should be reasonable, got {score}"
    assert len(centroids) > 0, "Should detect at least some stars"
    
    print(f"✓ Orion test passed: Match={match}, Score={score:.4f}, Stars={len(centroids)}")


def test_pipeline_with_rotation():
    """Test pipeline with rotated constellation."""
    templates = get_sample_templates()
    test_template = templates["Big Dipper"]
    
    # Generate with 45-degree rotation
    synthetic_image, _ = generate_synthetic_image(
        test_template,
        image_size=(512, 512),
        rotation_angle=45.0,
        scale_factor=10.0,
        translation=(0, 0),
        noise_level=0.0,
        random_seed=42
    )
    
    match, score, centroids = run_pipeline(
        synthetic_image,
        template_config_path="templates_config.json",
        detection_config={'intensity_threshold': 0.01}
    )
    
    # Should still match despite rotation (normalization handles it)
    assert match is not None, "Should find a match even with rotation"
    assert len(centroids) > 0, "Should detect stars"
    
    print(f"✓ Rotation test passed: Match={match}, Score={score:.4f}, Stars={len(centroids)}")


def test_pipeline_with_scale():
    """Test pipeline with scaled constellation."""
    templates = get_sample_templates()
    test_template = templates["Orion"]
    
    # Generate with different scale
    synthetic_image, _ = generate_synthetic_image(
        test_template,
        image_size=(512, 512),
        rotation_angle=0.0,
        scale_factor=15.0,  # Larger scale
        translation=(0, 0),
        noise_level=0.0,
        random_seed=42
    )
    
    match, score, centroids = run_pipeline(
        synthetic_image,
        template_config_path="templates_config.json",
        detection_config={'intensity_threshold': 0.01}
    )
    
    assert match is not None, "Should find a match even with different scale"
    assert len(centroids) > 0, "Should detect stars"
    
    print(f"✓ Scale test passed: Match={match}, Score={score:.4f}, Stars={len(centroids)}")


def test_pipeline_with_noise():
    """Test pipeline with noise added."""
    templates = get_sample_templates()
    test_template = templates["Big Dipper"]
    
    # Generate with noise
    synthetic_image, _ = generate_synthetic_image(
        test_template,
        image_size=(512, 512),
        rotation_angle=0.0,
        scale_factor=10.0,
        translation=(0, 0),
        noise_level=1.0,  # Add noise
        random_seed=42
    )
    
    match, score, centroids = run_pipeline(
        synthetic_image,
        template_config_path="templates_config.json",
        detection_config={'intensity_threshold': 0.01}
    )
    
    assert match is not None, "Should find a match even with noise"
    assert len(centroids) > 0, "Should detect stars"
    
    print(f"✓ Noise test passed: Match={match}, Score={score:.4f}, Stars={len(centroids)}")


def test_pipeline_no_match_threshold():
    """Test pipeline with no-match threshold."""
    templates = get_sample_templates()
    test_template = templates["Big Dipper"]
    
    synthetic_image, _ = generate_synthetic_image(
        test_template,
        image_size=(512, 512),
        rotation_angle=0.0,
        scale_factor=10.0,
        translation=(0, 0),
        noise_level=0.0,
        random_seed=42
    )
    
    # First, get score without threshold
    match1, score1, _ = run_pipeline(
        synthetic_image,
        template_config_path="templates_config.json",
        detection_config={'intensity_threshold': 0.01},
        no_match_threshold=None
    )
    
    # Then with very low threshold (should return no match)
    match2, score2, _ = run_pipeline(
        synthetic_image,
        template_config_path="templates_config.json",
        detection_config={'intensity_threshold': 0.01},
        no_match_threshold=0.01  # Very low threshold
    )
    
    assert match1 is not None, "Should match without threshold"
    assert match2 is None, "Should return no match with low threshold"
    assert score1 == score2, "Scores should be the same"
    
    print(f"✓ No-match threshold test passed: Match1={match1}, Match2={match2}, Score={score1:.4f}")


def run_all_tests():
    """Run all pipeline tests."""
    print("=" * 60)
    print("Testing Constellation Detection Pipeline")
    print("=" * 60)
    
    try:
        test_pipeline_big_dipper()
        test_pipeline_orion()
        test_pipeline_with_rotation()
        test_pipeline_with_scale()
        test_pipeline_with_noise()
        test_pipeline_no_match_threshold()
        
        print("\n" + "=" * 60)
        print("✓ All pipeline tests passed!")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

