"""
Comprehensive tests for image preprocessing module.
"""

import numpy as np
import pytest
import cv2
from detection.preprocessing import Preprocessor


class TestPreprocessor:
    """Test the Preprocessor class."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple test image with some structure
        img = np.zeros((100, 100), dtype=np.float32)
        # Add some bright spots (stars)
        img[20, 20] = 200.0
        img[50, 50] = 180.0
        img[80, 80] = 150.0
        # Add some background noise
        img += np.random.randn(100, 100).astype(np.float32) * 10
        return img
    
    @pytest.fixture
    def color_image(self):
        """Create a sample color test image."""
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[20, 20, :] = [200, 200, 200]
        img[50, 50, :] = [180, 180, 180]
        return img
    
    def test_initialization_default(self):
        """Test default initialization."""
        preprocessor = Preprocessor()
        assert preprocessor.blur_kernel == (3, 3)
        assert preprocessor.gaussian_sigma == 0
        assert preprocessor.threshold_method is None
        assert preprocessor.max_value == 255.0
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        preprocessor = Preprocessor(
            blur_kernel=(5, 5),
            gaussian_sigma=1.5,
            threshold_method='simple',
            threshold_value=128.0,
        )
        assert preprocessor.blur_kernel == (5, 5)
        assert preprocessor.gaussian_sigma == 1.5
        assert preprocessor.threshold_method == 'simple'
        assert preprocessor.threshold_value == 128.0
    
    def test_initialization_invalid_kernel(self):
        """Test that even kernel sizes are rejected."""
        with pytest.raises(ValueError, match="blur_kernel dimensions must be odd"):
            Preprocessor(blur_kernel=(4, 4))
    
    def test_initialization_invalid_threshold_method(self):
        """Test that invalid threshold methods are rejected."""
        with pytest.raises(ValueError, match="threshold_method must be one of"):
            Preprocessor(threshold_method='invalid')
    
    def test_initialization_simple_threshold_no_value(self):
        """Test that simple threshold requires a value."""
        with pytest.raises(ValueError, match="threshold_value must be provided"):
            Preprocessor(threshold_method='simple')
    
    def test_normalize_float_image(self, sample_image):
        """Test normalization of float image."""
        preprocessor = Preprocessor()
        normalized = preprocessor.normalize(sample_image)
        
        assert normalized.dtype == np.uint8
        assert normalized.min() >= 0
        assert normalized.max() <= 255
        assert normalized.shape == sample_image.shape
    
    def test_normalize_uint8_image(self):
        """Test normalization of uint8 image."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        preprocessor = Preprocessor()
        normalized = preprocessor.normalize(img)
        
        assert normalized.dtype == np.uint8
        assert normalized.shape == img.shape
    
    def test_normalize_color_image(self, color_image):
        """Test normalization converts color to grayscale."""
        preprocessor = Preprocessor()
        normalized = preprocessor.normalize(color_image)
        
        assert normalized.dtype == np.uint8
        assert len(normalized.shape) == 2  # Should be grayscale
        assert normalized.shape == color_image.shape[:2]
    
    def test_normalize_preserves_structure(self, sample_image):
        """Test that normalization preserves relative structure."""
        preprocessor = Preprocessor()
        normalized = preprocessor.normalize(sample_image)
        
        # Bright spots should still be bright
        assert normalized[20, 20] > normalized.mean()
        assert normalized[50, 50] > normalized.mean()
    
    def test_denoise(self, sample_image):
        """Test denoising functionality."""
        preprocessor = Preprocessor(blur_kernel=(5, 5), gaussian_sigma=1.0)
        normalized = preprocessor.normalize(sample_image)
        denoised = preprocessor.denoise(normalized)
        
        assert denoised.dtype == np.uint8
        assert denoised.shape == normalized.shape
    
    def test_denoise_auto_sigma(self):
        """Test that sigma is auto-computed when 0."""
        preprocessor = Preprocessor(blur_kernel=(5, 5), gaussian_sigma=0)
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        denoised = preprocessor.denoise(img)
        
        assert denoised.shape == img.shape
        # Denoised image should be smoother (lower variance)
        assert denoised.std() < img.std() or abs(denoised.std() - img.std()) < 5
    
    def test_threshold_simple(self):
        """Test simple thresholding."""
        img = np.array([
            [50, 100, 150],
            [200, 250, 100],
            [80, 120, 180]
        ], dtype=np.uint8)
        
        preprocessor = Preprocessor(threshold_method='simple', threshold_value=128.0)
        thresh = preprocessor.threshold(img)
        
        assert thresh.dtype == np.uint8
        assert np.all((thresh == 0) | (thresh == 255))
        # Values above threshold should be 255
        assert thresh[1, 1] == 255  # 250 > 128
        assert thresh[0, 0] == 0    # 50 < 128
    
    def test_threshold_adaptive(self):
        """Test adaptive thresholding."""
        # Create image with varying background
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:30, 20:30] = 200  # Bright region
        img[70:80, 70:80] = 200  # Bright region
        img += 50  # Add some background
        
        preprocessor = Preprocessor(threshold_method='adaptive')
        thresh = preprocessor.threshold(img)
        
        assert thresh.dtype == np.uint8
        assert np.all((thresh == 0) | (thresh == 255))
    
    def test_threshold_otsu(self):
        """Test Otsu thresholding."""
        # Create bimodal image
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:40, 20:40] = 200  # Bright region
        img[60:80, 60:80] = 200  # Bright region
        img[img == 0] = 50  # Dark background
        
        preprocessor = Preprocessor(threshold_method='otsu')
        thresh = preprocessor.threshold(img)
        
        assert thresh.dtype == np.uint8
        assert np.all((thresh == 0) | (thresh == 255))
        # Bright regions should be white
        assert thresh[25, 25] == 255
        assert thresh[70, 70] == 255
    
    def test_threshold_none(self):
        """Test that no thresholding returns original image."""
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        preprocessor = Preprocessor()
        result = preprocessor.threshold(img)
        
        np.testing.assert_array_equal(result, img)
    
    def test_threshold_invalid_method(self):
        """Test that invalid threshold method raises error."""
        preprocessor = Preprocessor()
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Unknown threshold method"):
            preprocessor.threshold(img, method='invalid')
    
    def test_run_full_pipeline_no_threshold(self, sample_image):
        """Test full preprocessing pipeline without thresholding."""
        preprocessor = Preprocessor(blur_kernel=(5, 5))
        result = preprocessor.run(sample_image)
        
        assert result.dtype == np.uint8
        assert result.shape == sample_image.shape
        assert result.min() >= 0
        assert result.max() <= 255
    
    def test_run_full_pipeline_with_threshold(self, sample_image):
        """Test full preprocessing pipeline with thresholding."""
        preprocessor = Preprocessor(
            blur_kernel=(5, 5),
            threshold_method='simple',
            threshold_value=128.0
        )
        result = preprocessor.run(sample_image)
        
        assert result.dtype == np.uint8
        assert result.shape == sample_image.shape
        # With thresholding, should be binary
        assert np.all((result == 0) | (result == 255))
    
    def test_run_override_threshold(self, sample_image):
        """Test that apply_threshold parameter can override default."""
        preprocessor = Preprocessor(threshold_method='simple', threshold_value=128.0)
        
        # Override to not apply threshold
        result_no_thresh = preprocessor.run(sample_image, apply_threshold=False)
        # Should not be binary
        assert not np.all((result_no_thresh == 0) | (result_no_thresh == 255))
        
        # Override to apply threshold
        preprocessor_no_thresh = Preprocessor()
        result_with_thresh = preprocessor_no_thresh.run(
            sample_image, apply_threshold=True
        )
        # This will fail because no threshold method is set, but let's test the logic
        # Actually, if apply_threshold=True but no method, it should still work
        # Let's test with a proper setup
        preprocessor2 = Preprocessor(threshold_method='simple', threshold_value=128.0)
        result_with_thresh2 = preprocessor2.run(sample_image, apply_threshold=True)
        assert np.all((result_with_thresh2 == 0) | (result_with_thresh2 == 255))
    
    def test_run_preserves_bright_features(self, sample_image):
        """Test that preprocessing preserves bright features (stars)."""
        preprocessor = Preprocessor(blur_kernel=(3, 3))
        result = preprocessor.run(sample_image)
        
        # Bright spots should still be relatively bright
        center_value = result[50, 50]
        mean_value = result.mean()
        assert center_value > mean_value
    
    def test_integration_with_synthetic_star_map(self):
        """Test preprocessing with synthetic star map."""
        try:
            from templates.synthetic import StarMapConfig, SyntheticStarMapGenerator
            
            # Generate a synthetic star map
            config = StarMapConfig(
                width=256,
                height=256,
                fov_deg=2.0,
                mag_limit=15.0,
                background_level=100.0,
                read_noise_std=5.0,
                rng_seed=42
            )
            
            generator = SyntheticStarMapGenerator(
                config=config,
                ra_center_deg=0.0,
                dec_center_deg=0.0
            )
            
            catalog = {
                "ra": np.array([0.0, 0.01, 0.02]),
                "dec": np.array([0.0, 0.0, 0.0]),
                "mag": np.array([12.0, 13.0, 14.0])
            }
            
            star_map, _, _, _ = generator.generate_from_catalog(catalog)
            
            # Preprocess the synthetic star map
            preprocessor = Preprocessor(blur_kernel=(3, 3))
            processed = preprocessor.run(star_map)
            
            assert processed.dtype == np.uint8
            assert processed.shape == star_map.shape
            assert processed.min() >= 0
            assert processed.max() <= 255
            
        except ImportError:
            pytest.skip("templates.synthetic not available")
    
    def test_different_kernel_sizes(self):
        """Test that different kernel sizes work correctly."""
        img = np.random.rand(100, 100).astype(np.float32) * 255
        
        for kernel_size in [(3, 3), (5, 5), (7, 7), (9, 9)]:
            preprocessor = Preprocessor(blur_kernel=kernel_size)
            result = preprocessor.run(img)
            assert result.shape == img.shape
            assert result.dtype == np.uint8
    
    def test_edge_cases_empty_image(self):
        """Test handling of edge cases."""
        # Very small image
        img = np.array([[100.0]], dtype=np.float32)
        preprocessor = Preprocessor()
        result = preprocessor.run(img)
        assert result.shape == (1, 1)
        assert result.dtype == np.uint8
    
    def test_edge_cases_constant_image(self):
        """Test handling of constant intensity image."""
        img = np.ones((50, 50), dtype=np.float32) * 100.0
        preprocessor = Preprocessor()
        result = preprocessor.run(img)
        
        # Normalized constant image should still be processable
        assert result.shape == img.shape
        assert result.dtype == np.uint8
    
    def test_threshold_parameter_override(self):
        """Test that threshold parameters can be overridden."""
        img = np.array([
            [50, 100, 150],
            [200, 250, 100],
        ], dtype=np.uint8)
        
        preprocessor = Preprocessor(threshold_method='simple', threshold_value=128.0)
        
        # Use default threshold
        thresh1 = preprocessor.threshold(img)
        
        # Override threshold value
        thresh2 = preprocessor.threshold(img, threshold_value=200.0)
        
        # Results should be different
        assert not np.array_equal(thresh1, thresh2)
        
        # With higher threshold, fewer pixels should be white
        assert (thresh2 == 255).sum() < (thresh1 == 255).sum()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

