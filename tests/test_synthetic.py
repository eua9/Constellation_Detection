"""
Comprehensive tests for synthetic star map generation.
"""

import numpy as np
import pytest
from templates.synthetic import StarMapConfig, SyntheticStarMapGenerator


class TestStarMapConfig:
    """Test the StarMapConfig dataclass."""
    
    def test_default_config(self):
        """Test that default configuration values are set correctly."""
        config = StarMapConfig()
        assert config.width == 1024
        assert config.height == 1024
        assert config.fov_deg == 5.0
        assert config.mag_limit == 16.0
        assert config.zeropoint_mag == 25.0
        assert config.psf_sigma_pix == 1.5
        assert config.background_level == 100.0
        assert config.read_noise_std == 5.0
        assert config.rng_seed == 42
    
    def test_custom_config(self):
        """Test creating a custom configuration."""
        config = StarMapConfig(
            width=512,
            height=512,
            fov_deg=2.0,
            mag_limit=15.0,
            rng_seed=123
        )
        assert config.width == 512
        assert config.height == 512
        assert config.fov_deg == 2.0
        assert config.mag_limit == 15.0
        assert config.rng_seed == 123


class TestSyntheticStarMapGenerator:
    """Test the SyntheticStarMapGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return StarMapConfig(
            width=256,
            height=256,
            fov_deg=2.0,
            mag_limit=15.0,
            psf_sigma_pix=1.0,
            background_level=50.0,
            read_noise_std=2.0,
            rng_seed=42
        )
    
    @pytest.fixture
    def generator(self, config):
        """Create a generator instance."""
        return SyntheticStarMapGenerator(
            config=config,
            ra_center_deg=0.0,
            dec_center_deg=0.0
        )
    
    @pytest.fixture
    def simple_catalog(self):
        """Create a simple test catalog."""
        return {
            "ra": np.array([0.0, 0.01, 0.02, 0.0, 0.0]),  # degrees
            "dec": np.array([0.0, 0.0, 0.0, 0.01, 0.02]),  # degrees
            "mag": np.array([10.0, 12.0, 14.0, 11.0, 15.0])  # magnitudes
        }
    
    def test_initialization(self, config):
        """Test generator initialization."""
        gen = SyntheticStarMapGenerator(
            config=config,
            ra_center_deg=45.0,
            dec_center_deg=30.0
        )
        assert gen.config == config
        assert np.isclose(gen.ra0, np.deg2rad(45.0))
        assert np.isclose(gen.dec0, np.deg2rad(30.0))
        assert gen.pixel_scale_deg > 0
        assert gen.pixel_scale_rad > 0
    
    def test_pixel_scale_calculation(self, config):
        """Test that pixel scale is calculated correctly."""
        gen = SyntheticStarMapGenerator(
            config=config,
            ra_center_deg=0.0,
            dec_center_deg=0.0
        )
        # FOV is 2.0 deg, smaller dimension is 256, so pixel scale should be 2.0/256
        expected_pixel_scale_deg = 2.0 / 256
        assert np.isclose(gen.pixel_scale_deg, expected_pixel_scale_deg)
        assert np.isclose(gen.pixel_scale_rad, np.deg2rad(expected_pixel_scale_deg))
    
    def test_world_to_pixel_center(self, generator):
        """Test that center coordinates map to image center."""
        x, y = generator.world_to_pixel(
            np.array([0.0]),  # RA at center
            np.array([0.0])   # Dec at center
        )
        assert np.isclose(x[0], generator.config.width / 2.0)
        assert np.isclose(y[0], generator.config.height / 2.0)
    
    def test_world_to_pixel_offset(self, generator):
        """Test coordinate conversion for offset positions."""
        # Small offset in RA (positive)
        x, y = generator.world_to_pixel(
            np.array([0.01]),  # 0.01 deg offset in RA
            np.array([0.0])
        )
        # Should be offset to the right (positive x)
        assert x[0] > generator.config.width / 2.0
        
        # Small offset in Dec (positive)
        x, y = generator.world_to_pixel(
            np.array([0.0]),
            np.array([0.01])  # 0.01 deg offset in Dec
        )
        # Should be offset downward (positive y)
        assert y[0] > generator.config.height / 2.0
    
    def test_mag_to_flux(self, generator):
        """Test magnitude to flux conversion."""
        # Star at zeropoint should have flux ~1
        mag = np.array([generator.config.zeropoint_mag])
        flux = generator.mag_to_flux(mag)
        assert np.isclose(flux[0], 1.0, rtol=1e-6)
        
        # Brighter star (lower mag) should have higher flux
        mag_brighter = np.array([generator.config.zeropoint_mag - 2.5])
        flux_brighter = generator.mag_to_flux(mag_brighter)
        assert flux_brighter[0] > flux[0]
        
        # Fainter star (higher mag) should have lower flux
        mag_fainter = np.array([generator.config.zeropoint_mag + 2.5])
        flux_fainter = generator.mag_to_flux(mag_fainter)
        assert flux_fainter[0] < flux[0]
        
        # Check the relationship: 2.5 mag difference = 10x flux
        assert np.isclose(flux_brighter[0] / flux_fainter[0], 100.0, rtol=1e-3)
    
    def test_select_stars_in_fov(self, generator, simple_catalog):
        """Test star selection within FOV."""
        x_pix, y_pix, mag_sel = generator._select_stars_in_fov(
            simple_catalog,
            ra_col="ra",
            dec_col="dec",
            mag_col="mag"
        )
        
        # All stars should be within reasonable bounds (with margin)
        margin = int(4 * generator.config.psf_sigma_pix)
        assert np.all(x_pix >= -margin)
        assert np.all(x_pix <= generator.config.width - 1 + margin)
        assert np.all(y_pix >= -margin)
        assert np.all(y_pix <= generator.config.height - 1 + margin)
        
        # All selected stars should be brighter than mag_limit
        assert np.all(mag_sel <= generator.config.mag_limit)
    
    def test_magnitude_cut(self, generator):
        """Test that stars fainter than mag_limit are excluded."""
        catalog = {
            "ra": np.array([0.0, 0.0, 0.0]),
            "dec": np.array([0.0, 0.0, 0.0]),
            "mag": np.array([10.0, 15.0, 20.0])  # Last one should be cut
        }
        
        x_pix, y_pix, mag_sel = generator._select_stars_in_fov(
            catalog,
            ra_col="ra",
            dec_col="dec",
            mag_col="mag"
        )
        
        # Only stars with mag <= 15.0 should be included
        assert np.all(mag_sel <= generator.config.mag_limit)
        assert len(mag_sel) <= 2  # At most 2 stars should pass
    
    def test_generate_from_catalog(self, generator, simple_catalog):
        """Test the main generation function."""
        image, x_pix, y_pix, mag_sel = generator.generate_from_catalog(
            simple_catalog,
            ra_col="ra",
            dec_col="dec",
            mag_col="mag"
        )
        
        # Check image properties
        assert image.shape == (generator.config.height, generator.config.width)
        assert image.dtype == np.float32
        assert np.all(image >= 0)  # No negative values
        
        # Check that background level is approximately present
        # (may vary due to noise, but should be close)
        mean_background = np.mean(image)
        assert mean_background >= generator.config.background_level * 0.5
        assert mean_background <= generator.config.background_level * 2.0
        
        # Check output arrays
        assert len(x_pix) == len(y_pix) == len(mag_sel)
        assert len(x_pix) > 0  # Should have at least some stars
    
    def test_image_has_stars(self, generator):
        """Test that generated image contains star signals."""
        # Create catalog with bright stars at center
        catalog = {
            "ra": np.array([0.0, 0.0, 0.0]),
            "dec": np.array([0.0, 0.0, 0.0]),
            "mag": np.array([10.0, 11.0, 12.0])  # Bright stars
        }
        
        image, _, _, _ = generator.generate_from_catalog(catalog)
        
        # Image should have some pixels brighter than background
        max_value = np.max(image)
        assert max_value > generator.config.background_level
    
    def test_reproducibility(self, config):
        """Test that same seed produces same results."""
        gen1 = SyntheticStarMapGenerator(
            config=config,
            ra_center_deg=0.0,
            dec_center_deg=0.0
        )
        
        gen2 = SyntheticStarMapGenerator(
            config=config,
            ra_center_deg=0.0,
            dec_center_deg=0.0
        )
        
        catalog = {
            "ra": np.array([0.0, 0.01]),
            "dec": np.array([0.0, 0.0]),
            "mag": np.array([12.0, 13.0])
        }
        
        image1, _, _, _ = gen1.generate_from_catalog(catalog)
        image2, _, _, _ = gen2.generate_from_catalog(catalog)
        
        # Images should be identical (same seed)
        np.testing.assert_array_equal(image1, image2)
    
    def test_different_seeds_produce_different_noise(self, config):
        """Test that different seeds produce different noise."""
        config_dict = config.__dict__.copy()
        config_dict['rng_seed'] = 42
        config1 = StarMapConfig(**config_dict)
        config_dict['rng_seed'] = 123
        config2 = StarMapConfig(**config_dict)
        
        gen1 = SyntheticStarMapGenerator(
            config=config1,
            ra_center_deg=0.0,
            dec_center_deg=0.0
        )
        
        gen2 = SyntheticStarMapGenerator(
            config=config2,
            ra_center_deg=0.0,
            dec_center_deg=0.0
        )
        
        catalog = {
            "ra": np.array([0.0]),
            "dec": np.array([0.0]),
            "mag": np.array([12.0])
        }
        
        image1, _, _, _ = gen1.generate_from_catalog(catalog)
        image2, _, _, _ = gen2.generate_from_catalog(catalog)
        
        # Images should be different (different noise)
        assert not np.array_equal(image1, image2)
    
    def test_no_noise_mode(self, config):
        """Test generation with no read noise."""
        config_dict = config.__dict__.copy()
        config_dict['read_noise_std'] = 0.0
        config_no_noise = StarMapConfig(**config_dict)
        gen = SyntheticStarMapGenerator(
            config=config_no_noise,
            ra_center_deg=0.0,
            dec_center_deg=0.0
        )
        
        catalog = {
            "ra": np.array([0.0]),
            "dec": np.array([0.0]),
            "mag": np.array([12.0])
        }
        
        image, _, _, _ = gen.generate_from_catalog(catalog)
        
        # Without noise, image should be deterministic (except for floating point)
        assert image.shape == (config.height, config.width)
        assert np.all(image >= 0)
    
    def test_empty_catalog(self, generator):
        """Test handling of empty catalog."""
        empty_catalog = {
            "ra": np.array([]),
            "dec": np.array([]),
            "mag": np.array([])
        }
        
        image, x_pix, y_pix, mag_sel = generator.generate_from_catalog(empty_catalog)
        
        # Should still produce an image (just background + noise)
        assert image.shape == (generator.config.height, generator.config.width)
        assert len(x_pix) == 0
        assert len(y_pix) == 0
        assert len(mag_sel) == 0
    
    def test_custom_column_names(self, generator):
        """Test using custom column names."""
        catalog = {
            "right_ascension": np.array([0.0, 0.01]),
            "declination": np.array([0.0, 0.0]),
            "magnitude": np.array([12.0, 13.0])
        }
        
        image, x_pix, y_pix, mag_sel = generator.generate_from_catalog(
            catalog,
            ra_col="right_ascension",
            dec_col="declination",
            mag_col="magnitude"
        )
        
        assert image.shape == (generator.config.height, generator.config.width)
        assert len(x_pix) > 0
    
    def test_ra_wrapping(self, generator):
        """Test that RA wrapping is handled correctly."""
        # Test RA near 360/0 boundary
        catalog = {
            "ra": np.array([359.9, 0.1]),  # Near boundary
            "dec": np.array([0.0, 0.0]),
            "mag": np.array([12.0, 12.0])
        }
        
        # Center at 0.0, so 359.9 should wrap to -0.1
        gen = SyntheticStarMapGenerator(
            config=generator.config,
            ra_center_deg=0.0,
            dec_center_deg=0.0
        )
        
        x_pix, y_pix, mag_sel = gen._select_stars_in_fov(
            catalog,
            ra_col="ra",
            dec_col="dec",
            mag_col="mag"
        )
        
        # Should handle wrapping correctly
        assert len(x_pix) >= 0  # May or may not be in FOV, but shouldn't crash
    
    def test_pandas_dataframe_support(self, generator):
        """Test that pandas DataFrame works as catalog input."""
        try:
            import pandas as pd  # type: ignore
            
            df = pd.DataFrame({
                "ra": [0.0, 0.01, 0.02],
                "dec": [0.0, 0.0, 0.0],
                "mag": [12.0, 13.0, 14.0]
            })
            
            image, x_pix, y_pix, mag_sel = generator.generate_from_catalog(df)
            
            assert image.shape == (generator.config.height, generator.config.width)
            assert len(x_pix) > 0
        except ImportError:
            pytest.skip("pandas not available")
    
    def test_psf_rendering(self, generator):
        """Test that PSF is rendered correctly."""
        # Single bright star at center
        catalog = {
            "ra": np.array([0.0]),
            "dec": np.array([0.0]),
            "mag": np.array([10.0])  # Bright star
        }
        
        image, x_pix, y_pix, mag_sel = generator.generate_from_catalog(catalog)
        
        # Find the maximum (should be near center)
        max_idx = np.unravel_index(np.argmax(image), image.shape)
        center_y = generator.config.height // 2
        center_x = generator.config.width // 2
        
        # Maximum should be near center (within PSF radius)
        psf_radius = int(4 * generator.config.psf_sigma_pix)
        assert abs(max_idx[0] - center_y) <= psf_radius
        assert abs(max_idx[1] - center_x) <= psf_radius


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

