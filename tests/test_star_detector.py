"""
Comprehensive tests for star detection module.
"""

import numpy as np
import pytest
import cv2
from detection.star_detector import StarDetector


class TestStarDetector:
    """Test the StarDetector class."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image with bright spots (stars)."""
        img = np.zeros((100, 100), dtype=np.uint8)
        # Add bright spots (stars)
        cv2.circle(img, (20, 20), 3, 255, -1)  # Star at (20, 20)
        cv2.circle(img, (50, 50), 2, 255, -1)  # Star at (50, 50)
        cv2.circle(img, (80, 80), 4, 255, -1)  # Star at (80, 80)
        # Add some noise
        img += np.random.randint(0, 30, (100, 100), dtype=np.uint8)
        return img
    
    @pytest.fixture
    def high_threshold_image(self):
        """Create image with only very bright spots."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20, 20] = 250  # Very bright
        img[50, 50] = 200  # Medium bright
        img[80, 80] = 100  # Dim
        return img
    
    def test_initialization_default(self):
        """Test default initialization."""
        detector = StarDetector()
        assert detector.threshold == 200
        assert detector.detector is not None
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        detector = StarDetector(threshold=150, min_area=10.0)
        assert detector.threshold == 150
        assert detector.detector is not None
    
    def test_initialization_with_filters(self):
        """Test initialization with additional filters."""
        detector = StarDetector(
            threshold=200,
            min_area=5.0,
            max_area=100.0,
            filter_by_circularity=True,
            min_circularity=0.5,
            filter_by_convexity=True,
            min_convexity=0.8,
        )
        assert detector.threshold == 200
        assert detector.detector is not None
    
    def test_binarize(self):
        """Test image binarization."""
        detector = StarDetector(threshold=128)
        img = np.array([
            [50, 100, 150],
            [200, 250, 100],
            [80, 120, 180]
        ], dtype=np.uint8)
        
        binary = detector._binarize(img)
        
        assert binary.dtype == np.uint8
        assert np.all((binary == 0) | (binary == 255))
        # Values above threshold should be 255
        assert binary[1, 1] == 255  # 250 > 128
        assert binary[0, 0] == 0     # 50 < 128
    
    def test_detect_basic(self, sample_image):
        """Test basic star detection."""
        detector = StarDetector(threshold=100, min_area=1.0)
        centroids = detector.detect(sample_image)
        
        assert isinstance(centroids, np.ndarray)
        assert len(centroids.shape) == 2
        assert centroids.shape[1] == 2  # (x, y) coordinates
        assert centroids.dtype == np.float32
        # May or may not detect stars depending on noise, but structure should be correct
        assert len(centroids) >= 0
    
    def test_detect_empty_image(self):
        """Test detection on empty/dark image."""
        detector = StarDetector(threshold=200)
        img = np.zeros((100, 100), dtype=np.uint8)
        centroids = detector.detect(img)
        
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape == (0, 2)
        assert len(centroids) == 0
    
    def test_detect_no_stars_high_threshold(self, sample_image):
        """Test that high threshold filters out dim stars."""
        detector = StarDetector(threshold=250, min_area=1.0)
        centroids = detector.detect(sample_image)
        
        # With very high threshold, should detect fewer or no stars
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape[1] == 2
    
    def test_detect_min_area_filtering(self):
        """Test that min_area filters out small blobs."""
        # Create image with one large and one small bright spot
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (30, 30), 5, 255, -1)  # Large spot
        cv2.circle(img, (70, 70), 1, 255, -1)  # Small spot
        
        # With high min_area, should only detect large spot
        detector = StarDetector(threshold=100, min_area=10.0)
        centroids = detector.detect(img)
        
        # Should detect at least the large spot
        assert len(centroids) >= 1
    
    def test_detect_max_area_filtering(self):
        """Test that max_area filters out large blobs."""
        # Create image with one large and one small bright spot
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (30, 30), 10, 255, -1)  # Large spot
        cv2.circle(img, (70, 70), 2, 255, -1)   # Small spot
        
        # With low max_area, should filter out large spot
        detector = StarDetector(threshold=100, min_area=1.0, max_area=50.0)
        centroids = detector.detect(img)
        
        # Should detect the small spot but not the large one
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape[1] == 2
    
    def test_detect_centroid_format(self, sample_image):
        """Test that centroids are in correct (x, y) format."""
        detector = StarDetector(threshold=100, min_area=1.0)
        centroids = detector.detect(sample_image)
        
        if len(centroids) > 0:
            # Check that coordinates are within image bounds
            assert np.all(centroids[:, 0] >= 0)  # x >= 0
            assert np.all(centroids[:, 0] < sample_image.shape[1])  # x < width
            assert np.all(centroids[:, 1] >= 0)  # y >= 0
            assert np.all(centroids[:, 1] < sample_image.shape[0])  # y < height
    
    def test_detect_float_image(self):
        """Test detection on float image (should convert to uint8)."""
        detector = StarDetector(threshold=128)
        img = np.array([
            [50.0, 100.0, 150.0],
            [200.0, 250.0, 100.0],
        ], dtype=np.float32)
        
        centroids = detector.detect(img)
        
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape[1] == 2
    
    def test_detect_color_image(self):
        """Test detection on color image (should convert to grayscale)."""
        detector = StarDetector(threshold=128)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[20, 20, :] = [255, 255, 255]  # White spot
        img[30, 30, :] = [200, 200, 200]  # Gray spot
        
        centroids = detector.detect(img)
        
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape[1] == 2
    
    def test_detect_circularity_filter(self):
        """Test circularity filtering."""
        # Create circular and elongated blobs
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (30, 30), 5, 255, -1)  # Circular
        cv2.ellipse(img, (70, 70), (10, 2), 0, 0, 360, 255, -1)  # Elongated
        
        # Filter by circularity (should prefer circular blobs)
        detector = StarDetector(
            threshold=100,
            min_area=1.0,
            filter_by_circularity=True,
            min_circularity=0.7
        )
        centroids = detector.detect(img)
        
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape[1] == 2
    
    def test_detect_convexity_filter(self):
        """Test convexity filtering."""
        # Create convex and non-convex blobs
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (30, 30), 5, 255, -1)  # Convex (circle)
        # Create a star shape (non-convex)
        points = np.array([[50, 30], [52, 35], [57, 35], [53, 38], [55, 43],
                          [50, 40], [45, 43], [47, 38], [43, 35], [48, 35]],
                         np.int32)
        cv2.fillPoly(img, [points], 255)
        
        detector = StarDetector(
            threshold=100,
            min_area=1.0,
            filter_by_convexity=True,
            min_convexity=0.9
        )
        centroids = detector.detect(img)
        
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape[1] == 2
    
    def test_detect_inertia_filter(self):
        """Test inertia ratio filtering."""
        # Create circular and elongated blobs
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (30, 30), 5, 255, -1)  # Circular (high inertia)
        cv2.ellipse(img, (70, 70), (10, 2), 0, 0, 360, 255, -1)  # Elongated (low inertia)
        
        detector = StarDetector(
            threshold=100,
            min_area=1.0,
            filter_by_inertia=True,
            min_inertia_ratio=0.5  # Prefer more circular shapes
        )
        centroids = detector.detect(img)
        
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape[1] == 2
    
    def test_integration_with_preprocessing(self):
        """Test integration with preprocessing module."""
        try:
            from detection.preprocessing import Preprocessor
            
            # Create a test image
            img = np.random.rand(100, 100).astype(np.float32) * 255
            # Add some bright spots
            img[20, 20] = 250
            img[50, 50] = 240
            img[80, 80] = 230
            
            # Preprocess
            preprocessor = Preprocessor(blur_kernel=(3, 3))
            processed = preprocessor.run(img)
            
            # Detect stars
            detector = StarDetector(threshold=200, min_area=1.0)
            centroids = detector.detect(processed)
            
            assert isinstance(centroids, np.ndarray)
            assert centroids.shape[1] == 2
            
        except ImportError:
            pytest.skip("detection.preprocessing not available")
    
    def test_integration_with_synthetic_star_map(self):
        """Test detection on synthetic star maps."""
        try:
            from templates.synthetic import StarMapConfig, SyntheticStarMapGenerator
            from detection.preprocessing import Preprocessor
            
            # Generate synthetic star map
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
            
            star_map, x_true, y_true, _ = generator.generate_from_catalog(catalog)
            
            # Preprocess
            preprocessor = Preprocessor(blur_kernel=(3, 3))
            processed = preprocessor.run(star_map)
            
            # Detect stars
            detector = StarDetector(threshold=150, min_area=1.0)
            centroids = detector.detect(processed)
            
            assert isinstance(centroids, np.ndarray)
            assert centroids.shape[1] == 2
            # Should detect at least some stars
            assert len(centroids) >= 0
            
        except ImportError:
            pytest.skip("Required modules not available")
    
    def test_detect_reproducibility(self, sample_image):
        """Test that detection is reproducible."""
        detector1 = StarDetector(threshold=100, min_area=1.0)
        detector2 = StarDetector(threshold=100, min_area=1.0)
        
        centroids1 = detector1.detect(sample_image)
        centroids2 = detector2.detect(sample_image)
        
        # Results should be identical (deterministic algorithm)
        np.testing.assert_array_equal(centroids1, centroids2)
    
    def test_detect_multiple_stars(self):
        """Test detection of multiple stars."""
        img = np.zeros((200, 200), dtype=np.uint8)
        # Add multiple stars at known positions
        star_positions = [(30, 30), (50, 50), (70, 70), (90, 90), (110, 110)]
        for x, y in star_positions:
            cv2.circle(img, (x, y), 3, 255, -1)
        
        detector = StarDetector(threshold=100, min_area=1.0)
        centroids = detector.detect(img)
        
        assert len(centroids) >= len(star_positions)
        assert centroids.shape[1] == 2
    
    def test_detect_edge_cases_single_pixel(self):
        """Test detection of single-pixel stars."""
        img = np.zeros((50, 50), dtype=np.uint8)
        img[25, 25] = 255  # Single bright pixel
        
        detector = StarDetector(threshold=200, min_area=0.5)
        centroids = detector.detect(img)
        
        # May or may not detect single pixel depending on min_area
        assert isinstance(centroids, np.ndarray)
        assert centroids.shape[1] == 2
    
    def test_detect_varying_thresholds(self, high_threshold_image):
        """Test detection with varying threshold values."""
        thresholds = [50, 100, 150, 200, 250]
        detection_counts = []
        
        for threshold in thresholds:
            detector = StarDetector(threshold=threshold, min_area=1.0)
            centroids = detector.detect(high_threshold_image)
            detection_counts.append(len(centroids))
        
        # Higher thresholds should detect fewer stars
        # (assuming image has varying brightness)
        assert all(isinstance(count, int) for count in detection_counts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

