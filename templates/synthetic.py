"""
Synthetic star map generation from catalogs.

Assumptions:
- Catalog has RA, Dec in degrees, and a magnitude column.
- We work in a small field-of-view where small-angle approximations are OK.
- Each star is drawn as a 2D Gaussian PSF onto a pixel grid.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class StarMapConfig:
    """Configuration for synthetic star map generation."""
    
    # Image geometry
    width: int = 1024                  # pixels
    height: int = 1024                 # pixels
    fov_deg: float = 5.0               # field of view (short side) in degrees
    
    # Photometric + PSF
    mag_limit: float = 16.0            # ignore stars fainter than this
    zeropoint_mag: float = 25.0        # magnitude that gives 1 ADU (arbitrary)
    psf_sigma_pix: float = 1.5         # Gaussian sigma of PSF in pixels
    
    # Noise model
    background_level: float = 100.0    # constant background level (ADU)
    read_noise_std: float = 5.0        # Gaussian noise sigma (ADU)
    rng_seed: Optional[int] = 42       # for reproducibility


class SyntheticStarMapGenerator:
    """
    Generate synthetic CCD-like star images from a star catalog.
    
    Coordinate system:
    - x: 0 .. width-1  (columns)
    - y: 0 .. height-1 (rows)
    - image center corresponds to (ra_center_deg, dec_center_deg).
    """
    
    def __init__(
        self,
        config: StarMapConfig,
        ra_center_deg: float,
        dec_center_deg: float,
    ):
        self.config = config
        self.ra0 = np.deg2rad(ra_center_deg)
        self.dec0 = np.deg2rad(dec_center_deg)
        
        # Pixel scale in degrees/pixel based on FOV on the *smaller* image dimension
        self.pixel_scale_deg = config.fov_deg / min(config.width, config.height)
        self.pixel_scale_rad = np.deg2rad(self.pixel_scale_deg)
        
        self.rng = np.random.default_rng(config.rng_seed)
    
    # ---------- Public high-level API ----------
    
    def generate_from_catalog(
        self,
        catalog,
        ra_col: str = "ra",
        dec_col: str = "dec",
        mag_col: str = "mag",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Main entry point.
        
        Parameters
        ----------
        catalog : pandas.DataFrame or dict-like
            Must support catalog[ra_col], catalog[dec_col], catalog[mag_col].
        ra_col, dec_col, mag_col : str
            Column names for RA [deg], Dec [deg], and magnitude.
        
        Returns
        -------
        image : np.ndarray, shape (height, width)
            The synthetic star map image.
        x_pix, y_pix : np.ndarray
            Pixel coordinates of *rendered* stars (after FOV and mag cuts).
        mag_sel : np.ndarray
            Magnitudes of the rendered stars.
        
        Notes
        -----
        The (x_pix, y_pix) and mag_sel outputs are useful for evaluating your
        constellation detection algorithm (ground truth positions).
        """
        x_pix, y_pix, mag_sel = self._select_stars_in_fov(
            catalog, ra_col=ra_col, dec_col=dec_col, mag_col=mag_col
        )
        image = self._render_stars(x_pix, y_pix, mag_sel)
        return image, x_pix, y_pix, mag_sel
    
    # ---------- Internal steps ----------
    
    def _select_stars_in_fov(
        self, catalog, ra_col: str, dec_col: str, mag_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select stars within the FOV and brighter than mag_limit."""
        # Extract arrays (works for pandas DataFrame, astropy Table, etc.)
        ra_deg = np.asarray(catalog[ra_col], dtype=float)
        dec_deg = np.asarray(catalog[dec_col], dtype=float)
        mag = np.asarray(catalog[mag_col], dtype=float)
        
        # Magnitude cut
        mask_mag = mag <= self.config.mag_limit
        
        # Project RA/Dec to pixel coordinates
        x_pix, y_pix = self.world_to_pixel(ra_deg, dec_deg)
        
        # Some stars near edges still contribute via PSF, so include a small margin
        margin = int(4 * self.config.psf_sigma_pix)
        mask_spatial = (
            (x_pix >= -margin)
            & (x_pix <= self.config.width - 1 + margin)
            & (y_pix >= -margin)
            & (y_pix <= self.config.height - 1 + margin)
        )
        
        mask = mask_mag & mask_spatial
        return x_pix[mask], y_pix[mask], mag[mask]
    
    def _render_stars(
        self,
        x_pix: np.ndarray,
        y_pix: np.ndarray,
        mag: np.ndarray,
    ) -> np.ndarray:
        """Render stars onto an image using a Gaussian PSF model."""
        cfg = self.config
        
        # Start with constant background
        image = np.full(
            (cfg.height, cfg.width), cfg.background_level, dtype=np.float32
        )
        
        # Precompute PSF kernel
        radius = int(np.ceil(4 * cfg.psf_sigma_pix))  # 4-sigma radius is enough
        yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        r2 = xx**2 + yy**2
        psf = np.exp(-0.5 * r2 / (cfg.psf_sigma_pix**2))
        psf /= psf.sum()  # normalize to 1 total flux
        
        # Convert magnitudes to total fluxes
        fluxes = self.mag_to_flux(mag)
        
        # Draw each star
        for xi, yi, fi in zip(x_pix, y_pix, fluxes):
            ix = int(np.round(xi))
            iy = int(np.round(yi))
            
            # Bounding box in image coordinates
            x0 = ix - radius
            x1 = ix + radius + 1
            y0 = iy - radius
            y1 = iy + radius + 1
            
            # Skip if completely outside
            if x1 <= 0 or y1 <= 0 or x0 >= cfg.width or y0 >= cfg.height:
                continue
            
            # Corresponding PSF indices
            kx0 = 0
            ky0 = 0
            kx1 = psf.shape[1]
            ky1 = psf.shape[0]
            
            # Clip to image, adjust kernel slice accordingly
            if x0 < 0:
                kx0 = -x0
                x0 = 0
            if y0 < 0:
                ky0 = -y0
                y0 = 0
            if x1 > cfg.width:
                kx1 -= (x1 - cfg.width)
                x1 = cfg.width
            if y1 > cfg.height:
                ky1 -= (y1 - cfg.height)
                y1 = cfg.height
            
            # Add scaled PSF to image
            image[y0:y1, x0:x1] += fi * psf[ky0:ky1, kx0:kx1]
        
        # Add Gaussian read noise
        if cfg.read_noise_std > 0:
            noise = self.rng.normal(
                loc=0.0, scale=cfg.read_noise_std, size=image.shape
            )
            image += noise
        
        # Ensure no negatives
        np.maximum(image, 0.0, out=image)
        return image
    
    # ---------- Utility methods ----------
    
    def world_to_pixel(
        self, ra_deg: np.ndarray, dec_deg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert RA/Dec [deg] to pixel coordinates using small-angle projection.
        
        Returns
        -------
        x, y : np.ndarray
            Pixel coordinates (floats). (0,0) is top-left corner, center of image is
            (width/2, height/2).
        """
        ra = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)
        
        # RA difference, wrapped to [-pi, pi]
        dra = ra - self.ra0
        dra = (dra + np.pi) % (2.0 * np.pi) - np.pi
        
        ddec = dec - self.dec0
        
        # Small-angle tangent-plane approximation
        # x ~ ΔRA * cos(dec0), y ~ ΔDec
        x_rad = dra * np.cos(self.dec0)
        y_rad = ddec
        
        x_pix = (self.config.width / 2.0) + x_rad / self.pixel_scale_rad
        y_pix = (self.config.height / 2.0) + y_rad / self.pixel_scale_rad
        
        return x_pix, y_pix
    
    def mag_to_flux(self, mag: np.ndarray) -> np.ndarray:
        """
        Convert magnitudes to arbitrary flux units.
        
        We assume:
            flux = 10^(-0.4 * (mag - zeropoint_mag))
        
        So a star at mag = zeropoint_mag has flux ~1.
        You can later scale this to match whatever units your pipeline expects.
        """
        return 10.0 ** (-0.4 * (mag - self.config.zeropoint_mag))

