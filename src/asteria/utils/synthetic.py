"""
Lightweight synthetic astronomical image generator for testing and demonstration.
Generates small simulated fields (<= 512px) with stars, galaxies, touching pairs, and noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from astropy.io import fits


def make_synthetic_field(
    size: int = 256,
    num_stars: int = 15,
    num_galaxies: int = 10,
    num_blended_pairs: int = 2,
    sky_mean: float = 5.0,
    sky_std: float = 1.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a lightweight synthetic astronomical image with ground truth segmentation labels.

    Parameters
    ----------
    size : int, default 256
        Image dimension in pixels (size x size). Kept <= 512 for lightweight testing.
    num_stars : int, default 15
        Number of isolated stellar point sources.
    num_galaxies : int, default 10
        Number of extended exponential galaxies.
    num_blended_pairs : int, default 2
        Number of overlapping/touching source pairs to evaluate deblending.
    sky_mean : float, default 5.0
        Mean sky background flux in counts/pixel.
    sky_std : float, default 1.2
        Gaussian sky noise standard deviation.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    image : np.ndarray
        2D float32 image array with physical flux and noise.
    segmap : np.ndarray
        2D int32 ground truth segmentation map where 0 is background and k >= 1 is source ID.
    """
    if size > 512:
        raise ValueError(f"Synthetic fields are restricted to <= 512 pixels for lightweight testing, got {size}")

    rng = np.random.RandomState(seed)

    # 1. Base sky background with Gaussian noise
    image = rng.normal(loc=sky_mean, scale=sky_std, size=(size, size)).astype(np.float32)
    segmap = np.zeros((size, size), dtype=np.int32)

    y_grid, x_grid = np.indices((size, size))
    current_id = 1

    # 2. Add Point Source Stars (2D Gaussian PSF)
    for _ in range(num_stars):
        cx = rng.uniform(15, size - 15)
        cy = rng.uniform(15, size - 15)
        peak_flux = rng.uniform(80.0, 1500.0)
        sigma = rng.uniform(1.2, 2.0)

        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        star_profile = peak_flux * np.exp(-0.5 * dist_sq / (sigma ** 2))
        image += star_profile.astype(np.float32)

        # Foreground label threshold (3-sigma over background)
        star_mask = star_profile > (3.0 * sky_std)
        if np.any(star_mask):
            segmap[star_mask] = current_id
            current_id += 1

    # 3. Add Extended Galaxies (Elliptical Exponential Profiles)
    for _ in range(num_galaxies):
        cx = rng.uniform(20, size - 20)
        cy = rng.uniform(20, size - 20)
        peak_flux = rng.uniform(30.0, 400.0)
        r_scale = rng.uniform(3.0, 7.0)
        q = rng.uniform(0.4, 0.9)  # Axis ratio
        theta = rng.uniform(0.0, np.pi)

        # Coordinate rotation
        x_rot = (x_grid - cx) * np.cos(theta) + (y_grid - cy) * np.sin(theta)
        y_rot = -(x_grid - cx) * np.sin(theta) + (y_grid - cy) * np.cos(theta)
        r_ellip = np.sqrt(x_rot ** 2 + (y_rot / q) ** 2)

        galaxy_profile = peak_flux * np.exp(-r_ellip / r_scale)
        image += galaxy_profile.astype(np.float32)

        gal_mask = galaxy_profile > (2.5 * sky_std)
        if np.any(gal_mask):
            segmap[gal_mask] = current_id
            current_id += 1

    # 4. Add Blended / Touching Source Pairs
    for _ in range(num_blended_pairs):
        base_x = rng.uniform(30, size - 30)
        base_y = rng.uniform(30, size - 30)
        separation = rng.uniform(6.0, 10.0)
        angle = rng.uniform(0.0, 2 * np.pi)

        cx1, cy1 = base_x, base_y
        cx2, cy2 = base_x + separation * np.cos(angle), base_y + separation * np.sin(angle)

        for (cx, cy) in [(cx1, cy1), (cx2, cy2)]:
            peak = rng.uniform(100.0, 500.0)
            r_scale = rng.uniform(3.5, 5.0)
            dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            src_profile = peak * np.exp(-dist / r_scale)
            image += src_profile.astype(np.float32)

            src_mask = src_profile > (2.5 * sky_std)
            if np.any(src_mask):
                # Only assign where not already labeled by preceding object to create touching boundary
                segmap[src_mask & (segmap == 0)] = current_id
                current_id += 1

    # 5. Add Poisson photon shot noise on positive fluxes
    pos_image = np.maximum(image, 0.0)
    shot_noise = rng.normal(loc=0.0, scale=np.sqrt(pos_image) * 0.2)
    image += shot_noise.astype(np.float32)

    return image, segmap


def save_synthetic_fits(
    filepath: Union[str, Path],
    size: int = 256,
    seed: int = 42,
) -> Path:
    """Generate and save a small synthetic astronomical FITS file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    image, segmap = make_synthetic_field(size=size, seed=seed)

    primary_hdu = fits.PrimaryHDU(data=image)
    primary_hdu.header["INSTRUME"] = "ASTERIA_SYNTHETIC"
    primary_hdu.header["PIXSCALE"] = 0.11
    primary_hdu.header["ZEROPT"] = 31.4

    seg_hdu = fits.ImageHDU(data=segmap.astype(np.int32), name="SEGMAP")

    hdul = fits.HDUList([primary_hdu, seg_hdu])
    hdul.writeto(str(path), overwrite=True)
    return path
