"""
Astronomical surface brightness photometric transformations.
Features the Leaky Surface Brightness (leakySB) non-linear stretch.
"""

from __future__ import annotations

from typing import Union
import numpy as np


def leaky_surface_brightness(
    data: np.ndarray,
    zero_points: float = 31.4,
    pixel_scale: float = 0.11,
    mmin: float = 13.0,
    mmax: float = 32.0,
    leak_slope: float = 0.05,
) -> np.ndarray:
    """
    Apply Leaky Surface Brightness (leakySB) non-linear transformation.

    Preserves dynamic range across bright stellar cores while lifting faint
    diffuse galaxy wings, with a linear leak below sky background to preserve
    statistical noise characteristics without hard clipping.

    Parameters
    ----------
    data : np.ndarray
        Raw 2D astronomical image array in physical flux units (e.g. counts/sec or electrons).
    zero_points : float, default 31.4
        Photometric zero point magnitude (Roman Space Telescope WFI default: 31.4).
    pixel_scale : float, default 0.11
        Detector plate scale in arcseconds per pixel (Roman WFI default: 0.11).
    mmin : float, default 13.0
        Bright-end surface brightness magnitude cutoff (mag / arcsec^2).
    mmax : float, default 32.0
        Faint-end surface brightness magnitude cutoff (mag / arcsec^2).
    leak_slope : float, default 0.05
        Slope of the linear response for negative/sub-background flux values.

    Returns
    -------
    np.ndarray
        Normalized surface brightness array in range [0.0, 1.0], float32.
    """
    arr = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pixel_area = pixel_scale * pixel_scale

    # Compute positive surface brightness: mu = ZP - 2.5 * log10(flux / area)
    pos_mask = arr > 1e-12
    sb_mag = np.full_like(arr, mmax + 5.0)

    if np.any(pos_mask):
        flux_per_sq_arcsec = arr[pos_mask] / pixel_area
        sb_mag[pos_mask] = zero_points - 2.5 * np.log10(np.maximum(flux_per_sq_arcsec, 1e-15))

    # Normalize between mmin (bright -> 1.0) and mmax (faint -> 0.0)
    mag_range = mmax - mmin
    norm = (mmax - sb_mag) / (mag_range + 1e-9)

    # Apply linear leak for background/negative pixels to prevent hard zeros
    neg_mask = arr <= 1e-12
    if np.any(neg_mask):
        sky_std = float(np.std(arr[neg_mask])) if np.sum(neg_mask) > 10 else 1.0
        if sky_std > 0:
            leak = leak_slope * (arr[neg_mask] / (3.0 * sky_std))
            norm[neg_mask] = np.clip(leak, -0.1, 0.1)

    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def asinh_stretch(data: np.ndarray, non_linear_factor: float = 0.005) -> np.ndarray:
    """Asinh non-linear stretch (Lupton et al. 2004)."""
    arr = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    p_low, p_high = np.percentile(arr, (1.0, 99.8))
    clipped = np.clip(arr, p_low, p_high) - p_low
    max_val = np.max(clipped) + 1e-9
    norm = clipped / max_val
    stretched = np.arcsinh(norm / non_linear_factor) / np.arcsinh(1.0 / non_linear_factor)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def linear_stretch(data: np.ndarray, pmin: float = 0.5, pmax: float = 99.5) -> np.ndarray:
    """Standard percentile linear normalization."""
    arr = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = np.percentile(arr, (pmin, pmax))
    if vmax <= vmin:
        vmax = vmin + 1e-5
    norm = (arr - vmin) / (vmax - vmin)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def stretch_image(
    data: np.ndarray,
    stretch_type: str = "leakysb",
    **kwargs,
) -> np.ndarray:
    """
    Unified entry point for astronomical stretching.

    Parameters
    ----------
    data : np.ndarray
        2D astronomical array.
    stretch_type : str, default 'leakysb'
        Algorithm: 'leakysb', 'asinh', 'linear'.
    """
    mode = stretch_type.lower()
    if mode == "leakysb":
        return leaky_surface_brightness(data, **kwargs)
    elif mode == "asinh":
        return asinh_stretch(data, **kwargs)
    elif mode == "linear":
        return linear_stretch(data, **kwargs)
    else:
        return leaky_surface_brightness(data, **kwargs)
