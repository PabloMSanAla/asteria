"""Tests for LeakySB (Leaky Surface Brightness) non-linear stretching routines."""

import numpy as np

from asteria.stretch.leakysb import (
    inverse_leaky_surface_brightness,
    inverse_stretch_image,
    leakySB,
    leakySB_inverse,
    leaky_surface_brightness,
    stretch_image,
)
from asteria.utils.synthetic import make_synthetic_field


def test_leakysb_synthetic_field():
    raw_img, _ = make_synthetic_field(size=128, seed=42)
    stretched = leaky_surface_brightness(raw_img, zero_points=31.4, pixel_scale=0.11)

    assert stretched.shape == raw_img.shape
    assert stretched.dtype == np.float32
    assert np.all(stretched >= 0.0)
    assert np.all(stretched <= 1.0)
    assert not np.any(np.isnan(stretched))
    assert not np.any(np.isinf(stretched))


def test_leakysb_rgb_and_inverse():
    raw_img, _ = make_synthetic_field(size=64, seed=42)
    rgb = leakySB(raw_img, raw_img, raw_img, zero_points=31.4, pixel_scale=0.11)
    assert rgb.shape == (64, 64, 3)
    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

    # Invert back to physical flux
    r_rec, g_rec, b_rec = leakySB_inverse(rgb, zero_points=31.4, pixel_scale=0.11)
    assert np.allclose(r_rec, raw_img, atol=1e-2)


def test_leakysb_monotonic_flux():
    # Brighter sources should map to higher surface brightness values
    fluxes = np.array([[10.0, 100.0, 1000.0, 10000.0]], dtype=np.float32)
    stretched = leaky_surface_brightness(fluxes)
    assert stretched[0, 0] < stretched[0, 1] < stretched[0, 2] < stretched[0, 3]


def test_leakysb_sky_subtraction():
    """Verify sky subtraction prior to LeakySB stretching."""
    raw_img, _ = make_synthetic_field(size=64, seed=42)
    sky_val = 15.0

    s_with_sky = stretch_image(raw_img, stretch_type="leakysb", sky=sky_val)
    s_manual = stretch_image(raw_img - sky_val, stretch_type="leakysb", sky=0.0)

    assert np.allclose(s_with_sky, s_manual, atol=1e-5)


def test_inverse_leaky_surface_brightness_roundtrip():
    """Verify exact 2D roundtrip through forward LeakySB and inverse LeakySB."""
    raw_img, _ = make_synthetic_field(size=64, seed=42)
    zp = 31.4
    ps = 0.11

    stretched = leaky_surface_brightness(raw_img, zero_points=zp, pixel_scale=ps)
    reconstructed = inverse_leaky_surface_brightness(stretched, zero_points=zp, pixel_scale=ps)

    assert reconstructed.shape == raw_img.shape
    assert reconstructed.dtype == np.float32
    # Check close match across physical flux range
    assert np.allclose(reconstructed, raw_img, atol=1e-2)


def test_inverse_leakysb_with_sky():
    """Verify that inverse_stretch_image restores original baseline with sky added back."""
    raw_img, _ = make_synthetic_field(size=64, seed=77)
    sky_val = 12.5

    stretched = stretch_image(raw_img, stretch_type="leakysb", sky=sky_val)
    reconstructed = inverse_stretch_image(stretched, stretch_type="leakysb", sky=sky_val)

    assert reconstructed.shape == raw_img.shape
    assert np.allclose(reconstructed, raw_img, atol=1e-2)
