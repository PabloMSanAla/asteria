"""Tests for astronomical photometric stretching routines."""

import numpy as np
from asteria.stretch.leakysb import (
    leaky_surface_brightness,
    asinh_stretch,
    linear_stretch,
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


def test_leakysb_monotonic_flux():
    # Brighter sources should map to higher surface brightness values
    fluxes = np.array([[10.0, 100.0, 1000.0, 10000.0]], dtype=np.float32)
    stretched = leaky_surface_brightness(fluxes)
    assert stretched[0, 0] < stretched[0, 1] < stretched[0, 2] < stretched[0, 3]


def test_asinh_and_linear():
    raw_img, _ = make_synthetic_field(size=128, seed=123)
    s_asinh = asinh_stretch(raw_img)
    s_lin = linear_stretch(raw_img)

    assert s_asinh.shape == (128, 128)
    assert s_lin.shape == (128, 128)
    assert np.all(s_asinh >= 0.0) and np.all(s_asinh <= 1.0)
    assert np.all(s_lin >= 0.0) and np.all(s_lin <= 1.0)


def test_stretch_image_dispatch():
    raw_img, _ = make_synthetic_field(size=64, seed=1)
    s1 = stretch_image(raw_img, stretch_type="leakysb")
    s2 = stretch_image(raw_img, stretch_type="asinh")
    s3 = stretch_image(raw_img, stretch_type="linear")

    assert s1.shape == (64, 64)
    assert s2.shape == (64, 64)
    assert s3.shape == (64, 64)
