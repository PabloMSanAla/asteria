"""Tests for astronomical image reader."""

import numpy as np
import pytest
from asteria.io.reader import load_image, AstroImage
from asteria.utils.synthetic import save_synthetic_fits


def test_load_image_from_numpy():
    arr = np.zeros((128, 128), dtype=np.float32)
    astro = load_image(arr)
    assert isinstance(astro, AstroImage)
    assert astro.data.shape == (128, 128)
    assert astro.file_type == "array"


def test_load_image_from_synthetic_fits(tmp_path):
    fits_file = tmp_path / "test_simulated.fits"
    save_synthetic_fits(fits_file, size=128, seed=42)

    astro = load_image(fits_file)
    assert isinstance(astro, AstroImage)
    assert astro.data.shape == (128, 128)
    assert astro.file_type == "fits"
    assert astro.header is not None
    assert astro.header.get("INSTRUME") == "ASTERIA_SYNTHETIC"


def test_load_image_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_image("nonexistent_galaxy_scene.fits")


def test_load_image_invalid_dim():
    with pytest.raises(ValueError, match="Expected 2D array"):
        load_image(np.zeros((10, 10, 10)))
