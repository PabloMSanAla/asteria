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


def test_load_image_roman_asdf_level2(tmp_path):
    pytest.importorskip("roman_datamodels")
    from roman_datamodels import datamodels

    l2_file = tmp_path / "roman_wfi_l2.asdf"
    img = datamodels.ImageModel.create_fake_data()
    img.data = np.full((32, 32), 10.0, dtype=np.float32)
    img.meta.photometry.conversion_megajanskys = 0.05
    img.save(str(l2_file))

    astro = load_image(l2_file)
    assert isinstance(astro, AstroImage)
    assert astro.file_type == "asdf"
    assert astro.level == 2
    assert astro.unit == "MJy/sr"
    assert astro.zero_point == 20.472
    assert astro.pixel_scale == 1.0
    # Verify data scaled by conversion_megajanskys (10.0 * 0.05 = 0.5)
    assert np.allclose(astro.data, 0.5)
    # Verify Level 2 automatically measures sky via sigma_clipped_stats median
    assert np.isclose(astro.sky, 0.5)
    assert astro.wcs is not None
    # Verify coordinates callable/convertible
    sky = astro.pixel_to_world(16, 16)
    assert sky is not None

    # Verify explicit sky override
    astro_custom_sky = load_image(l2_file, sky=0.12)
    assert astro_custom_sky.sky == 0.12
    assert astro_custom_sky.pixel_scale == 1.0


def test_load_image_roman_asdf_level3(tmp_path):
    pytest.importorskip("roman_datamodels")
    from roman_datamodels import datamodels

    l3_file = tmp_path / "roman_wfi_l3_mosaic.asdf"
    mos = datamodels.MosaicModel.create_fake_data()
    mos.data = np.full((32, 32), 2.5, dtype=np.float32)
    mos.save(str(l3_file))

    astro = load_image(l3_file)
    assert isinstance(astro, AstroImage)
    assert astro.file_type == "asdf"
    assert astro.level == 3
    assert astro.unit == "MJy/sr"
    assert astro.zero_point == 20.472
    assert astro.pixel_scale == 1.0
    # Level 3 defaults to sky 0.0
    assert astro.sky == 0.0
    # Level 3 is already in MJy/sr: remains 2.5
    assert np.allclose(astro.data, 2.5)


def test_load_image_sky_flag():
    """Verify sky flag behavior across levels and defaults."""
    np.random.seed(42)
    arr = np.random.randn(64, 64).astype(np.float32) + 7.5

    # Level 2 automatic measurement
    astro_l2_auto = load_image(arr, level=2)
    assert astro_l2_auto.level == 2
    assert np.isclose(astro_l2_auto.sky, 7.5, atol=0.2)

    # Level 2 explicit override
    astro_l2_override = load_image(arr, level=2, sky=3.0)
    assert astro_l2_override.sky == 3.0

    # Level 3 / default: sky defaults to 0.0
    astro_l3 = load_image(arr, level=3)
    assert astro_l3.sky == 0.0

    astro_default = load_image(arr)
    assert astro_default.sky == 0.0

    # Non-level-2 with explicit sky
    astro_explicit = load_image(arr, sky=4.2)
    assert astro_explicit.sky == 4.2

