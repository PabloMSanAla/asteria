"""Tests for astronomical image reader."""

import numpy as np
import pytest
from asteria.io.reader import (
    load_image,
    AstroImage,
    is_s3_url,
    parse_s3_url,
    download_s3_file,
)
from asteria.utils.synthetic import save_synthetic_fits
from unittest.mock import MagicMock, patch


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


def test_is_s3_url():
    """Verify S3 URI and HTTPS URL detection."""
    assert is_s3_url("s3://bucket-name/folder/image.asdf")
    assert is_s3_url("s3a://bucket-name/folder/image.fits.gz")
    assert is_s3_url("https://bucket-name.s3.amazonaws.com/image.fits")
    assert is_s3_url("https://bucket-name.s3.us-west-2.amazonaws.com/folder/image.asdf")
    assert is_s3_url("https://s3.us-west-2.amazonaws.com/bucket-name/folder/image.asdf")
    assert is_s3_url("https://s3.amazonaws.com/bucket-name/image.fits")

    # Non-S3 paths and URLs
    assert not is_s3_url("/local/path/image.fits")
    assert not is_s3_url("image.fits")
    assert not is_s3_url("https://otherdomain.com/image.fits")
    assert not is_s3_url("")


def test_parse_s3_url():
    """Verify bucket and key extraction from various S3 formats."""
    b, k = parse_s3_url("s3://roman-bucket/exposures/r0001_wfi.asdf")
    assert b == "roman-bucket"
    assert k == "exposures/r0001_wfi.asdf"

    b, k = parse_s3_url("s3a://roman-bucket/exposures/r0001_wfi.fits.gz")
    assert b == "roman-bucket"
    assert k == "exposures/r0001_wfi.fits.gz"

    b, k = parse_s3_url("https://roman-bucket.s3.amazonaws.com/mosaics/roman_mos.asdf")
    assert b == "roman-bucket"
    assert k == "mosaics/roman_mos.asdf"

    b, k = parse_s3_url("https://roman-bucket.s3.us-east-1.amazonaws.com/deep/roman.fits")
    assert b == "roman-bucket"
    assert k == "deep/roman.fits"

    b, k = parse_s3_url("https://s3.us-west-2.amazonaws.com/roman-bucket/deep/roman.fits")
    assert b == "roman-bucket"
    assert k == "deep/roman.fits"

    with pytest.raises(ValueError, match="Not a valid S3 URL"):
        parse_s3_url("https://example.com/not-s3.fits")


def test_download_s3_file_mocked(tmp_path):
    """Verify S3 download invocation with mocked boto3 client."""
    mock_client = MagicMock()
    mock_client.head_object.return_value = {"ContentLength": 1024}

    dest = tmp_path / "downloaded.fits"

    def fake_download(Bucket, Key, Filename, Callback=None):
        with open(Filename, "wb") as f:
            f.write(b"MOCK_FITS_DATA")
        if Callback:
            Callback(1024)

    mock_client.download_file.side_effect = fake_download

    out_path = download_s3_file("s3://my-test-bucket/data/sim.fits", destination=dest, client=mock_client)
    assert out_path == dest
    assert dest.exists()
    assert dest.read_bytes() == b"MOCK_FITS_DATA"
    mock_client.download_file.assert_called_once()


def test_load_image_s3_fits_mocked(tmp_path):
    """Verify load_image automatically detects S3 URL, downloads, and loads FITS."""
    fits_content_path = tmp_path / "source.fits"
    save_synthetic_fits(fits_content_path, size=64, seed=123)
    raw_bytes = fits_content_path.read_bytes()

    mock_client = MagicMock()
    mock_client.head_object.return_value = {"ContentLength": len(raw_bytes)}

    def fake_download(Bucket, Key, Filename, Callback=None):
        with open(Filename, "wb") as f:
            f.write(raw_bytes)
        if Callback:
            Callback(len(raw_bytes))

    mock_client.download_file.side_effect = fake_download

    with patch("boto3.client", return_value=mock_client):
        astro = load_image("s3://roman-stsci/survey/test_field.fits")
        assert isinstance(astro, AstroImage)
        assert astro.file_type == "fits"
        assert astro.s3_url == "s3://roman-stsci/survey/test_field.fits"
        assert astro.data.shape == (64, 64)
        assert astro.header is not None


def test_load_image_s3_asdf_mocked(tmp_path):
    """Verify load_image automatically detects S3 URL, downloads, and loads Roman ASDF."""
    pytest.importorskip("roman_datamodels")
    from roman_datamodels import datamodels

    l2_content_path = tmp_path / "source_l2.asdf"
    img = datamodels.ImageModel.create_fake_data()
    img.data = np.full((32, 32), 4.0, dtype=np.float32)
    img.meta.photometry.conversion_megajanskys = 0.1
    img.save(str(l2_content_path))
    raw_bytes = l2_content_path.read_bytes()

    mock_client = MagicMock()
    mock_client.head_object.return_value = {"ContentLength": len(raw_bytes)}

    def fake_download(Bucket, Key, Filename, Callback=None):
        with open(Filename, "wb") as f:
            f.write(raw_bytes)
        if Callback:
            Callback(len(raw_bytes))

    mock_client.download_file.side_effect = fake_download

    with patch("boto3.client", return_value=mock_client):
        astro = load_image("s3://roman-data-bucket/exposures/r0000101_wfi.asdf")
        assert isinstance(astro, AstroImage)
        assert astro.file_type == "asdf"
        assert astro.level == 2
        assert astro.s3_url == "s3://roman-data-bucket/exposures/r0000101_wfi.asdf"
        assert np.allclose(astro.data, 0.4)


