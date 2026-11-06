"""
Unit tests for Asteria Pixel Clarity visualization.
===================================================
"""

import os
from pathlib import Path
import numpy as np
import pytest

from asteria.visualization.clarity import (
    create_mask_edge_overlay,
    show_pixel_clarity,
    open_pixel_clarity,
    show_starnet_clarity,
    open_starnet_clarity,
    _prepare_starnet_mef,
    HAS_PIXEL_CLARITY,
)


def test_create_mask_edge_overlay_3class():
    """Verify high-contrast RGB overlay generation for 3-class semantic masks."""
    h, w = 64, 64
    stretched = np.full((h, w), 0.2, dtype=np.float32)
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    # Core interior (class 1)
    pred_mask[20:40, 20:40] = 1
    # Deblending boundary (class 2)
    pred_mask[18:42, 18:42] = np.where(pred_mask[18:42, 18:42] == 0, 2, pred_mask[18:42, 18:42])

    overlay = create_mask_edge_overlay(stretched, pred_mask)

    assert overlay.shape == (h, w, 3)
    assert overlay.dtype == np.float32
    assert np.all(overlay >= 0.0) and np.all(overlay <= 1.0)

    # Sky background should match grayscale stretched
    assert np.allclose(overlay[0, 0], [0.2, 0.2, 0.2], atol=1e-4)

    # Mask (core) should be tinted cyan (B channel > R channel)
    assert overlay[30, 30, 2] > overlay[30, 30, 0]

    # Edge should be tinted red (R channel > B channel)
    assert overlay[19, 19, 0] > overlay[19, 19, 2]


def test_create_mask_edge_overlay_binary():
    """Verify morphological edge extraction when only binary masks (0, 1) are given."""
    h, w = 50, 50
    stretched = np.full((h, w), 0.1, dtype=np.float32)
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    binary_mask[15:35, 15:35] = 1

    overlay = create_mask_edge_overlay(stretched, binary_mask)

    assert overlay.shape == (h, w, 3)
    # Interior must be tinted
    assert overlay[25, 25, 2] > overlay[25, 25, 0]
    # Edge surrounding 15:35 must be tinted red
    assert overlay[14, 25, 0] > overlay[14, 25, 2]


def test_create_mask_edge_overlay_instances():
    """Verify multi-instance labeled masks overlay."""
    h, w = 64, 64
    stretched = np.full((h, w), 0.1, dtype=np.float32)
    inst_map = np.zeros((h, w), dtype=np.int32)
    inst_map[10:30, 10:30] = 1
    inst_map[10:30, 30:50] = 2

    overlay = create_mask_edge_overlay(stretched, inst_map)
    assert overlay.shape == (h, w, 3)
    assert np.all(overlay >= 0.0) and np.all(overlay <= 1.0)


@pytest.mark.skipif(not HAS_PIXEL_CLARITY, reason="pixel-clarity not installed")
def test_show_pixel_clarity():
    """Verify pixel_clarity.show creates widget Viewer."""
    h, w = 32, 32
    stretched = np.random.rand(h, w).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    viewer = show_pixel_clarity(stretched, mask, title="Unit Test Viewer", anon=True)
    assert viewer is not None
    assert hasattr(viewer, "state")


@pytest.mark.skipif(not HAS_PIXEL_CLARITY, reason="pixel-clarity not installed")
def test_open_pixel_clarity_and_save(tmp_path, monkeypatch):
    """Verify open_pixel_clarity in default layer mode creates FITS mask container and saves overlay PNG."""
    import pixel_clarity as pc
    from astropy.io import fits

    mock_calls = []

    def mock_popup(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return "mock_popup_controller"

    monkeypatch.setattr(pc, "popup", mock_popup)

    h, w = 32, 32
    stretched = np.random.rand(h, w).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    out_png = tmp_path / "clarity_overlay.png"
    popup = open_pixel_clarity(stretched, mask, title="Unit Test Popup", save_path=out_png)

    assert popup == "mock_popup_controller"
    assert len(mock_calls) == 1
    args, kwargs = mock_calls[0]
    # Default layer mode: multi-extension FITS file with Primary + MASK
    assert isinstance(args[0], str)
    assert args[0].endswith(".fits")
    with fits.open(args[0]) as hdul:
        assert len(hdul) == 2
        assert hdul[0].name == "PRIMARY"
        assert hdul[1].name == "MASK"
        assert hdul[1].data.shape == (h, w)
    assert kwargs.get("title") == "Unit Test Popup"
    assert out_png.exists()
    assert out_png.stat().st_size > 0


@pytest.mark.skipif(not HAS_PIXEL_CLARITY, reason="pixel-clarity not installed")
def test_open_pixel_clarity_composite_mode(tmp_path, monkeypatch):
    """Verify open_pixel_clarity with mode='composite' passes PNG overlay."""
    import pixel_clarity as pc

    mock_calls = []

    def mock_popup(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return "mock_popup_controller"

    monkeypatch.setattr(pc, "popup", mock_popup)

    h, w = 32, 32
    stretched = np.random.rand(h, w).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    popup = open_pixel_clarity(stretched, mask, mode="composite")
    assert popup == "mock_popup_controller"
    args, kwargs = mock_calls[0]
    assert isinstance(args[0], str)
    assert args[0].endswith(".png")
    assert kwargs.get("names") == ["Mask & Edge Overlay"]


@pytest.mark.skipif(not HAS_PIXEL_CLARITY, reason="pixel-clarity not installed")
def test_open_pixel_clarity_side_by_side(tmp_path, monkeypatch):
    """Verify open_pixel_clarity with side_by_side=True passes both overlay and stretched."""
    import pixel_clarity as pc

    mock_calls = []

    def mock_popup(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return "mock_popup_controller"

    monkeypatch.setattr(pc, "popup", mock_popup)

    h, w = 32, 32
    stretched = np.random.rand(h, w).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    popup = open_pixel_clarity(stretched, mask, side_by_side=True)
    assert popup == "mock_popup_controller"
    args, kwargs = mock_calls[0]
    assert isinstance(args[0], list) and len(args[0]) == 2
    assert kwargs.get("layout") == "grid"
    assert kwargs.get("names") == ["Mask & Edge Overlay", "Stretched Exposure"]


def test_prepare_starnet_mef(tmp_path):
    """Verify serialization of 2-HDU multi-extension FITS with raw data and starless flux."""
    from astropy.io import fits
    h, w = 32, 32
    raw = np.full((h, w), 10.0, dtype=np.float32)
    starless_flux = np.full((h, w), 2.5, dtype=np.float32)

    save_path = tmp_path / "custom_starnet.fits"
    out_path = _prepare_starnet_mef(raw, starless_flux, save_path=save_path)
    assert out_path == save_path
    assert out_path.exists()

    with fits.open(out_path) as hdul:
        assert len(hdul) == 2
        # Primary HDU: raw un-stretched
        assert hdul[0].header.get("EXTNAME") == "ORIGINAL"
        assert np.allclose(hdul[0].data, 10.0)
        # Image HDU: starless prediction
        assert hdul[1].name == "STARLESS"
        assert np.allclose(hdul[1].data, 2.5)


@pytest.mark.skipif(not HAS_PIXEL_CLARITY, reason="pixel-clarity not installed")
def test_open_starnet_clarity(monkeypatch):
    """Verify open_starnet_clarity forwards 2-HDU multi-extension FITS to pixel_clarity.popup."""
    import pixel_clarity as pc
    from astropy.io import fits

    mock_calls = []

    def mock_popup(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return "mock_popup_controller"

    monkeypatch.setattr(pc, "popup", mock_popup)

    raw = np.full((16, 16), 15.0, dtype=np.float32)
    starless_flux = np.full((16, 16), 3.0, dtype=np.float32)

    res = open_starnet_clarity(raw, starless_flux, title="Test StarNet Popup")
    assert res == "mock_popup_controller"
    assert len(mock_calls) == 1
    args, kwargs = mock_calls[0]
    fits_path = args[0]
    assert isinstance(fits_path, str)
    assert fits_path.endswith(".fits")

    with fits.open(fits_path) as hdul:
        assert len(hdul) == 2
        assert hdul[0].header.get("EXTNAME") == "ORIGINAL"
        assert hdul[1].name == "STARLESS"


@pytest.mark.skipif(not HAS_PIXEL_CLARITY, reason="pixel-clarity not installed")
def test_show_starnet_clarity(monkeypatch):
    """Verify show_starnet_clarity forwards 2-HDU multi-extension FITS to pixel_clarity.show."""
    import pixel_clarity as pc
    from astropy.io import fits

    mock_calls = []

    def mock_show(*args, **kwargs):
        mock_calls.append((args, kwargs))
        return "mock_viewer_controller"

    monkeypatch.setattr(pc, "show", mock_show)

    raw = np.full((16, 16), 15.0, dtype=np.float32)
    starless_flux = np.full((16, 16), 3.0, dtype=np.float32)

    res = show_starnet_clarity(raw, starless_flux, title="Test StarNet Show")
    assert res == "mock_viewer_controller"
    assert len(mock_calls) == 1
    args, kwargs = mock_calls[0]
    fits_path = args[0]
    assert isinstance(fits_path, str)
    assert fits_path.endswith(".fits")

    with fits.open(fits_path) as hdul:
        assert len(hdul) == 2
        assert hdul[0].header.get("EXTNAME") == "ORIGINAL"
        assert hdul[1].name == "STARLESS"
