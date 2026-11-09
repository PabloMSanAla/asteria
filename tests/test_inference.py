"""Tests for Asteria inference engine and tiled prediction."""

import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from asteria.models.unetpp import UNetPlusPlus
from asteria.inference.segmentor import AsteriaSegmentor, _resolve_stretch_kwargs
from asteria.io.reader import AstroImage
from asteria.utils.synthetic import make_synthetic_field
from asteria.visualization.plot import plot_two_panel, plot_instance_overlay


@pytest.fixture
def mock_checkpoint_path(tmp_path):
    """Create a lightweight model checkpoint on-the-fly for offline unit testing."""
    ckpt_file = tmp_path / "mock_unetpp.pt"
    model = UNetPlusPlus(
        in_channels=3,
        num_classes=3,
        features=(16, 32, 64, 128, 256),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 3,
            "num_classes": 3,
            "features": (16, 32, 64, 128, 256),
            "deep_supervision": False,
        },
        ckpt_file,
    )
    return ckpt_file


def test_segmentor_initialization(mock_checkpoint_path):
    segmentor = AsteriaSegmentor.from_pretrained(mock_checkpoint_path, device="cpu")
    assert segmentor.device.type == "cpu"
    assert segmentor.model.num_classes == 3


def test_predict_single_patch(mock_checkpoint_path):
    segmentor = AsteriaSegmentor.from_pretrained(mock_checkpoint_path, device="cpu")
    img, _ = make_synthetic_field(size=128, seed=42)

    pred_mask, probs, stretched = segmentor.predict(img, stretch_type="leakysb")

    assert pred_mask.shape == (128, 128)
    assert probs.shape == (3, 128, 128)
    assert stretched.shape == (128, 128)
    assert pred_mask.dtype == np.uint8
    assert set(np.unique(pred_mask)).issubset({0, 1, 2})


def test_predict_dense_tiling(mock_checkpoint_path):
    segmentor = AsteriaSegmentor.from_pretrained(mock_checkpoint_path, device="cpu")
    # Small simulated image (256x256) evaluated with 128x128 tiles and 32px overlap
    img, _ = make_synthetic_field(size=256, seed=99)

    pred_mask, probs, stretched = segmentor.predict_dense(
        img,
        tile_size=128,
        overlap=32,
        batch_size=2,
        stretch_type="leakysb",
        verbose=False,
    )

    assert pred_mask.shape == (256, 256)
    assert probs.shape == (3, 256, 256)
    assert stretched.shape == (256, 256)
    # Check probability partition of unity: sum over classes ~ 1.0
    prob_sums = np.sum(probs, axis=0)
    assert np.allclose(prob_sums, 1.0, atol=1e-3)


def test_predict_dense_batch_size_none(mock_checkpoint_path):
    """Verify predict_dense runs adaptively when batch_size is None."""
    segmentor = AsteriaSegmentor.from_pretrained(mock_checkpoint_path, device="cpu")
    img, _ = make_synthetic_field(size=256, seed=99)

    pred_mask, probs, stretched = segmentor.predict_dense(
        img,
        tile_size=128,
        overlap=32,
        batch_size=None,
        stretch_type="leakysb",
        verbose=True,
    )

    assert pred_mask.shape == (256, 256)
    assert probs.shape == (3, 256, 256)
    assert stretched.shape == (256, 256)
    assert np.allclose(np.sum(probs, axis=0), 1.0, atol=1e-3)


def test_predict_dense_custom_leakysb_parameters(mock_checkpoint_path):
    segmentor = AsteriaSegmentor.from_pretrained(mock_checkpoint_path, device="cpu")
    img, _ = make_synthetic_field(size=128, seed=42)

    # Test passing explicit parameters to predict_dense
    pred_mask, probs, stretched = segmentor.predict_dense(
        img,
        tile_size=64,
        overlap=16,
        zero_points=20.472,
        pixel_scale=0.11,
        mmin=14.0,
        mmax=30.0,
        leak_slope=0.08,
    )
    assert pred_mask.shape == (128, 128)
    assert stretched.min() >= 0.0
    assert stretched.max() <= 1.0


def test_predict_dense_leakysb_aliases(mock_checkpoint_path):
    segmentor = AsteriaSegmentor.from_pretrained(mock_checkpoint_path, device="cpu")
    img, _ = make_synthetic_field(size=128, seed=42)

    # Test passing alias kwargs (zeropoint, pixelscale, leakslope)
    pred_mask, probs, stretched = segmentor.predict_dense(
        img,
        tile_size=64,
        overlap=16,
        zeropoint=20.472,
        pixelscale=0.11,
        leakslope=0.08,
    )
    assert pred_mask.shape == (128, 128)
    assert stretched.min() >= 0.0
    assert stretched.max() <= 1.0


def test_plot_two_panel(tmp_path):
    h, w = 128, 128
    stretched = np.random.uniform(0.0, 1.0, (h, w)).astype(np.float32)
    pred_mask = np.random.choice([0, 1, 2], size=(h, w)).astype(np.uint8)

    save_file = tmp_path / "panel_test.png"
    fig = plot_two_panel(stretched, pred_mask, save_path=save_file)
    plt.close(fig)

    assert save_file.exists()
    assert save_file.stat().st_size > 0


def test_load_sam2roman_gh200_checkpoint():
    gh200_path = Path("/Users/pmsanch1/code/sam2roman/experiments/unet/gh200_3k/unetpp_best.pt")
    if not gh200_path.exists():
        pytest.skip("Local gh200_3k checkpoint not found")
    segmentor = AsteriaSegmentor.from_pretrained(gh200_path, device="cpu")
    assert segmentor.model.num_classes == 3
    assert segmentor.model.in_channels == 3
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = segmentor.model(x)
    assert out.shape == (1, 3, 128, 128)


def test_resolve_stretch_kwargs_auto_pixel_scale_levels():
    """Verify Level 2 and Level 3 images automatically set pixel_scale to 1.0."""
    dummy_data = np.zeros((32, 32), dtype=np.float32)

    # Level 2 ASDF image without explicit pixel_scale -> 1.0
    astro_l2 = AstroImage(data=dummy_data, level=2, unit="MJy/sr", zero_point=20.472)
    kw_l2 = _resolve_stretch_kwargs(astro=astro_l2, stretch_type="leakysb")
    assert kw_l2["pixel_scale"] == 1.0

    # Level 3 ASDF image without explicit pixel_scale -> 1.0
    astro_l3 = AstroImage(data=dummy_data, level=3, unit="MJy/sr", zero_point=20.472)
    kw_l3 = _resolve_stretch_kwargs(astro=astro_l3, stretch_type="leakysb")
    assert kw_l3["pixel_scale"] == 1.0

    # Non-level image without explicit pixel_scale -> not forced to 1.0
    astro_fits = AstroImage(data=dummy_data, level=None, unit="DN/s", zero_point=31.4)
    kw_fits = _resolve_stretch_kwargs(astro=astro_fits, stretch_type="leakysb")
    assert "pixel_scale" not in kw_fits

    # Explicit override takes precedence over auto 1.0
    kw_override = _resolve_stretch_kwargs(astro=astro_l2, stretch_type="leakysb", pixel_scale=0.11)
    assert kw_override["pixel_scale"] == 0.11

