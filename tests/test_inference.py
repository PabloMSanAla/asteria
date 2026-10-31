"""Tests for Asteria inference engine and tiled prediction."""

import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from asteria.models.unetpp import UNetPlusPlus
from asteria.inference.segmentor import AsteriaSegmentor
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


def test_segment_instances(mock_checkpoint_path):
    segmentor = AsteriaSegmentor.from_pretrained(mock_checkpoint_path, device="cpu")
    img, _ = make_synthetic_field(size=128, seed=7)

    result = segmentor.segment(
        img,
        interior_thresh=0.3,
        fg_thresh=0.3,
        min_area=4,
        tile_size=128,
        overlap=32,
    )
    assert isinstance(result.masks, list)
    assert isinstance(result.boxes, list)
    assert isinstance(result.centroids, list)
    assert result.segmap.shape == (128, 128)


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

