"""
Unit and integration tests for Asteria Center Offset Voting and RCCA UNet++.
=============================================================================
"""

import numpy as np
import pytest
import torch
from pathlib import Path

from asteria.models.unetpp_offset import (
    CrissCrossAttention,
    RecurrentCrissCrossAttention,
    UNetPlusPlusCenterOffset,
)
from asteria.models.offset_losses import (
    CenterFocalLoss,
    MaskedSmoothL1Loss,
    AngularCosineDistanceLoss,
    CenterOffsetCompositeLoss,
    compute_instance_centroids,
    generate_center_offset_targets,
    generate_center_offset_batch,
    evaluate_center_detection,
    compute_offset_epe,
)
from asteria.inference.center_offset import (
    CenterOffsetSegmentor,
    extract_center_peaks,
    deblend_center_offset,
)
from asteria.visualization.plot import (
    plot_center_offset_multipanel,
    plot_center_offset_segmentation,
)
from asteria.utils.synthetic import make_synthetic_field
from asteria import load_model, load_center_offset


def test_criss_cross_attention():
    """Verify Criss-Cross Attention shapes and gradient flow."""
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    cca = CrissCrossAttention(in_channels=32)
    out = cca(x)
    assert out.shape == (2, 32, 16, 16)
    out.sum().backward()
    assert x.grad is not None

    rcca = RecurrentCrissCrossAttention(in_channels=32, recurrence=2)
    out_r = rcca(x)
    assert out_r.shape == (2, 32, 16, 16)


def test_unetpp_center_offset_model():
    """Verify UNet++ model multi-task forward pass with RCCA."""
    x = torch.randn(2, 3, 64, 64)
    model = UNetPlusPlusCenterOffset(
        in_channels=3,
        semantic_classes=1,
        features=(16, 32, 64, 128, 256),
        use_cca=True,
    )
    preds = model(x)
    assert "semantic" in preds
    assert "center" in preds
    assert "offset" in preds

    assert preds["semantic"].shape == (2, 1, 64, 64)
    assert preds["center"].shape == (2, 1, 64, 64)
    assert preds["offset"].shape == (2, 2, 64, 64)


def test_target_generation_numpy():
    """Verify mathematical correctness of center Gaussian peaks and offset vectors."""
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=np.int32)

    # Place an object centered at (20, 30) of radius 5
    cy_true, cx_true = 20, 30
    for y in range(cy_true - 5, cy_true + 6):
        for x in range(cx_true - 5, cx_true + 6):
            if (y - cy_true) ** 2 + (x - cx_true) ** 2 <= 25:
                mask[y, x] = 1

    fg, c_map, o_map, info = generate_center_offset_targets(mask, sigma=2.0)

    assert fg.shape == (h, w)
    assert c_map.shape == (h, w)
    assert o_map.shape == (2, h, w)

    # Centroid peak must be 1.0
    assert np.isclose(c_map[cy_true, cx_true], 1.0, atol=1e-3)
    assert c_map[0, 0] == 0.0

    # Test offset at pixel (cy_true + 3, cx_true - 2)
    # dy = cy - y = 20 - 23 = -3
    # dx = cx - x = 30 - 28 = +2
    py, px = cy_true + 3, cx_true - 2
    assert np.isclose(o_map[0, py, px], -3.0, atol=1e-3)
    assert np.isclose(o_map[1, py, px], 2.0, atol=1e-3)

    # Background offset must be 0
    assert o_map[0, 0, 0] == 0.0
    assert o_map[1, 0, 0] == 0.0


def test_target_generation_batch():
    """Verify collation of batch targets."""
    b, h, w = 2, 32, 32
    masks = np.zeros((b, h, w), dtype=np.int32)
    masks[0, 10:15, 10:15] = 1
    masks[1, 20:25, 20:25] = 2

    fg, ctr, off = generate_center_offset_batch(masks, sigma=2.0)
    assert fg.shape == (b, 1, h, w)
    assert ctr.shape == (b, 1, h, w)
    assert off.shape == (b, 2, h, w)


def test_center_offset_losses():
    """Verify multi-task loss calculation and backward step."""
    pred_sem = torch.randn(2, 1, 32, 32, requires_grad=True)
    pred_ctr = torch.randn(2, 1, 32, 32, requires_grad=True)
    pred_off = torch.randn(2, 2, 32, 32, requires_grad=True)
    preds = {"semantic": pred_sem, "center": pred_ctr, "offset": pred_off}

    target_fg = torch.zeros(2, 1, 32, 32)
    target_fg[:, :, 10:20, 10:20] = 1.0

    target_ctr = torch.zeros(2, 1, 32, 32)
    target_ctr[:, :, 15, 15] = 1.0

    target_off = torch.zeros(2, 2, 32, 32)
    target_off[:, 0, 10:20, 10:20] = 2.0
    target_off[:, 1, 10:20, 10:20] = -3.0

    criterion = CenterOffsetCompositeLoss(
        semantic_weight=1.0,
        center_weight=2.0,
        offset_weight=1.0,
        angle_weight=0.5,
    )
    loss_dict = criterion(preds, target_fg, target_ctr, target_off)

    assert "loss" in loss_dict
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["loss"].item() > 0

    loss_dict["loss"].backward()
    assert pred_sem.grad is not None
    assert pred_ctr.grad is not None
    assert pred_off.grad is not None


def test_metrics_evaluation():
    """Verify peak extraction, center matching, and offset EPE."""
    hm = np.zeros((64, 64), dtype=np.float32)
    hm[15, 20] = 0.95
    hm[45, 50] = 0.88

    peaks = extract_center_peaks(hm, threshold=0.3, min_distance=3)
    assert len(peaks) == 2

    gt_centroids = [(15.0, 20.0), (45.0, 50.0)]
    eval_res = evaluate_center_detection(peaks, gt_centroids, tolerance=2.0)
    assert eval_res["precision"] == 1.0
    assert eval_res["recall"] == 1.0
    assert eval_res["f1"] == 1.0
    assert eval_res["mae"] == 0.0

    pred_o = np.ones((2, 64, 64), dtype=np.float32) * 5.0
    target_o = np.ones((2, 64, 64), dtype=np.float32) * 2.0
    fg = np.ones((64, 64), dtype=np.float32)
    epe = compute_offset_epe(pred_o, target_o, fg)
    assert np.isclose(epe, 3.0 * np.sqrt(2.0), atol=1e-3)


def test_center_offset_segmentor_predict():
    """Verify CenterOffsetSegmentor predict and segment_instances on synthetic fields."""
    model = UNetPlusPlusCenterOffset(
        in_channels=3,
        semantic_classes=1,
        features=(16, 32, 64, 128, 256),
        use_cca=False,
    )
    segmentor = CenterOffsetSegmentor(model=model, device="cpu")

    img, _ = make_synthetic_field(size=64, seed=42)
    p_sem, p_ctr, p_off, stretched = segmentor.predict(img, stretch_type="leakysb")

    assert p_sem.shape == (64, 64)
    assert p_ctr.shape == (64, 64)
    assert p_off.shape == (2, 64, 64)
    assert stretched.shape == (64, 64)

    res = segmentor.segment_instances(img, stretch_type="leakysb", verbose=False)
    assert hasattr(res, "masks")
    assert hasattr(res, "boxes")
    assert hasattr(res, "scores")
    assert hasattr(res, "centroids")
    assert res.segmap.shape == (64, 64)


def test_center_offset_segmentor_tiled(tmp_path):
    """Verify tiled dense inference on larger exposures."""
    model = UNetPlusPlusCenterOffset(
        in_channels=3,
        semantic_classes=1,
        features=(16, 32, 64, 128, 256),
        use_cca=False,
    )
    segmentor = CenterOffsetSegmentor(model=model, device="cpu")

    img, _ = make_synthetic_field(size=128, seed=123)
    p_sem, p_ctr, p_off, stretched = segmentor.predict_dense(
        img,
        tile_size=64,
        overlap=16,
        batch_size=2,
    )

    assert p_sem.shape == (128, 128)
    assert p_ctr.shape == (128, 128)
    assert p_off.shape == (2, 128, 128)


def test_center_offset_factories(tmp_path):
    """Verify checkpoint saving, loading, and factory loading functions."""
    ckpt_path = tmp_path / "mock_offset.pt"
    model = UNetPlusPlusCenterOffset(
        in_channels=3,
        semantic_classes=1,
        features=(16, 32, 64, 128, 256),
        use_cca=False,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 3,
            "semantic_classes": 1,
            "features": (16, 32, 64, 128, 256),
            "use_cca": False,
        },
        ckpt_path,
    )

    seg1 = load_model(str(ckpt_path), device="cpu", model_type="center_offset")
    assert isinstance(seg1, CenterOffsetSegmentor)

    seg2 = load_center_offset(str(ckpt_path), device="cpu")
    assert isinstance(seg2, CenterOffsetSegmentor)


def test_plot_center_offset_multipanel(tmp_path):
    """Verify multi-panel visualization rendering."""
    stretched = np.random.rand(64, 64).astype(np.float32)
    p_sem = np.random.rand(64, 64).astype(np.float32)
    p_ctr = np.random.rand(64, 64).astype(np.float32)
    p_off = np.random.randn(2, 64, 64).astype(np.float32)
    peaks = [(20.0, 30.0, 0.9)]

    out_png = tmp_path / "test_offset_multipanel.png"
    fig = plot_center_offset_multipanel(
        stretched=stretched,
        pred_semantic=p_sem,
        pred_center=p_ctr,
        pred_offset=p_off,
        peaks=peaks,
        quiver_step=8,
        save_path=out_png,
    )
    assert out_png.exists()
    assert fig is not None


def test_plot_center_offset_segmentation(tmp_path):
    """Verify side-by-side stretched image vs. segmented mask rendering."""
    stretched = np.random.rand(64, 64).astype(np.float32)
    p_sem = np.zeros((64, 64), dtype=np.float32)
    p_ctr = np.zeros((64, 64), dtype=np.float32)
    p_off = np.zeros((2, 64, 64), dtype=np.float32)

    # Place two mock objects
    p_sem[15:25, 15:25] = 0.9
    p_ctr[20, 20] = 0.95
    p_sem[40:50, 40:50] = 0.85
    p_ctr[45, 45] = 0.90

    # Test direct computation from model outcomes (pred_semantic, pred_center, pred_offset)
    out_png1 = tmp_path / "test_seg_direct.png"
    fig1 = plot_center_offset_segmentation(
        stretched=stretched,
        pred_semantic=p_sem,
        pred_center=p_ctr,
        pred_offset=p_off,
        fg_thresh=0.5,
        center_thresh=0.3,
        save_path=out_png1,
    )
    assert out_png1.exists()
    assert fig1 is not None

    # Test with explicitly provided pred_mask
    mock_mask = np.zeros((64, 64), dtype=np.int32)
    mock_mask[10:20, 10:20] = 1
    mock_mask[30:40, 30:40] = 2

    out_png2 = tmp_path / "test_seg_explicit.png"
    fig2 = plot_center_offset_segmentation(
        stretched=stretched,
        pred_mask=mock_mask,
        save_path=out_png2,
    )
    assert out_png2.exists()
    assert fig2 is not None


def test_deblend_center_offset_large_frame_batching(tmp_path):
    """Verify batched deblending and plotting on a 4088x4088 exposure without OOM or crash."""
    h, w = 4088, 4088

    p_sem = np.zeros((h, w), dtype=np.float32)
    p_ctr = np.zeros((h, w), dtype=np.float32)
    p_off = np.zeros((2, h, w), dtype=np.float32)
    stretched = np.zeros((h, w), dtype=np.float32)

    np.random.seed(42)
    cy_list = np.random.randint(50, 4000, size=50)
    cx_list = np.random.randint(50, 4000, size=50)

    for cy, cx in zip(cy_list, cx_list):
        p_ctr[cy, cx] = 0.95
        ys = np.arange(cy - 5, cy + 6)
        xs = np.arange(cx - 5, cx + 6)
        p_sem[cy - 5 : cy + 6, cx - 5 : cx + 6] = 0.85
        stretched[cy - 5 : cy + 6, cx - 5 : cx + 6] = 0.75
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        p_off[0, cy - 5 : cy + 6, cx - 5 : cx + 6] = cy - grid_y
        p_off[1, cy - 5 : cy + 6, cx - 5 : cx + 6] = cx - grid_x

    # 1. Deblend with return_masks=False (memory-efficient mode)
    labeled_map, masks, boxes, scores, areas, centroids = deblend_center_offset(
        pred_semantic=p_sem,
        pred_center=p_ctr,
        pred_offsets=p_off,
        fg_thresh=0.5,
        center_thresh=0.3,
        return_masks=False,
    )
    assert labeled_map.shape == (h, w)
    assert len(masks) == 0
    assert len(boxes) == 50
    assert len(centroids) == 50
    assert len(areas) == 50

    # 2. Test plotting with large frame downsampling
    out_png = tmp_path / "test_4088_segmentation.png"
    fig = plot_center_offset_segmentation(
        stretched=stretched,
        pred_semantic=p_sem,
        pred_center=p_ctr,
        pred_offset=p_off,
        save_path=out_png,
        max_display_size=1024,
    )
    assert out_png.exists()
    assert fig is not None

