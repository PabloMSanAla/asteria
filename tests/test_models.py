"""Tests for Asteria UNet++ neural network model."""

import pytest
import torch
from asteria.models.unetpp import UNetPlusPlus, VGGBlock


def test_vgg_block():
    block = VGGBlock(in_channels=3, out_channels=16)
    x = torch.randn(2, 3, 32, 32)
    out = block(x)
    assert out.shape == (2, 16, 32, 32)


def test_unetpp_forward():
    model = UNetPlusPlus(
        in_channels=3,
        num_classes=3,
        features=(16, 32, 64, 128, 256),
        deep_supervision=False,
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 64, 64)


def test_unetpp_deep_supervision():
    model = UNetPlusPlus(
        in_channels=3,
        num_classes=3,
        features=(16, 32, 64, 128, 256),
        deep_supervision=True,
    )
    x = torch.randn(1, 3, 64, 64)
    model.train()
    outs = model(x)
    assert isinstance(outs, list)
    assert len(outs) == 4
    for o in outs:
        assert o.shape == (1, 3, 64, 64)

    # In eval mode, returns final output tensor directly
    model.eval()
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 64, 64)


def test_unetpp_invalid_features():
    with pytest.raises(ValueError, match="requires exactly 5 feature depths"):
        UNetPlusPlus(features=(16, 32, 64))
