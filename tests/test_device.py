"""Tests for Asteria hardware device manager and fallbacks."""

import torch
from asteria.inference.device import get_optimal_device, get_device_info


def test_device_auto_detection():
    dev = get_optimal_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_device_explicit_cpu():
    dev = get_optimal_device("cpu")
    assert dev.type == "cpu"


def test_device_fallback_on_unavailable():
    # Requesting an unavailable device should gracefully fall back to CPU or MPS without crashing
    dev = get_optimal_device("cuda:999")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_device_info_string():
    info = get_device_info(torch.device("cpu"))
    assert "CPU" in info
