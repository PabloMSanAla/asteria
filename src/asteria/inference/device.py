"""
Hardware device auto-detection and graceful fallback manager.
Supports NVIDIA CUDA, Apple Silicon MPS, and standard CPU.
"""

from __future__ import annotations

import logging
from typing import Optional, Union
import torch

logger = logging.getLogger(__name__)


def get_optimal_device(requested: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Select optimal execution device with automatic hardware detection and fallback.

    Fallback Hierarchy:
    1. NVIDIA CUDA (if available and drivers initialized)
    2. Apple Silicon MPS (Metal Performance Shaders on macOS)
    3. CPU (standard universal fallback)

    Parameters
    ----------
    requested : str or torch.device, optional
        Explicitly requested device ('cuda', 'mps', 'cpu', 'cuda:0', etc.).
        If the requested device is unavailable, logs a warning and falls back.

    Returns
    -------
    torch.device
        Verified active PyTorch device.
    """
    if requested is not None:
        req_str = str(requested).lower()
        if "cuda" in req_str:
            if torch.cuda.is_available():
                return torch.device(requested)
            logger.warning("[asteria] CUDA was requested but is unavailable. Falling back.")
        elif "mps" in req_str:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("[asteria] Apple Silicon MPS was requested but is unavailable. Falling back.")
        elif "cpu" in req_str:
            return torch.device("cpu")
        else:
            logger.warning(f"[asteria] Unrecognized device '{requested}'. Falling back.")

    # Automatic detection hierarchy
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> str:
    """Return a human-readable summary of the active accelerator."""
    dev_type = device.type
    if dev_type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"NVIDIA CUDA ({name}, {vram_gb:.1f} GB VRAM)"
    elif dev_type == "mps":
        return "Apple Silicon MPS (Metal Performance Shaders)"
    return "Universal CPU"
