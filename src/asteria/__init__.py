"""
Asteria: Lightweight Astronomical Deep Learning Inference Engine & Instance Deblender.

Powered by boundary-aware UNet++ (Nested U-Net) for astronomical imaging.
"""

from __future__ import annotations

from asteria.models.unetpp import UNetPlusPlus
from asteria.inference.segmentor import AsteriaSegmentor
from asteria.inference.watershed import deblend_sources, SegmentationResult
from asteria.inference.device import get_optimal_device, get_device_info
from asteria.io.reader import AstroImage, load_image
from asteria.stretch.leakysb import (
    leaky_surface_brightness,
    asinh_stretch,
    linear_stretch,
    stretch_image,
)
from asteria.visualization.plot import plot_two_panel, plot_instance_overlay
from asteria.weights.registry import WEIGHT_REGISTRY, LATEST_VERSION

__version__ = "0.1.0"
__author__ = "Pablo M. Sanchez-Alarcon"
__license__ = "MIT"


def load_model(
    version_or_path: str = "latest",
    device: str | None = None,
) -> AsteriaSegmentor:
    """
    Convenience factory to load an AsteriaSegmentor.

    Parameters
    ----------
    version_or_path : str, default 'latest'
        Version string (e.g. 'v0.1.0', 'latest') or direct path to a local .pt file.
    device : str, optional
        Target accelerator ('cuda', 'mps', 'cpu'). Auto-selected if None.

    Returns
    -------
    AsteriaSegmentor
    """
    return AsteriaSegmentor.from_pretrained(version_or_path=version_or_path, device=device)


__all__ = [
    "__version__",
    "AsteriaSegmentor",
    "UNetPlusPlus",
    "AstroImage",
    "SegmentationResult",
    "load_model",
    "load_image",
    "stretch_image",
    "leaky_surface_brightness",
    "asinh_stretch",
    "linear_stretch",
    "deblend_sources",
    "plot_two_panel",
    "plot_instance_overlay",
    "get_optimal_device",
    "get_device_info",
    "WEIGHT_REGISTRY",
    "LATEST_VERSION",
]
