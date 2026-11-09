"""
Asteria: Lightweight Astronomical Deep Learning Inference Engine & Instance Deblender.

Powered by boundary-aware UNet++ for source deblending and StarNet3 for astronomical star removal.
"""

from __future__ import annotations

from typing import Union, Optional
from pathlib import Path

from asteria.models.unetpp import UNetPlusPlus
from asteria.models.unetpp_offset import (
    UNetPlusPlusCenterOffset,
    CrissCrossAttention,
    RecurrentCrissCrossAttention,
)
from asteria.models.starnet3 import StarNet3Model, ProgressiveMultiScaleUNetWithRefiner
from asteria.inference.segmentor import AsteriaSegmentor
from asteria.inference.starnet3 import StarNet3, starnet3
from asteria.inference.center_offset import (
    CenterOffsetSegmentor,
    deblend_center_offset,
    extract_center_peaks,
)
from asteria.inference.results import SegmentationResult
from asteria.inference.device import (
    get_optimal_device,
    get_device_info,
    get_free_memory,
    get_adaptive_batch_size,
    empty_device_cache,
)
from asteria.io.reader import AstroImage, load_image
from asteria.stretch.leakysb import (
    leaky_surface_brightness,
    asinh_stretch,
    linear_stretch,
    stretch_image,
    inverse_leaky_surface_brightness,
    inverse_stretch_image,
)
from asteria.visualization.plot import (
    plot_two_panel,
    plot_instance_overlay,
    plot_center_offset_multipanel,
    plot_center_offset_segmentation,
)
from asteria.visualization.clarity import (
    create_mask_edge_overlay,
    show_pixel_clarity,
    open_pixel_clarity,
    show_starnet_clarity,
    open_starnet_clarity,
)
from asteria.utils.synthetic import make_synthetic_field
from asteria.weights.registry import WEIGHT_REGISTRY, LATEST_VERSION
from asteria.weights.starnet3_registry import STARNET3_WEIGHT_REGISTRY, STARNET3_LATEST_VERSION
from asteria.weights.starnet3_manager import (
    download_starnet3_model,
    resolve_starnet3_checkpoint_path,
)

__version__ = "0.1.0"
__author__ = "Pablo M. Sanchez-Alarcon"
__license__ = "MIT"


def load_model(
    version_or_path: str = "latest",
    device: Optional[str] = None,
    model_type: str = "asteria",
) -> Union[AsteriaSegmentor, StarNet3, CenterOffsetSegmentor]:
    """
    Convenience factory to load an AsteriaSegmentor, StarNet3, or CenterOffsetSegmentor model.

    Parameters
    ----------
    version_or_path : str, default 'latest'
        Version string (e.g. 'v0.1.0', 'latest') or direct path to a local .pt file.
    device : str, optional
        Target accelerator ('cuda', 'mps', 'cpu'). Auto-selected if None.
    model_type : str, default 'asteria'
        Model architecture to load ('asteria', 'starnet3', or 'center_offset').

    Returns
    -------
    AsteriaSegmentor, StarNet3, or CenterOffsetSegmentor
    """
    m_type = model_type.lower()
    if m_type in ("starnet3", "starnet"):
        return StarNet3.from_pretrained(version_or_path=version_or_path, device=device)
    elif m_type in ("center_offset", "offset", "unetpp_offset"):
        return CenterOffsetSegmentor.from_pretrained(version_or_path=version_or_path, device=device)
    return AsteriaSegmentor.from_pretrained(version_or_path=version_or_path, device=device)


def load_starnet3(
    version_or_path: str = "latest",
    device: Optional[str] = None,
    **model_kwargs,
) -> StarNet3:
    """
    Convenience factory to load a StarNet3 star removal engine.

    Parameters
    ----------
    version_or_path : str, default 'latest'
        Version tag (e.g. 'v0.1.0', 'latest') or direct path to a .pt checkpoint.
    device : str, optional
        Target accelerator ('cuda', 'mps', 'cpu'). Auto-selected if None.
    **model_kwargs
        Optional architecture configuration overrides.

    Returns
    -------
    StarNet3
    """
    return StarNet3.from_pretrained(version_or_path=version_or_path, device=device, **model_kwargs)


def load_center_offset(
    version_or_path: str = "latest",
    device: Optional[str] = None,
    **model_kwargs,
) -> CenterOffsetSegmentor:
    """
    Convenience factory to load a Center Offset Voting instance segmentor.

    Parameters
    ----------
    version_or_path : str, default 'latest'
        Direct path to a .pt checkpoint or 'latest'.
    device : str, optional
        Target accelerator ('cuda', 'mps', 'cpu'). Auto-selected if None.
    **model_kwargs
        Optional architecture configuration overrides.

    Returns
    -------
    CenterOffsetSegmentor
    """
    return CenterOffsetSegmentor.from_pretrained(version_or_path=version_or_path, device=device, **model_kwargs)


__all__ = [
    "__version__",
    "AsteriaSegmentor",
    "CenterOffsetSegmentor",
    "StarNet3",
    "starnet3",
    "UNetPlusPlus",
    "UNetPlusPlusCenterOffset",
    "CrissCrossAttention",
    "RecurrentCrissCrossAttention",
    "StarNet3Model",
    "ProgressiveMultiScaleUNetWithRefiner",
    "AstroImage",
    "SegmentationResult",
    "load_model",
    "load_starnet3",
    "load_center_offset",
    "download_starnet3_model",
    "resolve_starnet3_checkpoint_path",
    "load_image",
    "stretch_image",
    "inverse_stretch_image",
    "inverse_leaky_surface_brightness",
    "leaky_surface_brightness",
    "asinh_stretch",
    "linear_stretch",
    "deblend_center_offset",
    "extract_center_peaks",
    "plot_two_panel",
    "plot_instance_overlay",
    "plot_center_offset_multipanel",
    "plot_center_offset_segmentation",
    "create_mask_edge_overlay",
    "show_pixel_clarity",
    "open_pixel_clarity",
    "show_starnet_clarity",
    "open_starnet_clarity",
    "make_synthetic_field",
    "get_optimal_device",
    "get_device_info",
    "get_free_memory",
    "get_adaptive_batch_size",
    "empty_device_cache",
    "WEIGHT_REGISTRY",
    "LATEST_VERSION",
    "STARNET3_WEIGHT_REGISTRY",
    "STARNET3_LATEST_VERSION",
]
