"""Inference routines, segmentors, and star removal for Asteria and StarNet3."""

from asteria.inference.device import (
    get_optimal_device,
    get_device_info,
    get_free_memory,
    get_adaptive_batch_size,
    empty_device_cache,
)
from asteria.inference.segmentor import AsteriaSegmentor
from asteria.inference.starnet3 import StarNet3, starnet3
from asteria.inference.center_offset import (
    CenterOffsetSegmentor,
    deblend_center_offset,
    extract_center_peaks,
)
from asteria.inference.watershed import deblend_sources, SegmentationResult

__all__ = [
    "AsteriaSegmentor",
    "CenterOffsetSegmentor",
    "StarNet3",
    "starnet3",
    "deblend_sources",
    "deblend_center_offset",
    "extract_center_peaks",
    "SegmentationResult",
    "get_optimal_device",
    "get_device_info",
    "get_free_memory",
    "get_adaptive_batch_size",
    "empty_device_cache",
]
