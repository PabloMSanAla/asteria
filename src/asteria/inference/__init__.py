"""Inference routines and segmentors for Asteria."""

from asteria.inference.device import get_optimal_device, get_device_info
from asteria.inference.segmentor import AsteriaSegmentor
from asteria.inference.watershed import deblend_sources, SegmentationResult

__all__ = [
    "AsteriaSegmentor",
    "deblend_sources",
    "SegmentationResult",
    "get_optimal_device",
    "get_device_info",
]
