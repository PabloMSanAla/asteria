"""Data structures for instance segmentation and catalog results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class SegmentationResult:
    """
    Catalog of extracted astronomical source instances.

    Attributes
    ----------
    masks : List[np.ndarray]
        List of 2D boolean masks for each discrete celestial source.
    boxes : List[List[float]]
        Bounding boxes [x0, y0, x1, y1] in pixel coordinates.
    centroids : List[Tuple[float, float]]
        Centroid coordinates (cx, cy) in pixel space.
    scores : List[float]
        Mean core probability score per object.
    areas : List[int]
        Source area in pixels.
    segmap : np.ndarray
        2D integer segmentation map where 0 is background and k >= 1 is source ID k.
    sky_centroids : Optional[List[Tuple[float, float]]]
        Celestial sky coordinates (RA, Dec) if WCS is available.
    """
    masks: List[np.ndarray] = field(default_factory=list)
    boxes: List[List[float]] = field(default_factory=list)
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    areas: List[int] = field(default_factory=list)
    segmap: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    sky_centroids: Optional[List[Tuple[float, float]]] = None

    def __len__(self) -> int:
        return len(self.masks)
