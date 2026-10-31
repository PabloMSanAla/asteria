"""
Marker-controlled watershed instance deblender for astronomical sources.
Converts continuous 3-class boundary-aware probability maps into discrete catalogs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import scipy.ndimage as ndimage


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
    """
    masks: List[np.ndarray] = field(default_factory=list)
    boxes: List[List[float]] = field(default_factory=list)
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    areas: List[int] = field(default_factory=list)
    segmap: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))

    def __len__(self) -> int:
        return len(self.masks)


def deblend_sources(
    probs: np.ndarray,
    interior_thresh: float = 0.45,
    fg_thresh: float = 0.50,
    min_area: int = 10,
    max_area: Optional[int] = None,
) -> SegmentationResult:
    """
    Execute marker-controlled watershed deblending on 3-class UNet++ probabilities.

    Parameters
    ----------
    probs : np.ndarray
        Softmax probability map of shape (3, H, W) where:
        - probs[0]: Sky Background probability
        - probs[1]: Source Interior / Core probability
        - probs[2]: Deblending Boundary / Edge probability
    interior_thresh : float, default 0.45
        Probability threshold on source interior to identify discrete seeds.
    fg_thresh : float, default 0.50
        Threshold on total foreground probability (1 - P_bg).
    min_area : int, default 10
        Minimum area in pixels to retain a detected object.
    max_area : int, optional
        Maximum area cap to filter out large background artifacts.

    Returns
    -------
    SegmentationResult
        Extracted source instances with masks, boxes, centroids, and photometry statistics.
    """
    if probs.ndim != 3 or probs.shape[0] != 3:
        raise ValueError(f"Expected probabilities of shape (3, H, W), got {probs.shape}")

    _, h, w = probs.shape
    p_bg = probs[0]
    p_int = probs[1]
    p_edge = probs[2]

    # 1. Detect topological core seeds
    core_mask = (p_int >= interior_thresh)
    markers, num_cores = ndimage.label(core_mask)

    # 2. Total foreground region
    fg_mask = (p_bg < fg_thresh) | ((p_int + p_edge) >= fg_thresh)

    # 3. Distance transform watershed expansion
    if num_cores > 0 and np.any(fg_mask):
        dist = ndimage.distance_transform_edt(fg_mask)
        dist_int = (np.max(dist) - dist).astype(np.uint8)
        labeled_map = ndimage.watershed_ift(dist_int, markers.astype(np.int32))
        labeled_map[~fg_mask] = 0
    else:
        labeled_map = np.zeros((h, w), dtype=np.int32)

    # 4. Filter objects by area and compute metadata
    masks: List[np.ndarray] = []
    boxes: List[List[float]] = []
    centroids: List[Tuple[float, float]] = []
    scores: List[float] = []
    areas: List[int] = []

    unique_ids = np.unique(labeled_map)
    unique_ids = unique_ids[unique_ids > 0]

    final_segmap = np.zeros_like(labeled_map)
    new_id = 1

    for uid in unique_ids:
        m = (labeled_map == uid)
        area = int(np.sum(m))

        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        ys, xs = np.where(m)
        x0, x1 = float(np.min(xs)), float(np.max(xs))
        y0, y1 = float(np.min(ys)), float(np.max(ys))

        # Centroid weighted by interior probability
        weights = p_int[m]
        w_sum = float(np.sum(weights))
        if w_sum > 0:
            cx = float(np.sum(xs * weights) / w_sum)
            cy = float(np.sum(ys * weights) / w_sum)
        else:
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

        score = float(np.mean(p_int[m]))

        masks.append(m)
        boxes.append([x0, y0, x1, y1])
        centroids.append((cx, cy))
        scores.append(score)
        areas.append(area)

        final_segmap[m] = new_id
        new_id += 1

    return SegmentationResult(
        masks=masks,
        boxes=boxes,
        centroids=centroids,
        scores=scores,
        areas=areas,
        segmap=final_segmap,
    )
