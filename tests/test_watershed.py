"""Tests for marker-controlled watershed deblender."""

import numpy as np
from asteria.inference.watershed import deblend_sources, SegmentationResult


def test_deblend_two_touching_sources():
    # Construct synthetic 3-class probability map (3, 64, 64) with two touching stars
    h, w = 64, 64
    probs = np.zeros((3, h, w), dtype=np.float32)
    probs[0] = 1.0  # Background default

    # Source 1 at (25, 25)
    y1, x1 = 25, 25
    y, x = np.ogrid[:h, :w]
    d1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    c1 = d1 <= 4.0
    probs[1, c1] = 0.9
    probs[0, c1] = 0.1

    # Source 2 at (25, 33)
    y2, x2 = 25, 33
    d2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    c2 = d2 <= 4.0
    probs[1, c2] = 0.9
    probs[0, c2] = 0.1

    # Boundary interface between the two sources
    edge = (x >= 28) & (x <= 29) & (y >= 22) & (y <= 28)
    probs[2, edge] = 0.8
    probs[1, edge] = 0.1
    probs[0, edge] = 0.1

    res = deblend_sources(probs, interior_thresh=0.5, fg_thresh=0.5, min_area=5)

    assert isinstance(res, SegmentationResult)
    # The watershed should cleanly resolve 2 distinct celestial objects
    assert len(res.masks) == 2
    assert len(res.boxes) == 2
    assert len(res.centroids) == 2
    assert np.max(res.segmap) == 2


def test_deblend_empty_sky():
    probs = np.zeros((3, 64, 64), dtype=np.float32)
    probs[0] = 1.0  # Pure background
    res = deblend_sources(probs)
    assert len(res.masks) == 0
    assert np.all(res.segmap == 0)
