"""
Pre-trained model weights registry for Asteria.
Maintains official checkpoint metadata, training details, benchmark metrics, and URLs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class WeightMetadata:
    """Metadata and benchmark metrics for an official Asteria weight release."""
    version: str
    filename: str
    url: str
    sha256: Optional[str]
    in_channels: int
    num_classes: int
    features: tuple[int, ...]
    deep_supervision: bool
    epochs: int
    training_hardware: str
    dataset_description: str
    interior_iou: float
    boundary_iou_tol2: float
    soft_boundary_iou: float
    macro_iou: float
    release_date: str


# Official Model Weights Registry
WEIGHT_REGISTRY: Dict[str, WeightMetadata] = {
    "v0.1.0": WeightMetadata(
        version="v0.1.0",
        filename="unetpp_v0.1.0.pt",
        url="https://drive.google.com/file/d/1HfPMtcR0jjEfHIlAC-gEWAQeIFoljBwC/view?usp=sharing",
        sha256="71feeee2bc372693c361c234d2c114a1cc2d0805e03bc9e2c1c95004b3d21df4",
        in_channels=3,
        num_classes=3,
        features=(32, 64, 128, 256, 512),
        deep_supervision=False,
        epochs=80,
        training_hardware="NVIDIA Grace Hopper GH200 480GB (NASA HECC)",
        dataset_description="Synthetic Roman Space Telescope WFI F106 exposures (romanisim, Poisson+Gaussian noise, 2.8k cutouts)",
        interior_iou=0.754,
        boundary_iou_tol2=0.513,
        soft_boundary_iou=0.398,
        macro_iou=0.6335,
        release_date="2026-10-31",
    )
}

LATEST_VERSION = "v0.1.0"


def get_weight_metadata(version: str = "latest") -> WeightMetadata:
    """Retrieve metadata for a specified weight version."""
    v = LATEST_VERSION if version.lower() == "latest" else version
    if v not in WEIGHT_REGISTRY:
        available = list(WEIGHT_REGISTRY.keys())
        raise KeyError(f"Unknown weight version '{version}'. Available versions: {available}")
    return WEIGHT_REGISTRY[v]
