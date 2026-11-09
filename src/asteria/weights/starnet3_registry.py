"""
Pre-trained model weights registry for StarNet3.
Maintains official StarNet3 checkpoint metadata, training details, and download URLs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class StarNet3WeightMetadata:
    """Metadata and benchmark details for an official StarNet3 weight release."""
    version: str
    filename: str
    url: str
    sha256: Optional[str]
    dim: int
    unet_scales: Tuple[int, ...]
    use_pos_guide: bool
    training_hardware: str
    dataset_description: str
    release_date: str


# Official StarNet3 Model Weights Registry
STARNET3_WEIGHT_REGISTRY: Dict[str, StarNet3WeightMetadata] = {
    "v0.1.0": StarNet3WeightMetadata(
        version="v0.1.0",
        filename="starnet3_v0.1.0.pt",
        url="https://github.com/PabloMSanAla/asteria/releases/download/v0.1.0/starnet3_best.pt",
        sha256="7b91d24c084f49e49195d852a4e8d35756358c56c2d8c360db4f7962483d987a",
        dim=24,
        unet_scales=(2, 16),
        use_pos_guide=True,
        training_hardware="NVIDIA Grace Hopper GH200 480GB (NASA HECC)",
        dataset_description="Roman WFI F106 synthetic exposures with LeakySB stretch (36k iterations)",
        release_date="2026-11-01",
    )
}

STARNET3_LATEST_VERSION = "v0.1.0"


def get_starnet3_weight_metadata(version: str = "latest") -> StarNet3WeightMetadata:
    """Retrieve metadata for a specified StarNet3 weight version."""
    v = STARNET3_LATEST_VERSION if version.lower() == "latest" else version
    if v not in STARNET3_WEIGHT_REGISTRY:
        available = list(STARNET3_WEIGHT_REGISTRY.keys())
        raise KeyError(f"Unknown StarNet3 weight version '{version}'. Available versions: {available}")
    return STARNET3_WEIGHT_REGISTRY[v]
