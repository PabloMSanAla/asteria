"""Pre-trained weights registry and management for Asteria and StarNet3."""

from asteria.weights.registry import (
    WEIGHT_REGISTRY,
    LATEST_VERSION,
    WeightMetadata,
    get_weight_metadata,
)
from asteria.weights.manager import (
    resolve_checkpoint_path,
    compute_sha256,
)
from asteria.weights.starnet3_registry import (
    STARNET3_WEIGHT_REGISTRY,
    STARNET3_LATEST_VERSION,
    StarNet3WeightMetadata,
    get_starnet3_weight_metadata,
)
from asteria.weights.starnet3_manager import (
    resolve_starnet3_checkpoint_path,
    download_starnet3_model,
    download_starnet3_weight_file,
    DEFAULT_STARNET3_CACHE_DIR,
)

__all__ = [
    "WEIGHT_REGISTRY",
    "LATEST_VERSION",
    "WeightMetadata",
    "get_weight_metadata",
    "resolve_checkpoint_path",
    "compute_sha256",
    "STARNET3_WEIGHT_REGISTRY",
    "STARNET3_LATEST_VERSION",
    "StarNet3WeightMetadata",
    "get_starnet3_weight_metadata",
    "resolve_starnet3_checkpoint_path",
    "download_starnet3_model",
    "download_starnet3_weight_file",
    "DEFAULT_STARNET3_CACHE_DIR",
]
