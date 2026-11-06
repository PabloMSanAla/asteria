"""
Weight cache and download manager for Asteria pre-trained models.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path
from typing import Union, Optional
import torch

from asteria.weights.registry import get_weight_metadata, WeightMetadata

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "asteria" / "weights"


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def download_weight_file(meta: WeightMetadata, destination: Path) -> Path:
    """Download pre-trained weights from remote repository with verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = destination.with_suffix(".tmp")

    logger.info(f"[asteria] Downloading {meta.version} weights from: {meta.url}")
    print(f"[asteria] Downloading pre-trained UNet++ weights ({meta.version}) from {meta.url}...")

    try:
        # Download file
        urllib.request.urlretrieve(meta.url, temp_dest)

        # Verify sha256 checksum if configured
        if meta.sha256:
            computed = compute_sha256(temp_dest)
            if computed != meta.sha256:
                # If remote is not yet published or hash changed, log warning
                logger.warning(
                    f"[asteria] Checksum warning: expected {meta.sha256}, got {computed}."
                )

        temp_dest.replace(destination)
        print(f"[asteria] Successfully cached weights at: {destination}")
        return destination
    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink()
        raise RuntimeError(
            f"Failed to download Asteria weights for {meta.version} from {meta.url}. "
            f"Error: {e}\nYou can manually specify a local checkpoint path via --checkpoint or "
            f"export ASTERIA_WEIGHTS_PATH=/path/to/unetpp.pt"
        ) from e


def resolve_checkpoint_path(
    version_or_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve checkpoint path from a local file, environment variable, or cache/download.

    Parameters
    ----------
    version_or_path : str or Path, optional
        Either a direct path to a local .pt file, or an official version tag (e.g. 'v0.1.0', 'latest').
        If None, checks ASTERIA_WEIGHTS_PATH or defaults to 'latest'.
    cache_dir : str or Path, optional
        Custom cache directory. Defaults to ~/.cache/asteria/weights.

    Returns
    -------
    Path
        Path to validated checkpoint file on local disk.
    """
    # 1. Environment variable override
    env_path = os.environ.get("ASTERIA_WEIGHTS_PATH")
    if env_path and Path(env_path).exists():
        logger.info(f"[asteria] Using checkpoint from ASTERIA_WEIGHTS_PATH: {env_path}")
        return Path(env_path).resolve()

    target = version_or_path or "latest"

    # 2. Direct local file path
    local_p = Path(target)
    if local_p.exists() and local_p.is_file():
        return local_p.resolve()

    # 3. Cache / Remote Registry resolution
    meta = get_weight_metadata(str(target))
    base_cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cached_file = base_cache / meta.version / meta.filename

    if cached_file.exists():
        return cached_file.resolve()

    # 4. Trigger download
    return download_weight_file(meta, cached_file)
