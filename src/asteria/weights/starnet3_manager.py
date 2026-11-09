"""
Weight cache and download manager for StarNet3 pre-trained models.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional, Union

from asteria.weights.starnet3_registry import (
    get_starnet3_weight_metadata,
    StarNet3WeightMetadata,
    STARNET3_WEIGHT_REGISTRY,
)
from asteria.weights.manager import _download_remote_file, compute_sha256

logger = logging.getLogger(__name__)

DEFAULT_STARNET3_CACHE_DIR = Path.home() / ".cache" / "starnet3" / "weights"


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def download_starnet3_weight_file(meta: StarNet3WeightMetadata, destination: Path) -> Path:
    """Download pre-trained StarNet3 weights from remote repository with verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = destination.with_suffix(".tmp")

    logger.info(f"[starnet3] Downloading {meta.version} weights from: {meta.url}")
    print(f"[starnet3] Downloading pre-trained StarNet3 weights ({meta.version}) from {meta.url}...")

    try:
        _download_remote_file(meta.url, temp_dest)

        if meta.sha256:
            computed = compute_sha256(temp_dest)
            if computed != meta.sha256:
                logger.warning(
                    f"[starnet3] Checksum warning: expected {meta.sha256}, got {computed}."
                )

        temp_dest.replace(destination)
        print(f"[starnet3] Successfully cached weights at: {destination}")
        return destination
    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink()
        raise RuntimeError(
            f"Failed to download StarNet3 weights for {meta.version} from {meta.url}. "
            f"Error: {e}\nYou can manually specify a local checkpoint path via --checkpoint or "
            f"export STARNET3_WEIGHTS_PATH=/path/to/starnet3.pt"
        ) from e


def resolve_starnet3_checkpoint_path(
    version_or_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve StarNet3 checkpoint path from local file, environment variable, or cache/download.

    Parameters
    ----------
    version_or_path : str or Path, optional
        Either a direct path to a local .pt file, or an official version tag (e.g. 'v0.1.0', 'latest'),
        or an explicit remote URL (e.g. Google Drive or HTTP link).
        If None, checks STARNET3_WEIGHTS_PATH or defaults to 'latest'.
    cache_dir : str or Path, optional
        Custom cache directory. Defaults to ~/.cache/starnet3/weights.

    Returns
    -------
    Path
        Path to validated checkpoint file on local disk.
    """
    target = version_or_path or "latest"

    # 1. Environment variable override
    env_path = os.environ.get("STARNET3_WEIGHTS_PATH")
    if env_path and Path(env_path).exists():
        logger.info(f"[starnet3] Using checkpoint from STARNET3_WEIGHTS_PATH: {env_path}")
        return Path(env_path).resolve()

    # 2. Direct local file path
    local_p = Path(target)
    if local_p.exists() and local_p.is_file():
        return local_p.resolve()

    # 3. Direct URL override (e.g. Google Drive or custom HTTP link)
    if isinstance(target, str) and (
        target.startswith("http://") or target.startswith("https://") or "drive.google.com" in target
    ):
        url_hash = hashlib.sha256(target.encode()).hexdigest()[:12]
        base_cache = Path(cache_dir) if cache_dir else DEFAULT_STARNET3_CACHE_DIR
        cached_file = base_cache / "custom" / f"starnet3_{url_hash}.pt"
        if cached_file.exists():
            return cached_file.resolve()
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[starnet3] Downloading checkpoint from URL: {target}...")
        _download_remote_file(target, cached_file)
        return cached_file.resolve()

    # 4. Cache / Remote Registry resolution
    meta = get_starnet3_weight_metadata(str(target))
    base_cache = Path(cache_dir) if cache_dir else DEFAULT_STARNET3_CACHE_DIR
    cached_file = base_cache / meta.version / meta.filename

    if cached_file.exists():
        return cached_file.resolve()

    # 5. Trigger download
    return download_starnet3_weight_file(meta, cached_file)


def download_starnet3_model(
    version: str = "latest",
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Download a StarNet3 model checkpoint automatically to the cache directory.

    Parameters
    ----------
    version : str, default 'latest'
        Version tag to download.
    cache_dir : str or Path, optional
        Cache directory. Defaults to ~/.cache/starnet3/weights.

    Returns
    -------
    Path
        Path to downloaded checkpoint on local disk.
    """
    meta = get_starnet3_weight_metadata(version)
    base_cache = Path(cache_dir) if cache_dir else DEFAULT_STARNET3_CACHE_DIR
    target_dest = base_cache / meta.version / meta.filename
    if target_dest.exists():
        logger.info(f"[starnet3] Checkpoint already cached at {target_dest}")
        return target_dest.resolve()
    return download_starnet3_weight_file(meta, target_dest)
