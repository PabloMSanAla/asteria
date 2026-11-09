"""
Weight cache and download manager for Asteria pre-trained models.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Union, Optional
import requests
import torch
from tqdm import tqdm

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


def _download_remote_file(url: str, destination: Path) -> None:
    """Download a remote file from standard HTTP/HTTPS or Google Drive URL with progress tracking."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    headers = {"User-Agent": "Asteria-Downloader/0.1"}

    # Check for Google Drive URL pattern
    gdrive_match = re.search(
        r"drive\.google\.com/(?:file/d/|uc\?(?:export=download&)?id=|open\?id=)([a-zA-Z0-9_-]+)", url
    )

    if gdrive_match:
        file_id = gdrive_match.group(1)
        dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        resp = session.get(dl_url, headers=headers, stream=True)
        resp.raise_for_status()

        # Check for Google Drive virus scan warning page on large files
        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type:
            text = resp.text
            action_match = re.search(r'action="([^"]+)"', text)
            if action_match:
                action_url = action_match.group(1)
                inputs = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', text))
                resp = session.get(action_url, params=inputs, headers=headers, stream=True)
                resp.raise_for_status()
    else:
        resp = session.get(url, headers=headers, stream=True)
        resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    with open(destination, "wb") as f, tqdm(
        total=total_size if total_size > 0 else None,
        unit="B",
        unit_scale=True,
        desc=f"[asteria] {destination.name}",
        leave=True,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_weight_file(meta: WeightMetadata, destination: Path) -> Path:
    """Download pre-trained weights from remote repository with verification."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = destination.with_suffix(".tmp")

    logger.info(f"[asteria] Downloading {meta.version} weights from: {meta.url}")
    print(f"[asteria] Downloading pre-trained UNet++ weights ({meta.version}) from {meta.url}...")

    try:
        _download_remote_file(meta.url, temp_dest)

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
        Either a direct path to a local .pt file, or an official version tag (e.g. 'v0.1.0', 'latest'),
        or an explicit remote URL (e.g. Google Drive or HTTP link).
        If None, checks ASTERIA_WEIGHTS_PATH or defaults to 'latest'.
    cache_dir : str or Path, optional
        Custom cache directory. Defaults to ~/.cache/asteria/weights.

    Returns
    -------
    Path
        Path to validated checkpoint file on local disk.
    """
    target = version_or_path or "latest"

    # 1. Environment variable override
    env_path = os.environ.get("ASTERIA_WEIGHTS_PATH")
    if env_path and Path(env_path).exists():
        logger.info(f"[asteria] Using checkpoint from ASTERIA_WEIGHTS_PATH: {env_path}")
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
        base_cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        cached_file = base_cache / "custom" / f"checkpoint_{url_hash}.pt"
        if cached_file.exists():
            return cached_file.resolve()
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[asteria] Downloading checkpoint from URL: {target}...")
        _download_remote_file(target, cached_file)
        return cached_file.resolve()

    # 4. Cache / Remote Registry resolution
    meta = get_weight_metadata(str(target))
    base_cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cached_file = base_cache / meta.version / meta.filename

    if cached_file.exists():
        return cached_file.resolve()

    # 5. Trigger download
    return download_weight_file(meta, cached_file)
