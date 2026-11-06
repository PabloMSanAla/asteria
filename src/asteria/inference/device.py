"""
Hardware device auto-detection and graceful fallback manager.
Supports NVIDIA CUDA, Apple Silicon MPS, and standard CPU.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional, Union, Tuple
import torch

logger = logging.getLogger(__name__)


def get_optimal_device(requested: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Select optimal execution device with automatic hardware detection and fallback.

    Fallback Hierarchy:
    1. NVIDIA CUDA (if available and drivers initialized)
    2. Apple Silicon MPS (Metal Performance Shaders on macOS)
    3. CPU (standard universal fallback)

    Parameters
    ----------
    requested : str or torch.device, optional
        Explicitly requested device ('cuda', 'mps', 'cpu', 'cuda:0', etc.).
        If the requested device is unavailable, logs a warning and falls back.

    Returns
    -------
    torch.device
        Verified active PyTorch device.
    """
    if requested is not None:
        req_str = str(requested).lower()
        if "cuda" in req_str:
            if torch.cuda.is_available():
                return torch.device(requested)
            logger.warning("[asteria] CUDA was requested but is unavailable. Falling back.")
        elif "mps" in req_str:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("[asteria] Apple Silicon MPS was requested but is unavailable. Falling back.")
        elif "cpu" in req_str:
            return torch.device("cpu")
        else:
            logger.warning(f"[asteria] Unrecognized device '{requested}'. Falling back.")

    # Automatic detection hierarchy
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> str:
    """Return a human-readable summary of the active accelerator."""
    dev_type = device.type
    if dev_type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"NVIDIA CUDA ({name}, {vram_gb:.1f} GB VRAM)"
    elif dev_type == "mps":
        return "Apple Silicon MPS (Metal Performance Shaders)"
    return "Universal CPU"


def _get_system_free_memory() -> int:
    """Query available system RAM in bytes across platforms with multi-tier fallback."""
    # Method 1: psutil (fastest and most accurate cross-platform)
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    # Method 2: Linux /proc/meminfo
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass

    # Method 3: macOS sysctl hw.memsize
    try:
        out = subprocess.check_output(["sysctl", "hw.memsize"], text=True)
        total = int(out.split(":")[1].strip())
        # Conservative assumption: ~40% of unified memory is available
        return int(total * 0.4)
    except Exception:
        pass

    # Method 4: POSIX os.sysconf
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except Exception:
        pass

    # Generic conservative fallback: 4 GB
    return int(4 * (1024 ** 3))


def get_free_memory(device: Optional[Union[str, torch.device]] = None) -> int:
    """
    Query available / free memory in bytes for the specified hardware device.
    Supports NVIDIA CUDA, Apple Silicon MPS (Metal), and CPU.

    Parameters
    ----------
    device : str or torch.device, optional
        Target hardware accelerator. Auto-selected if None.

    Returns
    -------
    int
        Free / available memory in bytes.
    """
    dev = get_optimal_device(device) if device is None or isinstance(device, str) else device

    # 1. NVIDIA CUDA
    if dev.type == "cuda":
        if torch.cuda.is_available():
            try:
                idx = dev.index if dev.index is not None else torch.cuda.current_device()
                free_bytes, _ = torch.cuda.mem_get_info(idx)
                return int(free_bytes)
            except Exception as e:
                logger.debug(f"[asteria] CUDA mem_get_info query failed: {e}")
                try:
                    idx = dev.index if dev.index is not None else torch.cuda.current_device()
                    total = torch.cuda.get_device_properties(idx).total_memory
                    allocated = torch.cuda.memory_allocated(idx)
                    return int(max(0, total - allocated))
                except Exception:
                    pass
        return int(2 * (1024 ** 3))  # Conservative fallback 2 GB

    # 2. Apple Silicon MPS
    if dev.type == "mps":
        sys_free = _get_system_free_memory()
        mps_free = None
        if hasattr(torch, "mps") and hasattr(torch.mps, "recommended_max_memory"):
            try:
                rec_max = torch.mps.recommended_max_memory()
                curr_alloc = (
                    torch.mps.current_allocated_memory()
                    if hasattr(torch.mps, "current_allocated_memory")
                    else 0
                )
                mps_free = max(0, rec_max - curr_alloc)
            except Exception as e:
                logger.debug(f"[asteria] MPS recommended_max_memory query failed: {e}")

        if mps_free is not None and sys_free is not None:
            return int(min(mps_free, sys_free))
        elif mps_free is not None:
            return int(mps_free)
        elif sys_free is not None:
            return int(sys_free)
        return int(4 * (1024 ** 3))

    # 3. CPU
    return _get_system_free_memory()


def get_adaptive_batch_size(
    device: Optional[Union[str, torch.device]] = None,
    tile_size: Union[int, Tuple[int, int]] = 512,
    memory_fraction: float = 0.2,
    bytes_per_tile: Optional[int] = None,
    min_batch_size: int = 1,
    max_batch_size: Optional[int] = None,
) -> int:
    """
    Calculate an optimal inference batch size dynamically adapted to the
    available free memory of the target accelerator (CUDA, MPS, or CPU).

    Parameters
    ----------
    device : str or torch.device, optional
        Target hardware accelerator.
    tile_size : int or tuple of int, default 512
        Tile dimensions (H, W) or square side length.
    memory_fraction : float, default 0.2
        Fraction of currently free memory to budget for the batch (e.g. 0.2 = 20%).
    bytes_per_tile : int, optional
        Estimated peak memory footprint per tile in bytes. If None, estimated from tile_size.
    min_batch_size : int, default 1
        Minimum allowed batch size.
    max_batch_size : int, optional
        Maximum allowed batch size cap (defaults to 1 on CPU to prevent cache blowout,
        and 16 on GPU accelerators like CUDA and MPS).

    Returns
    -------
    int
        Calculated batch size >= min_batch_size.
    """
    dev = get_optimal_device(device) if device is None or isinstance(device, str) else device

    if max_batch_size is None:
        # On CPU, batch_size=1 gives the highest throughput and lowest latency per tile
        # by keeping intermediate activations within L2/L3 cache, avoiding memory bus contention.
        # On GPUs (CUDA/MPS), larger batches exploit massive ALU parallelism.
        max_batch_size = 1 if dev.type == "cpu" else 16

    if isinstance(tile_size, (int, float)):
        th, tw = int(tile_size), int(tile_size)
    else:
        th, tw = int(tile_size[0]), int(tile_size[1])

    frac = max(0.01, min(float(memory_fraction), 0.90))
    free_mem = get_free_memory(dev)
    budget = free_mem * frac

    if bytes_per_tile is None:
        # Peak activation footprint per tile:
        # 4 bytes per float32 * ~200 peak intermediate feature channels across stages
        bytes_per_tile = int(th * tw * 4 * 200)

    calculated = int(budget // max(1, bytes_per_tile))
    batch_size = max(min_batch_size, min(calculated, max_batch_size))
    logger.debug(
        f"[asteria] Adaptive batch size: free_mem={free_mem / 1e6:.1f} MB, "
        f"budget={budget / 1e6:.1f} MB ({frac * 100:.1f}%), "
        f"tile={th}x{tw} (~{bytes_per_tile / 1e6:.1f} MB/tile) -> batch_size={batch_size} on {dev.type}"
    )
    return batch_size


def empty_device_cache(device: Optional[Union[str, torch.device]] = None) -> None:
    """Safely release cached allocator memory on CUDA or MPS devices."""
    if device is None:
        device = get_optimal_device()
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

