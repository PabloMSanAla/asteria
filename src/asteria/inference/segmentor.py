"""
Asteria UNet++ Astronomical Inference Engine and Deblender.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from asteria.models.unetpp import UNetPlusPlus
from asteria.inference.device import (
    get_optimal_device,
    get_device_info,
    get_adaptive_batch_size,
    empty_device_cache,
)
from asteria.io.reader import load_image, AstroImage
from asteria.stretch.leakysb import stretch_image
from asteria.weights.manager import resolve_checkpoint_path

logger = logging.getLogger(__name__)


def _build_tile_window(
    size: int,
    overlap: int,
    is_top: bool,
    is_bottom: bool,
    is_left: bool,
    is_right: bool,
) -> np.ndarray:
    """
    Construct a 2D cosine blending window for sliding tile inference.
    Edges touching physical image boundaries maintain weight 1.0 to prevent border fading.
    """
    if overlap <= 0:
        return np.ones((size, size), dtype=np.float32)

    ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, overlap, dtype=np.float32)))

    wy = np.ones(size, dtype=np.float32)
    if not is_top:
        wy[:overlap] = ramp
    if not is_bottom:
        wy[-overlap:] = ramp[::-1]

    wx = np.ones(size, dtype=np.float32)
    if not is_left:
        wx[:overlap] = ramp
    if not is_right:
        wx[-overlap:] = ramp[::-1]

    return np.outer(wy, wx)


def _resolve_stretch_kwargs(
    astro: AstroImage,
    stretch_type: str,
    zero_points: Optional[float] = None,
    pixel_scale: Optional[float] = None,
    mmin: Optional[float] = None,
    mmax: Optional[float] = None,
    leak_slope: Optional[float] = None,
    negative_slope: Optional[float] = None,
    positive_slope: Optional[float] = None,
    sky: Optional[float] = None,
    **stretch_kwargs,
) -> Dict[str, Any]:
    """
    Resolve and validate astronomical surface brightness stretch parameters.

    Handles parameter aliases (e.g. zeropoint, zero_point, pixelscale) and defaults
    to image metadata (e.g. Roman L2/L3 zero point 20.472 for MJy/sr) when omitted.
    """
    kwargs = dict(stretch_kwargs)

    # Resolve sky subtraction: explicit argument > alias in kwargs > astro.sky > 0.0
    sky_val = sky
    if sky_val is None:
        for k in ("sky", "sky_val", "sky_level", "background"):
            if k in kwargs and kwargs[k] is not None:
                sky_val = kwargs.pop(k)
                break
    if sky_val is None:
        sky_val = getattr(astro, "sky", 0.0)
    kwargs["sky"] = float(sky_val) if sky_val is not None else 0.0

    if stretch_type.lower() in ("leakysb", "leaky_sb"):
        # Resolve zero point: explicit argument > alias in kwargs > astro.zero_point > 31.4
        zp = zero_points
        if zp is None:
            for k in ("zero_points", "zero_point", "zeropoint", "zp"):
                if k in kwargs and kwargs[k] is not None:
                    zp = kwargs.pop(k)
                    break
        if zp is None:
            zp = getattr(astro, "zero_point", 31.4)
        kwargs["zero_points"] = float(zp)

        # Resolve pixel scale: explicit argument > alias in kwargs > astro.pixel_scale > (1.0 if level in (2, 3))
        ps = pixel_scale
        if ps is None:
            for k in ("pixel_scale", "pixelscale", "ps"):
                if k in kwargs and kwargs[k] is not None:
                    ps = kwargs.pop(k)
                    break
        if ps is None:
            if getattr(astro, "pixel_scale", None) is not None:
                ps = astro.pixel_scale
            elif getattr(astro, "level", None) in (2, 3):
                ps = 1.0
        if ps is not None:
            kwargs["pixel_scale"] = float(ps)

        # Resolve magnitude cutoffs
        if mmin is not None:
            kwargs["mmin"] = float(mmin)
        elif "mmin" in kwargs and kwargs["mmin"] is not None:
            kwargs["mmin"] = float(kwargs["mmin"])

        if mmax is not None:
            kwargs["mmax"] = float(mmax)
        elif "mmax" in kwargs and kwargs["mmax"] is not None:
            kwargs["mmax"] = float(kwargs["mmax"])

        # Resolve positive slope / leak slope
        pos_s = positive_slope if positive_slope is not None else leak_slope
        if pos_s is None:
            for k in ("positive_slope", "leak_slope", "leakslope"):
                if k in kwargs and kwargs[k] is not None:
                    pos_s = kwargs.pop(k)
                    break
        if pos_s is not None:
            kwargs["positive_slope"] = float(pos_s)

        # Resolve negative slope
        neg_s = negative_slope
        if neg_s is None:
            for k in ("negative_slope",):
                if k in kwargs and kwargs[k] is not None:
                    neg_s = kwargs.pop(k)
                    break
        if neg_s is not None:
            kwargs["negative_slope"] = float(neg_s)

    return kwargs


class AsteriaSegmentor:
    """
    High-performance astronomical inference segmentor powered by boundary-aware UNet++.

    Executes promptless deep learning segmentation across astronomical exposures,
    yielding continuous 3-class probability maps (Sky, Source Interior, Deblending Edge).

    Parameters
    ----------
    model : UNetPlusPlus
        Instantiated UNet++ neural network.
    device : torch.device or str, optional
        Target hardware accelerator ('cuda', 'mps', 'cpu'). Auto-selected if None.
    """

    _build_tile_window = staticmethod(_build_tile_window)

    def __init__(
        self,
        model: UNetPlusPlus,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = get_optimal_device(device)
        self.model = model.to(self.device)
        if self.device.type in ("cpu", "cuda"):
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        version_or_path: Union[str, Path] = "latest",
        device: Optional[Union[str, torch.device]] = None,
    ) -> "AsteriaSegmentor":
        """
        Load an Asteria segmentor from pre-trained weights.

        Parameters
        ----------
        version_or_path : str or Path, default 'latest'
            Version string (e.g. 'v0.1.0', 'latest') or direct path to a .pt checkpoint.
        device : str or torch.device, optional
            Hardware device. Auto-selects CUDA -> MPS -> CPU if None.

        Returns
        -------
        AsteriaSegmentor
        """
        ckpt_path = resolve_checkpoint_path(version_or_path)
        dev = get_optimal_device(device)

        ckpt = torch.load(ckpt_path, map_location=dev)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        in_channels = ckpt.get("in_channels", 3) if isinstance(ckpt, dict) else 3
        num_classes = ckpt.get("num_classes", 3) if isinstance(ckpt, dict) else 3
        features = tuple(ckpt.get("features", (32, 64, 128, 256, 512))) if isinstance(ckpt, dict) else (32, 64, 128, 256, 512)
        deep_supervision = ckpt.get("deep_supervision", False) if isinstance(ckpt, dict) else False
        norm_type = ckpt.get("norm_type", "batch") if isinstance(ckpt, dict) else "batch"

        # Sanitize keys: strip DDP 'module.' or compiled '_orig_mod.' prefixes
        cleaned_sd = {}
        for k, v in state_dict.items():
            k_clean = k
            if k_clean.startswith("module."):
                k_clean = k_clean[len("module."):]
            if k_clean.startswith("_orig_mod."):
                k_clean = k_clean[len("_orig_mod."):]
            # Remap legacy VGGBlock keys if present
            k_clean = k_clean.replace(".conv1.", ".block.0.")
            k_clean = k_clean.replace(".bn1.", ".block.1.")
            k_clean = k_clean.replace(".conv2.", ".block.3.")
            k_clean = k_clean.replace(".bn2.", ".block.4.")
            cleaned_sd[k_clean] = v

        model = UNetPlusPlus(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features,
            deep_supervision=deep_supervision,
            norm_type=norm_type,
        )
        model.load_state_dict(cleaned_sd)
        logger.info(f"[asteria] Loaded weights from: {ckpt_path} onto {get_device_info(dev)}")

        return cls(model=model, device=dev)

    def predict(
        self,
        image: Union[np.ndarray, AstroImage, str, Path],
        stretch_type: str = "leakysb",
        sky: Optional[float] = None,
        zero_points: Optional[float] = None,
        pixel_scale: Optional[float] = None,
        mmin: Optional[float] = None,
        mmax: Optional[float] = None,
        leak_slope: Optional[float] = None,
        negative_slope: Optional[float] = None,
        positive_slope: Optional[float] = None,
        **stretch_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run single-forward-pass inference on an astronomical image patch.

        Parameters
        ----------
        image : np.ndarray, AstroImage, str, or Path
            Input astronomical image.
        stretch_type : str, default 'leakysb'
            Intensity transformation algorithm ('leakysb', 'asinh', 'linear').
        sky : float, optional
            Background sky value to subtract before stretching (default 0.0; automatically
            estimated via sigma_clipped_stats median for Roman Level 2 images if None).
        zero_points : float, optional
            Photometric zero point for surface brightness transformation
            (defaults to 20.472 for Roman L2/L3 MJy/sr images, 31.4 for FITS).
            Aliases: zeropoint, zero_point, zp.
        pixel_scale : float, optional
            Detector plate scale in arcsec/pixel (default: 0.11 for Roman WFI).
            Alias: pixelscale.
        mmin : float, optional
            Bright-end surface brightness magnitude cutoff in mag/arcsec^2 (default: 13.0).
        mmax : float, optional
            Faint-end surface brightness magnitude cutoff in mag/arcsec^2 (default: 32.0).
        leak_slope : float, optional
            Slope of linear response for negative/sub-background flux values (default: 0.05).
            Alias: leakslope.
        negative_slope : float, optional
            Negative/faint slope for leakySB (default: 0.1).
        positive_slope : float, optional
            Positive/bright slope for leakySB (default: 0.05).
        **stretch_kwargs
            Additional arguments passed to the stretch function.

        Returns
        -------
        pred_mask : np.ndarray
            Discrete 2D integer mask (H, W) where:
            0 = Sky Background, 1 = Source Interior, 2 = Deblending Boundary.
        probs : np.ndarray
            Softmax probability map of shape (3, H, W).
        stretched : np.ndarray
            Normalized 2D image [0.0, 1.0] after astronomical stretch.
        """
        astro = load_image(image, sky=sky)
        raw_data = astro.data
        h, w = raw_data.shape

        resolved_stretch = _resolve_stretch_kwargs(
            astro=astro,
            stretch_type=stretch_type,
            zero_points=zero_points,
            pixel_scale=pixel_scale,
            mmin=mmin,
            mmax=mmax,
            leak_slope=leak_slope,
            negative_slope=negative_slope,
            positive_slope=positive_slope,
            sky=sky,
            **stretch_kwargs,
        )

        # Photometric stretch
        stretched = stretch_image(raw_data, stretch_type=stretch_type, **resolved_stretch)

        # Pad to multiple of 16
        pad_h = (16 - (h % 16)) % 16
        pad_w = (16 - (w % 16)) % 16

        norm_3ch = np.repeat(stretched[np.newaxis, :, :], 3, axis=0)  # (3, H, W)
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(norm_3ch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        else:
            padded = norm_3ch

        input_t = torch.from_numpy(padded).unsqueeze(0).to(self.device)
        if self.device.type in ("cpu", "cuda"):
            input_t = input_t.to(memory_format=torch.channels_last)

        with torch.inference_mode():
            logits = self.model(input_t)
            if isinstance(logits, list):
                logits = logits[-1]
            probs_t = F.softmax(logits, dim=1).squeeze(0)  # (3, H_pad, W_pad)

        probs = probs_t[:, :h, :w].cpu().numpy().astype(np.float32)
        pred_mask = np.argmax(probs, axis=0).astype(np.uint8)

        return pred_mask, probs, stretched

    def predict_dense(
        self,
        image: Union[np.ndarray, AstroImage, str, Path],
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: Optional[int] = None,
        memory_fraction: float = 0.2,
        stretch_type: str = "leakysb",
        verbose: bool = False,
        sky: Optional[float] = None,
        zero_points: Optional[float] = None,
        pixel_scale: Optional[float] = None,
        mmin: Optional[float] = None,
        mmax: Optional[float] = None,
        leak_slope: Optional[float] = None,
        negative_slope: Optional[float] = None,
        positive_slope: Optional[float] = None,
        **stretch_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run dense 3-class segmentation inference over full astronomical exposures.
        Seamlessly processes arbitrary image sizes (e.g. 4088x4088 Roman WFI fields)
        via sliding-window tiling with edge-aware cosine window blending.
        Dynamically calculates hardware-adaptive batch size based on available free memory.

        Parameters
        ----------
        image : np.ndarray, AstroImage, str, or Path
            Input astronomical image.
        tile_size : int, default 512
            Tile width/height in pixels. Set to 0 to bypass tiling. Must be divisible by 16.
        overlap : int, default 64
            Overlap in pixels between adjacent tiles.
        batch_size : int, optional
            Number of tiles per GPU/accelerator forward batch. If None (default),
            adaptively calculated based on currently available free device memory.
        memory_fraction : float, default 0.2
            Fraction of available free memory budgeted for the batch forward pass.
        stretch_type : str, default 'leakysb'
            Intensity transformation ('leakysb', 'asinh', 'linear').
        verbose : bool, default False
            If True, prints execution timing and tile progress.
        sky : float, optional
            Background sky value to subtract before stretching (default 0.0; automatically
            estimated via sigma_clipped_stats median for Roman Level 2 images if None).
        zero_points : float, optional
            Photometric zero point for surface brightness transformation
            (defaults to 20.472 for Roman L2/L3 MJy/sr images, 31.4 for FITS).
            Aliases: zeropoint, zero_point, zp.
        pixel_scale : float, optional
            Detector plate scale in arcsec/pixel (default: 0.11 for Roman WFI).
            Alias: pixelscale.
        mmin : float, optional
            Bright-end surface brightness magnitude cutoff in mag/arcsec^2 (default: 13.0).
        mmax : float, optional
            Faint-end surface brightness magnitude cutoff in mag/arcsec^2 (default: 32.0).
        leak_slope : float, optional
            Slope of linear response for negative/sub-background flux values (default: 0.05).
            Alias: leakslope.
        negative_slope : float, optional
            Negative slope for leakySB (default: 0.1).
        positive_slope : float, optional
            Positive slope for leakySB (default: 0.05).
        **stretch_kwargs
            Additional arguments passed to the stretch function.

        Returns
        -------
        pred_mask : np.ndarray
            Discrete 2D integer mask (H, W): 0 = Sky, 1 = Core, 2 = Deblending Edge.
        probs : np.ndarray
            Softmax probability map of shape (3, H, W).
        stretched : np.ndarray
            Normalized 2D image [0.0, 1.0] after astronomical stretch.
        """
        astro = load_image(image, sky=sky)
        raw_data = astro.data
        h, w = raw_data.shape

        resolved_stretch = _resolve_stretch_kwargs(
            astro=astro,
            stretch_type=stretch_type,
            zero_points=zero_points,
            pixel_scale=pixel_scale,
            mmin=mmin,
            mmax=mmax,
            leak_slope=leak_slope,
            negative_slope=negative_slope,
            positive_slope=positive_slope,
            sky=sky,
            **stretch_kwargs,
        )

        t0 = time.time()
        stretched = stretch_image(raw_data, stretch_type=stretch_type, **resolved_stretch)
        norm_3ch = np.repeat(stretched[np.newaxis, :, :], 3, axis=0)

        # Case 1: Direct full-frame inference
        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            pad_h = (16 - (h % 16)) % 16
            pad_w = (16 - (w % 16)) % 16

            if pad_h > 0 or pad_w > 0:
                padded = np.pad(norm_3ch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            else:
                padded = norm_3ch

            input_t = torch.from_numpy(padded).unsqueeze(0).to(self.device)
            if self.device.type in ("cpu", "cuda"):
                input_t = input_t.to(memory_format=torch.channels_last)

            with torch.inference_mode():
                logits = self.model(input_t)
                if isinstance(logits, list):
                    logits = logits[-1]
                probs = F.softmax(logits, dim=1).squeeze(0)[:, :h, :w].cpu().numpy().astype(np.float32)

            pred_mask = np.argmax(probs, axis=0).astype(np.uint8)
            elapsed = time.time() - t0
            if verbose:
                print(f"[asteria] Direct UNet++ inference on {w}x{h} completed in {elapsed:.2f}s on {self.device}.")
            return pred_mask, probs, stretched

        # Case 2: Tiled sliding-window inference with edge-aware cosine blending
        tile_size = (tile_size // 16) * 16
        overlap = min(overlap, tile_size // 4)

        if batch_size is None:
            eff_batch_size = get_adaptive_batch_size(
                device=self.device,
                tile_size=tile_size,
                memory_fraction=memory_fraction,
            )
        else:
            eff_batch_size = max(1, int(batch_size))

        def _get_coords(total_len: int, size: int, ov: int) -> List[int]:
            if total_len <= size:
                return [0]
            stride = size - ov
            coords = []
            curr = 0
            while curr + size < total_len:
                coords.append(curr)
                curr += stride
            coords.append(total_len - size)
            return sorted(list(set(coords)))

        y_coords = _get_coords(h, tile_size, overlap)
        x_coords = _get_coords(w, tile_size, overlap)
        num_tiles = len(y_coords) * len(x_coords)

        if verbose:
            print(
                f"[asteria] Starting tiled UNet++ inference: {w}x{h} across {len(x_coords)}x{len(y_coords)} = {num_tiles} tiles "
                f"(tile_size={tile_size}, overlap={overlap}, batch_size={eff_batch_size}) on {self.device}..."
            )

        prob_canvas = np.zeros((3, h, w), dtype=np.float32)
        weight_canvas = np.zeros((h, w), dtype=np.float32)

        # Cache blending windows across standard edge variations
        window_cache: Dict[Tuple[int, int, bool, bool, bool, bool], np.ndarray] = {}

        def _get_window(sz: int, ov: int, top: bool, btm: bool, left: bool, right: bool) -> np.ndarray:
            key = (sz, ov, top, btm, left, right)
            if key not in window_cache:
                window_cache[key] = self._build_tile_window(sz, ov, top, btm, left, right)
            return window_cache[key]

        tile_list: List[Tuple[int, int]] = []
        for y in y_coords:
            for x in x_coords:
                tile_list.append((y, x))

        def _run_forward_with_retry(b_crops: List[np.ndarray]) -> np.ndarray:
            nonlocal eff_batch_size
            while True:
                try:
                    b_t = torch.from_numpy(np.stack(b_crops)).to(self.device)
                    if self.device.type in ("cpu", "cuda"):
                        b_t = b_t.to(memory_format=torch.channels_last)
                    with torch.inference_mode():
                        logits = self.model(b_t)
                        if isinstance(logits, list):
                            logits = logits[-1]
                        b_probs = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
                    del b_t, logits
                    return b_probs
                except (torch.OutOfMemoryError, RuntimeError) as err:
                    empty_device_cache(self.device)
                    is_oom = isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()
                    if is_oom and len(b_crops) > 1:
                        new_bs = max(1, len(b_crops) // 2)
                        logger.warning(
                            f"[asteria] Out of memory with batch size {len(b_crops)}. "
                            f"Adapting batch size to {new_bs} and retrying."
                        )
                        eff_batch_size = new_bs
                        mid = len(b_crops) // 2
                        p1 = _run_forward_with_retry(b_crops[:mid])
                        p2 = _run_forward_with_retry(b_crops[mid:])
                        return np.concatenate([p1, p2], axis=0)
                    raise err

        i = 0
        batch_idx = 0
        total_est = (len(tile_list) + eff_batch_size - 1) // eff_batch_size

        pbar = tqdm(
            total=num_tiles,
            desc="[asteria] UNet++ Tiling",
            unit="tile",
            disable=not verbose,
            leave=True,
        )

        while i < len(tile_list):
            curr_bs = min(eff_batch_size, len(tile_list) - i)
            batch_coords = tile_list[i : i + curr_bs]
            batch_crops = [norm_3ch[:, y : y + tile_size, x : x + tile_size] for y, x in batch_coords]

            batch_probs = _run_forward_with_retry(batch_crops)

            for b_idx, (y, x) in enumerate(batch_coords):
                is_top = (y == 0)
                is_bottom = (y == y_coords[-1])
                is_left = (x == 0)
                is_right = (x == x_coords[-1])
                w2d = _get_window(tile_size, overlap, is_top, is_bottom, is_left, is_right)

                prob_canvas[:, y : y + tile_size, x : x + tile_size] += batch_probs[b_idx] * w2d
                weight_canvas[y : y + tile_size, x : x + tile_size] += w2d

            i += len(batch_coords)
            batch_idx += 1

            pbar.update(len(batch_coords))
            pbar.set_postfix({"batch": f"{batch_idx}/{total_est}", "bs": len(batch_coords)})

        pbar.close()

        probs = prob_canvas / np.maximum(weight_canvas, 1e-7)
        pred_mask = np.argmax(probs, axis=0).astype(np.uint8)

        elapsed = time.time() - t0
        if verbose:
            print(f"[asteria] Tiled UNet++ inference completed in {elapsed:.2f}s ({num_tiles} tiles, {elapsed / num_tiles * 1000:.1f}ms/tile).")

        return pred_mask, probs, stretched

