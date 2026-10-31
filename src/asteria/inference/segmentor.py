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

from asteria.models.unetpp import UNetPlusPlus
from asteria.inference.device import get_optimal_device, get_device_info
from asteria.inference.watershed import deblend_sources, SegmentationResult
from asteria.io.reader import load_image, AstroImage
from asteria.stretch.leakysb import stretch_image
from asteria.weights.manager import resolve_checkpoint_path

logger = logging.getLogger(__name__)


def _make_2d_cosine_window(tile_size: int) -> np.ndarray:
    """Create a 2D Hann / cosine window for smooth tile blending."""
    w1d = np.hanning(tile_size).astype(np.float32)
    # Avoid zero weight on extreme borders
    w1d = np.clip(w1d, 0.05, 1.0)
    return np.outer(w1d, w1d)


class AsteriaSegmentor:
    """
    High-performance astronomical inference segmentor powered by boundary-aware UNet++.

    Executes promptless deep learning segmentation across astronomical exposures,
    yielding continuous 3-class probability maps (Sky, Source Interior, Deblending Edge)
    and deblended discrete object catalogs with marker-controlled watershed.

    Parameters
    ----------
    model : UNetPlusPlus
        Instantiated UNet++ neural network.
    device : torch.device or str, optional
        Target hardware accelerator ('cuda', 'mps', 'cpu'). Auto-selected if None.
    """

    def __init__(
        self,
        model: UNetPlusPlus,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = get_optimal_device(device)
        self.model = model.to(self.device)
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
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        in_channels = ckpt.get("in_channels", 3) if isinstance(ckpt, dict) else 3
        num_classes = ckpt.get("num_classes", 3) if isinstance(ckpt, dict) else 3
        features = tuple(ckpt.get("features", (32, 64, 128, 256, 512))) if isinstance(ckpt, dict) else (32, 64, 128, 256, 512)
        deep_supervision = ckpt.get("deep_supervision", False) if isinstance(ckpt, dict) else False

        model = UNetPlusPlus(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features,
            deep_supervision=deep_supervision,
        )
        model.load_state_dict(state_dict)
        logger.info(f"[asteria] Loaded weights from: {ckpt_path} onto {get_device_info(dev)}")

        return cls(model=model, device=dev)

    def predict(
        self,
        image: Union[np.ndarray, AstroImage, str, Path],
        stretch_type: str = "leakysb",
        **stretch_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run single-forward-pass inference on an astronomical image patch.

        Parameters
        ----------
        image : np.ndarray, AstroImage, str, or Path
            Input astronomical image.
        stretch_type : str, default 'leakysb'
            Intensity transformation.

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
        astro = load_image(image)
        raw_data = astro.data
        h, w = raw_data.shape

        # Photometric stretch
        stretched = stretch_image(raw_data, stretch_type=stretch_type, **stretch_kwargs)

        # Pad to multiple of 16
        pad_h = (16 - (h % 16)) % 16
        pad_w = (16 - (w % 16)) % 16

        norm_3ch = np.repeat(stretched[np.newaxis, :, :], 3, axis=0)  # (3, H, W)
        if pad_h > 0 or pad_w > 0:
            padded = np.pad(norm_3ch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        else:
            padded = norm_3ch

        input_t = torch.from_numpy(padded).unsqueeze(0).to(self.device)

        with torch.no_grad():
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
        batch_size: int = 2,
        stretch_type: str = "leakysb",
        verbose: bool = False,
        **stretch_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run dense 3-class segmentation inference over full astronomical exposures.
        Seamlessly processes arbitrary image sizes (e.g. 4088x4088 Roman WFI fields)
        via sliding-window tiling with 2D cosine window blending and bounded memory.

        Parameters
        ----------
        image : np.ndarray, AstroImage, str, or Path
            Input astronomical image.
        tile_size : int, default 512
            Tile width/height in pixels. Set to 0 to bypass tiling.
        overlap : int, default 64
            Overlap in pixels between adjacent tiles.
        batch_size : int, default 2
            Number of tiles per GPU forward batch.
        stretch_type : str, default 'leakysb'
            Intensity transformation.
        verbose : bool, default False

        Returns
        -------
        pred_mask : np.ndarray
            Discrete 2D integer mask (H, W): 0 = Sky, 1 = Core, 2 = Deblending Edge.
        probs : np.ndarray
            Softmax probability map of shape (3, H, W).
        stretched : np.ndarray
            Normalized 2D image [0.0, 1.0] after astronomical stretch.
        """
        astro = load_image(image)
        raw_data = astro.data
        h, w = raw_data.shape

        # Small image fallback
        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            return self.predict(astro, stretch_type=stretch_type, **stretch_kwargs)

        t0 = time.time()
        stretched = stretch_image(raw_data, stretch_type=stretch_type, **stretch_kwargs)
        norm_3ch = np.repeat(stretched[np.newaxis, :, :], 3, axis=0)

        # Plan tile grid
        stride = tile_size - overlap
        y_starts = list(range(0, h - tile_size + 1, stride))
        if len(y_starts) == 0 or y_starts[-1] + tile_size < h:
            y_starts.append(max(0, h - tile_size))

        x_starts = list(range(0, w - tile_size + 1, stride))
        if len(x_starts) == 0 or x_starts[-1] + tile_size < w:
            x_starts.append(max(0, w - tile_size))

        tile_coords: List[Tuple[int, int]] = [(ys, xs) for ys in y_starts for xs in x_starts]
        num_tiles = len(tile_coords)

        if verbose:
            print(
                f"[asteria] Tiled UNet++ inference: {w}x{h} across {len(y_starts)}x{len(x_starts)} = "
                f"{num_tiles} tiles (tile_size={tile_size}, overlap={overlap}, batch_size={batch_size}) on {self.device}..."
            )

        prob_accum = np.zeros((3, h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)
        cos_weight = _make_2d_cosine_window(tile_size)

        for b_idx in range(0, num_tiles, batch_size):
            b_coords = tile_coords[b_idx : b_idx + batch_size]
            b_crops = [norm_3ch[:, ys : ys + tile_size, xs : xs + tile_size] for ys, xs in b_coords]

            batch_t = torch.from_numpy(np.stack(b_crops)).to(self.device)

            with torch.no_grad():
                logits = self.model(batch_t)
                if isinstance(logits, list):
                    logits = logits[-1]
                b_probs = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

            for crop_p, (ys, xs) in zip(b_probs, b_coords):
                prob_accum[:, ys : ys + tile_size, xs : xs + tile_size] += crop_p * cos_weight[np.newaxis, :, :]
                weight_accum[ys : ys + tile_size, xs : xs + tile_size] += cos_weight

            del batch_t, logits
            if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Normalize accumulated probabilities by weights
        weight_accum = np.maximum(weight_accum, 1e-6)
        probs = prob_accum / weight_accum[np.newaxis, :, :]
        pred_mask = np.argmax(probs, axis=0).astype(np.uint8)

        if verbose:
            elapsed = time.time() - t0
            print(f"[asteria] Tiled inference finished in {elapsed:.2f}s ({elapsed / num_tiles * 1000:.1f}ms/tile).")

        return pred_mask, probs, stretched

    def segment(
        self,
        image: Union[np.ndarray, AstroImage, str, Path],
        interior_thresh: float = 0.45,
        fg_thresh: float = 0.50,
        min_area: int = 10,
        max_area: Optional[int] = None,
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: int = 2,
        stretch_type: str = "leakysb",
        **stretch_kwargs,
    ) -> SegmentationResult:
        """
        Segment astronomical sources into discrete, deblended object instances.

        Parameters
        ----------
        image : np.ndarray, AstroImage, str, or Path
            Input astronomical exposure.
        interior_thresh : float, default 0.45
            Probability threshold on source interior to mark object seeds.
        fg_thresh : float, default 0.50
            Probability threshold for total foreground boundary.
        min_area : int, default 10
            Minimum pixel area to retain a detected object.
        max_area : int, optional
        tile_size : int, default 512
        overlap : int, default 64
        batch_size : int, default 2
        stretch_type : str, default 'leakysb'

        Returns
        -------
        SegmentationResult
            Catalog containing masks, bounding boxes, centroids, and photometry statistics.
        """
        _, probs, _ = self.predict_dense(
            image=image,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            stretch_type=stretch_type,
            **stretch_kwargs,
        )
        return deblend_sources(
            probs=probs,
            interior_thresh=interior_thresh,
            fg_thresh=fg_thresh,
            min_area=min_area,
            max_area=max_area,
        )
