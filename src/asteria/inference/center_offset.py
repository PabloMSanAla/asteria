"""
Center Offset Voting Astronomical Instance Segmentation & Deblending Engine.
=============================================================================

Inference engine and instance deblender for UNet++ with Recurrent Criss-Cross
Attention (RCCA) and Multi-Task Center Offset Heads.

Converts multi-task predictions:
1. Semantic foreground probability map
2. Object centroid keypoint heatmap
3. Dense 2D offset displacement vector field (delta_y, delta_x)
into discrete astronomical source instances with celestial WCS coordinates and photometry.

Includes colinear ray-aligned voting to deblend long diffraction spikes from intersecting galaxies.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from tqdm import tqdm

from asteria.models.unetpp_offset import UNetPlusPlusCenterOffset
from asteria.inference.device import (
    get_optimal_device,
    get_device_info,
    get_adaptive_batch_size,
    empty_device_cache,
)
from asteria.inference.segmentor import _build_tile_window, _resolve_stretch_kwargs
from asteria.inference.watershed import SegmentationResult
from asteria.io.reader import load_image, AstroImage
from asteria.stretch.leakysb import stretch_image

logger = logging.getLogger(__name__)


def extract_center_peaks(
    center_prob: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.3,
    min_distance: int = 3,
) -> List[Tuple[float, float, float]]:
    """
    Extract 2D local maxima peaks from center probability heatmap via max-pooling NMS.

    Parameters
    ----------
    center_prob : np.ndarray or torch.Tensor
        Center probability array (H, W) in [0.0, 1.0].
    threshold : float, default 0.3
        Probability cutoff to consider a candidate peak.
    min_distance : int, default 3
        Footprint radius for local non-maximum suppression.

    Returns
    -------
    List[Tuple[float, float, float]]
        List of (cy, cx, score) for each detected peak, sorted descending by score.
    """
    if isinstance(center_prob, torch.Tensor):
        c_np = center_prob.detach().float().cpu().numpy()
    else:
        c_np = np.asarray(center_prob, dtype=np.float32)

    if c_np.ndim == 3 and c_np.shape[0] == 1:
        c_np = c_np[0]

    ksize = 2 * min_distance + 1
    max_filtered = ndimage.maximum_filter(c_np, size=ksize, mode="constant")
    is_peak = (c_np == max_filtered) & (c_np >= threshold)

    ys, xs = np.where(is_peak)
    scores = c_np[ys, xs]

    # Sort descending by peak score
    order = np.argsort(-scores)
    peaks = [(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]
    return peaks


def deblend_center_offset(
    pred_semantic: np.ndarray,
    pred_center: np.ndarray,
    pred_offsets: np.ndarray,
    fg_thresh: float = 0.50,
    center_thresh: float = 0.30,
    min_distance: int = 3,
    colinear_thresh: float = 0.70,
    max_vote_dist: float = 64.0,
    min_area: int = 4,
    max_area: Optional[int] = None,
    pixel_chunk_size: Optional[int] = None,
    return_masks: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray], List[List[float]], List[float], List[int], List[Tuple[float, float]]]:
    """
    Deblend instances directly from predicted semantic, center, and offset maps using
    batched ray-aligned colinear offset voting.

    Memory-efficient implementation: uses pixel chunking to keep intermediate distance
    matrices bounded (< 50MB) even on large images (e.g. 4088x4088 px with thousands of centers).

    Parameters
    ----------
    pred_semantic : np.ndarray
        Foreground probability array of shape (H, W).
    pred_center : np.ndarray
        Center keypoint heatmap of shape (H, W).
    pred_offsets : np.ndarray
        Offset vector field of shape (2, H, W) where [0] is delta_y and [1] is delta_x.
    fg_thresh : float, default 0.50
        Foreground probability threshold.
    center_thresh : float, default 0.30
        Center peak probability threshold.
    min_distance : int, default 3
        Peak NMS radius.
    colinear_thresh : float, default 0.70
        Cosine angle threshold for spike alignment.
    max_vote_dist : float, default 64.0
        Maximum destination residual distance.
    min_area : int, default 4
        Minimum pixel count to retain an instance.
    max_area : int, optional
        Maximum pixel count to filter large diffuse backgrounds.
    pixel_chunk_size : int, optional
        Number of foreground pixels processed per distance computation chunk.
        Auto-calculated from candidate count to cap memory at ~50 MB if None.
    return_masks : bool, default True
        Whether to return individual full-frame boolean masks in masks_list.
        Set to False for huge frames when only labeled_map and catalogs are needed.

    Returns
    -------
    Tuple containing:
        - labeled_map : (H, W) integer map of instance IDs (0: bg, 1..N: instances)
        - masks_list : List of boolean masks (empty if return_masks=False)
        - boxes_list : List of [xmin, ymin, xmax, ymax]
        - scores_list : List of mean foreground scores
        - areas_list : List of pixel counts
        - centroids_list : List of (cx, cy) pixel coordinates
    """
    p_fg = pred_semantic.squeeze()
    p_ctr = pred_center.squeeze()
    h, w = p_fg.shape

    # 1. Extract candidate centers via peak NMS
    peaks = extract_center_peaks(p_ctr, threshold=center_thresh, min_distance=min_distance)

    # 2. Foreground mask & chunked voting
    fg_mask = p_fg >= fg_thresh
    labeled_map = np.zeros((h, w), dtype=np.int32)

    if len(peaks) > 0 and np.any(fg_mask):
        centers = np.array([[p[0], p[1]] for p in peaks], dtype=np.float32)  # (K, 2): [cy, cx]
        fg_ys, fg_xs = np.where(fg_mask)
        n_fg = len(fg_ys)
        n_centers = len(centers)

        # Dynamic chunk size to keep intermediate memory footprint bounded (< 50MB)
        if pixel_chunk_size is None:
            chunk_sz = max(1000, min(50000, int(50_000_000 // (max(1, n_centers) * 20))))
        else:
            chunk_sz = max(1, pixel_chunk_size)

        dy_all = pred_offsets[0, fg_ys, fg_xs]
        dx_all = pred_offsets[1, fg_ys, fg_xs]

        for start_idx in range(0, n_fg, chunk_sz):
            end_idx = min(start_idx + chunk_sz, n_fg)
            c_ys = fg_ys[start_idx:end_idx]
            c_xs = fg_xs[start_idx:end_idx]
            c_dy = dy_all[start_idx:end_idx]
            c_dx = dx_all[start_idx:end_idx]

            dest_ys = c_ys + c_dy
            dest_xs = c_xs + c_dx

            # Distance matrix between destination points and center peaks: (chunk_sz, K)
            dists = np.sqrt(
                (dest_ys[:, None] - centers[None, :, 0]) ** 2
                + (dest_xs[:, None] - centers[None, :, 1]) ** 2
            )

            # Colinear angular check for distant points (diffraction spikes & extended halos)
            vec_to_centers_y = centers[None, :, 0] - c_ys[:, None]
            vec_to_centers_x = centers[None, :, 1] - c_xs[:, None]
            norm_c = np.sqrt(vec_to_centers_y ** 2 + vec_to_centers_x ** 2) + 1e-6
            norm_p = np.sqrt(c_dy ** 2 + c_dx ** 2)[:, None] + 1e-6

            cos_angles = (c_dy[:, None] * vec_to_centers_y + c_dx[:, None] * vec_to_centers_x) / (
                norm_p * norm_c
            )

            is_distant = norm_c > 10.0
            misaligned = is_distant & (cos_angles < colinear_thresh)
            dists[misaligned] += 1000.0

            best_center_indices = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(c_ys)), best_center_indices]

            valid = min_dists <= max_vote_dist
            if np.any(valid):
                labeled_map[c_ys[valid], c_xs[valid]] = best_center_indices[valid] + 1

    # 3. Extract individual instance properties via fast bounding-box slicing
    masks_list: List[np.ndarray] = []
    boxes_list: List[List[float]] = []
    scores_list: List[float] = []
    areas_list: List[int] = []
    centroids_list: List[Tuple[float, float]] = []

    if len(peaks) > 0 and np.any(labeled_map > 0):
        slices = ndimage.find_objects(labeled_map)
        for obj_idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            sub_labels = labeled_map[slc]
            sub_m = (sub_labels == obj_idx)
            area = int(np.sum(sub_m))
            if area < min_area or (max_area is not None and area > max_area):
                continue

            ys_local, xs_local = np.where(sub_m)
            y_start = slc[0].start
            x_start = slc[1].start
            ys = ys_local + y_start
            xs = xs_local + x_start

            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            score = float(np.mean(p_fg[slc][sub_m]))

            x_min = float(np.min(xs))
            x_max = float(np.max(xs))
            y_min = float(np.min(ys))
            y_max = float(np.max(ys))

            if return_masks:
                m = np.zeros((h, w), dtype=bool)
                m[ys, xs] = True
                masks_list.append(m)

            boxes_list.append([x_min, y_min, x_max, y_max])
            scores_list.append(score)
            areas_list.append(area)
            centroids_list.append((cx, cy))

    return labeled_map, masks_list, boxes_list, scores_list, areas_list, centroids_list


class CenterOffsetSegmentor:
    """
    Astronomical instance segmentor powered by UNet++ with Recurrent Criss-Cross Attention
    and Center Offset Voting.

    Parameters
    ----------
    model : UNetPlusPlusCenterOffset, optional
        Instantiated PyTorch neural network.
    checkpoint_path : str or Path, optional
        Path to fine-tuned model checkpoint file (.pt).
    device : str or torch.device, optional
        Target accelerator ('cuda', 'mps', 'cpu'). Auto-detected if None.
    in_channels : int, default 3
        Input channels (3 for RGB/LeakySB, 1 for single-band FITS).
    semantic_classes : int, default 1
        Semantic foreground classes.
    features : Tuple[int, ...], default (32, 64, 128, 256, 512)
        Feature map dimensions per hierarchical depth.
    use_cca : bool, default True
        Whether to enable Recurrent Criss-Cross Attention (RCCA).
    """

    deblend_from_maps = staticmethod(deblend_center_offset)
    _build_tile_window = staticmethod(_build_tile_window)

    def __init__(
        self,
        model: Optional[UNetPlusPlusCenterOffset] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        in_channels: int = 3,
        semantic_classes: int = 1,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        use_cca: bool = True,
    ):
        self.device = get_optimal_device(device)

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model = self._load_from_checkpoint(checkpoint_path)
        else:
            self.model = UNetPlusPlusCenterOffset(
                in_channels=in_channels,
                semantic_classes=semantic_classes,
                features=features,
                use_cca=use_cca,
            ).to(self.device)

        self.model.eval()

    def _load_from_checkpoint(self, path: Union[str, Path]) -> UNetPlusPlusCenterOffset:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location=self.device)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )

        # Sanitize keys: strip DDP 'module.' or compiled '_orig_mod.' prefixes
        cleaned_sd = {}
        for k, v in state_dict.items():
            k_clean = k
            if k_clean.startswith("module."):
                k_clean = k_clean[len("module.") :]
            if k_clean.startswith("_orig_mod."):
                k_clean = k_clean[len("_orig_mod.") :]
            cleaned_sd[k_clean] = v

        in_channels = ckpt.get("in_channels", 3) if isinstance(ckpt, dict) else 3
        semantic_classes = ckpt.get("semantic_classes", 1) if isinstance(ckpt, dict) else 1
        features = tuple(ckpt.get("features", (32, 64, 128, 256, 512))) if isinstance(ckpt, dict) else (32, 64, 128, 256, 512)
        use_cca = ckpt.get("use_cca", True) if isinstance(ckpt, dict) else True

        model = UNetPlusPlusCenterOffset(
            in_channels=in_channels,
            semantic_classes=semantic_classes,
            features=features,
            use_cca=use_cca,
        ).to(self.device)

        model.load_state_dict(cleaned_sd, strict=False)
        logger.info(f"[center_offset] Loaded weights from: {path} onto {get_device_info(self.device)}")
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "CenterOffsetSegmentor":
        return cls(checkpoint_path=checkpoint_path, device=device)

    @classmethod
    def from_pretrained(
        cls,
        version_or_path: Union[str, Path] = "latest",
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "CenterOffsetSegmentor":
        p = Path(version_or_path)
        if p.exists():
            return cls.from_checkpoint(p, device=device)
        # Default initialization if no local file
        return cls(device=device, **kwargs)

    def _predict_single_patch(
        self,
        norm_3ch: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run single un-tiled forward pass on a small image patch (CHW layout)."""
        h, w = orig_h, orig_w
        align_mult = 16

        pad_h = (align_mult - (h % align_mult)) % align_mult
        pad_w = (align_mult - (w % align_mult)) % align_mult

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(norm_3ch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        else:
            padded = norm_3ch

        in_channels = getattr(self.model, "in_channels", 3)
        if in_channels == 1:
            inp = padded[:1]
        else:
            inp = padded

        input_t = torch.from_numpy(inp).unsqueeze(0).to(torch.float32).to(self.device)

        try:
            with torch.no_grad():
                preds = self.model(input_t)
                sem_logits = preds["semantic"]
                ctr_logits = preds["center"]
                off_logits = preds["offset"]

                p_sem = torch.sigmoid(sem_logits[0, 0, :h, :w]).float().cpu().numpy()
                p_ctr = torch.sigmoid(ctr_logits[0, 0, :h, :w]).float().cpu().numpy()
                p_off = off_logits[0, :, :h, :w].float().cpu().numpy()
        finally:
            del input_t
            empty_device_cache(self.device)

        return p_sem, p_ctr, p_off

    def predict(
        self,
        image: Union[np.ndarray, AstroImage, str, Path],
        stretch_type: str = "leakysb",
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: Optional[int] = None,
        memory_fraction: float = 0.2,
        zero_points: Optional[float] = None,
        pixel_scale: Optional[float] = None,
        mmin: Optional[float] = None,
        mmax: Optional[float] = None,
        leak_slope: Optional[float] = None,
        negative_slope: Optional[float] = None,
        positive_slope: Optional[float] = None,
        verbose: bool = False,
        **stretch_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run multi-task inference predicting semantic foreground, center heatmap, and offset vectors.

        If the image dimensions exceed tile_size (default 512x512), inference is automatically
        partitioned into sliding tiles processed in memory-bounded batches sized to available
        device memory (CUDA, MPS, CPU).

        Parameters
        ----------
        image : np.ndarray, AstroImage, str, or Path
            Input astronomical exposure or cutout.
        stretch_type : str, default 'leakysb'
            Photometric stretch algorithm.
        tile_size : int, default 512
            Tile width/height in pixels. Set to 0 for unchunked forward pass.
        overlap : int, default 64
            Overlap in pixels between adjacent tiles.
        batch_size : int, optional
            Number of tiles per forward pass. If None, dynamically calculated.
        memory_fraction : float, default 0.2
            Fraction of free memory budgeted for tile batches.
        zero_points : float, optional
        pixel_scale : float, optional
        mmin, mmax : float, optional
        leak_slope, negative_slope, positive_slope : float, optional
        verbose : bool, default False
        **stretch_kwargs

        Returns
        -------
        pred_semantic : np.ndarray
            Foreground probability array in [0.0, 1.0] of shape (H, W).
        pred_center : np.ndarray
            Center keypoint probability heatmap in [0.0, 1.0] of shape (H, W).
        pred_offsets : np.ndarray
            2D displacement vector field of shape (2, H, W) [delta_y, delta_x].
        stretched : np.ndarray
            Photometrically stretched input image in [0.0, 1.0] of shape (H, W).
        """
        astro = load_image(image, sky=stretch_kwargs.get("sky", None))
        raw_data = astro.data
        h, w = raw_data.shape[:2]

        if tile_size > 0 and (h > tile_size or w > tile_size):
            return self.predict_dense(
                image=astro,
                tile_size=tile_size,
                overlap=overlap,
                batch_size=batch_size,
                memory_fraction=memory_fraction,
                stretch_type=stretch_type,
                verbose=verbose,
                zero_points=zero_points,
                pixel_scale=pixel_scale,
                mmin=mmin,
                mmax=mmax,
                leak_slope=leak_slope,
                negative_slope=negative_slope,
                positive_slope=positive_slope,
                **stretch_kwargs,
            )

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
            **stretch_kwargs,
        )

        stretched = stretch_image(raw_data, stretch_type=stretch_type, **resolved_stretch)

        if raw_data.ndim == 2:
            norm_3ch = np.repeat(stretched[np.newaxis, :, :], 3, axis=0)
        else:
            if stretched.shape[0] == 3:
                norm_3ch = stretched
            else:
                norm_3ch = np.transpose(stretched[:, :, :3], (2, 0, 1))

        p_sem, p_ctr, p_off = self._predict_single_patch(
            norm_3ch=norm_3ch,
            orig_h=h,
            orig_w=w,
        )
        return p_sem, p_ctr, p_off, stretched

    def predict_dense(
        self,
        image: Union[np.ndarray, AstroImage, str, Path],
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: Optional[int] = None,
        memory_fraction: float = 0.2,
        stretch_type: str = "leakysb",
        verbose: bool = False,
        zero_points: Optional[float] = None,
        pixel_scale: Optional[float] = None,
        mmin: Optional[float] = None,
        mmax: Optional[float] = None,
        leak_slope: Optional[float] = None,
        negative_slope: Optional[float] = None,
        positive_slope: Optional[float] = None,
        **stretch_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run dense tiled inference over full astronomical exposures.
        Seamlessly processes arbitrary image sizes via sliding-window tiling with
        edge-aware cosine window blending, executing in memory-bounded batches dynamically
        sized to available device memory.
        """
        astro = load_image(image, sky=stretch_kwargs.get("sky", None))
        raw_data = astro.data
        h, w = raw_data.shape[:2]

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
            **stretch_kwargs,
        )

        stretched = stretch_image(raw_data, stretch_type=stretch_type, **resolved_stretch)

        if raw_data.ndim == 2:
            norm_3ch = np.repeat(stretched[np.newaxis, :, :], 3, axis=0)
        else:
            if stretched.shape[0] == 3:
                norm_3ch = stretched
            else:
                norm_3ch = np.transpose(stretched[:, :, :3], (2, 0, 1))

        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            p_sem, p_ctr, p_off = self._predict_single_patch(
                norm_3ch=norm_3ch,
                orig_h=h,
                orig_w=w,
            )
            return p_sem, p_ctr, p_off, stretched

        if batch_size is None:
            eff_batch_size = get_adaptive_batch_size(
                device=self.device,
                tile_size=tile_size,
                memory_fraction=memory_fraction,
            )
        else:
            eff_batch_size = max(1, int(batch_size))

        step = tile_size - overlap
        y_starts = list(range(0, max(1, h - tile_size + 1), step))
        if len(y_starts) == 0 or y_starts[-1] + tile_size < h:
            y_starts.append(max(0, h - tile_size))
        y_starts = sorted(set(y_starts))

        x_starts = list(range(0, max(1, w - tile_size + 1), step))
        if len(x_starts) == 0 or x_starts[-1] + tile_size < w:
            x_starts.append(max(0, w - tile_size))
        x_starts = sorted(set(x_starts))

        accum_sem = np.zeros((h, w), dtype=np.float32)
        accum_ctr = np.zeros((h, w), dtype=np.float32)
        accum_off = np.zeros((2, h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        coords = [(ys, xs) for ys in y_starts for xs in x_starts]
        total_tiles = len(coords)

        if verbose:
            logger.info(
                f"[center_offset] Running dense tiling: {h}x{w} px with {total_tiles} tiles of {tile_size}x{tile_size} "
                f"(overlap {overlap} px, batch_size={eff_batch_size})"
            )

        in_channels = getattr(self.model, "in_channels", 3)

        def _run_forward_with_retry(
            b_patches: List[np.ndarray],
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            nonlocal eff_batch_size
            while True:
                try:
                    raw_stack = np.stack(b_patches, axis=0)
                    if in_channels == 1:
                        raw_stack = raw_stack[:, :1]
                    b_t = torch.from_numpy(raw_stack).to(torch.float32).to(self.device)

                    with torch.inference_mode():
                        preds = self.model(b_t)
                        sem_np = torch.sigmoid(preds["semantic"][:, 0]).float().cpu().numpy()
                        ctr_np = torch.sigmoid(preds["center"][:, 0]).float().cpu().numpy()
                        off_np = preds["offset"].float().cpu().numpy()

                    del b_t, preds
                    return sem_np, ctr_np, off_np
                except (torch.OutOfMemoryError, RuntimeError) as err:
                    empty_device_cache(self.device)
                    is_oom = isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()
                    if is_oom and len(b_patches) > 1:
                        new_bs = max(1, len(b_patches) // 2)
                        logger.warning(
                            f"[center_offset] OOM with batch size {len(b_patches)}. Halving to {new_bs} and retrying."
                        )
                        eff_batch_size = new_bs
                        mid = len(b_patches) // 2
                        s1, c1, o1 = _run_forward_with_retry(b_patches[:mid])
                        s2, c2, o2 = _run_forward_with_retry(b_patches[mid:])
                        return (
                            np.concatenate([s1, s2], axis=0),
                            np.concatenate([c1, c2], axis=0),
                            np.concatenate([o1, o2], axis=0),
                        )
                    raise err

        batch_patches = []
        batch_meta = []

        pbar = tqdm(
            total=total_tiles,
            desc="[asteria] Center Offset",
            unit="tile",
            disable=not verbose,
            leave=True,
        )

        for idx, (ys, xs) in enumerate(coords):
            ye = min(ys + tile_size, h)
            xe = min(xs + tile_size, w)

            patch = norm_3ch[:, ys:ye, xs:xe]
            ph, pw = patch.shape[1], patch.shape[2]

            if ph < tile_size or pw < tile_size:
                pad_y = tile_size - ph
                pad_x = tile_size - pw
                patch = np.pad(patch, ((0, 0), (0, pad_y), (0, pad_x)), mode="reflect")

            window = _build_tile_window(
                size=tile_size,
                overlap=overlap,
                is_top=(ys == 0),
                is_bottom=(ye >= h),
                is_left=(xs == 0),
                is_right=(xe >= w),
            )

            batch_patches.append(patch)
            batch_meta.append((ys, xs, ph, pw, window))

            if len(batch_patches) == eff_batch_size or idx == total_tiles - 1:
                b_sem, b_ctr, b_off = _run_forward_with_retry(batch_patches)

                for b_i, (b_ys, b_xs, b_ph, b_pw, b_win) in enumerate(batch_meta):
                    s_patch = b_sem[b_i, :b_ph, :b_pw]
                    c_patch = b_ctr[b_i, :b_ph, :b_pw]
                    o_patch = b_off[b_i, :, :b_ph, :b_pw]
                    w_patch = b_win[:b_ph, :b_pw]

                    accum_sem[b_ys : b_ys + b_ph, b_xs : b_xs + b_pw] += s_patch * w_patch
                    accum_ctr[b_ys : b_ys + b_ph, b_xs : b_xs + b_pw] += c_patch * w_patch
                    accum_off[:, b_ys : b_ys + b_ph, b_xs : b_xs + b_pw] += o_patch * w_patch[np.newaxis, :, :]
                    weight_map[b_ys : b_ys + b_ph, b_xs : b_xs + b_pw] += w_patch

                pbar.update(len(batch_patches))
                batch_patches.clear()
                batch_meta.clear()

        pbar.close()

        weight_map = np.maximum(weight_map, 1e-6)
        pred_sem = np.clip(accum_sem / weight_map, 0.0, 1.0)
        pred_ctr = np.clip(accum_ctr / weight_map, 0.0, 1.0)
        pred_off = accum_off / weight_map[np.newaxis, :, :]

        return pred_sem, pred_ctr, pred_off, stretched

    def segment_instances(
        self,
        image: Union[np.ndarray, AstroImage, str, Path],
        stretch_type: str = "leakysb",
        fg_thresh: float = 0.50,
        center_thresh: float = 0.30,
        min_distance: int = 3,
        colinear_thresh: float = 0.70,
        max_vote_dist: float = 64.0,
        min_area: int = 4,
        max_area: Optional[int] = None,
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: Optional[int] = None,
        pixel_chunk_size: Optional[int] = None,
        return_masks: bool = True,
        verbose: bool = False,
        **stretch_kwargs,
    ) -> SegmentationResult:
        """
        Segment all sources in astronomical exposure using Center Offset Voting.

        Parameters
        ----------
        image : np.ndarray, AstroImage, str, or Path
            Input astronomical exposure.
        stretch_type : str, default 'leakysb'
        fg_thresh : float, default 0.50
            Probability threshold on foreground semantic map.
        center_thresh : float, default 0.30
            Probability threshold on center centroid keypoint heatmap.
        min_distance : int, default 3
            Footprint radius for local peak NMS.
        colinear_thresh : float, default 0.70
            Cosine angular threshold for diffraction spike assignment.
        max_vote_dist : float, default 64.0
            Maximum residual Euclidean distance for offset clustering.
        min_area : int, default 4
            Minimum pixels to retain an instance.
        max_area : int, optional
        tile_size : int, default 512
        overlap : int, default 64
        batch_size : int, optional
        pixel_chunk_size : int, optional
            Foreground pixel batch size for chunked distance voting.
        return_masks : bool, default True
            Whether to construct full-frame boolean masks list. Set to False for massive frames.
        verbose : bool, default False
        **stretch_kwargs

        Returns
        -------
        SegmentationResult
            Discretely deblended instances with masks, boxes, scores, and celestial coordinates.
        """
        t0 = time.time()
        astro = load_image(image, sky=stretch_kwargs.get("sky", None))

        p_sem, p_ctr, p_off, _ = self.predict(
            image=astro,
            stretch_type=stretch_type,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            verbose=verbose,
            **stretch_kwargs,
        )

        labeled_map, masks_list, boxes_list, scores_list, areas_list, centroids_list = self.deblend_from_maps(
            pred_semantic=p_sem,
            pred_center=p_ctr,
            pred_offsets=p_off,
            fg_thresh=fg_thresh,
            center_thresh=center_thresh,
            min_distance=min_distance,
            colinear_thresh=colinear_thresh,
            max_vote_dist=max_vote_dist,
            min_area=min_area,
            max_area=max_area,
            pixel_chunk_size=pixel_chunk_size,
            return_masks=return_masks,
        )

        sky_centroids: Optional[List[Tuple[float, float]]] = None
        if astro.wcs is not None:
            sky_centroids = []
            for cx, cy in centroids_list:
                try:
                    sky_pt = astro.pixel_to_world(cx, cy)
                    if hasattr(sky_pt, "ra") and hasattr(sky_pt, "dec"):
                        sky_centroids.append((float(sky_pt.ra.deg), float(sky_pt.dec.deg)))
                    elif isinstance(sky_pt, (tuple, list)) and len(sky_pt) >= 2:
                        sky_centroids.append((float(sky_pt[0]), float(sky_pt[1])))
                except Exception:
                    pass

        elapsed = time.time() - t0
        if verbose:
            logger.info(
                f"[center_offset] Segmented ({astro.data.shape[1]}x{astro.data.shape[0]}) in {elapsed*1000:.1f}ms on {self.device}. "
                f"Deblended {len(masks_list)} sources."
            )

        return SegmentationResult(
            masks=masks_list,
            boxes=boxes_list,
            centroids=centroids_list,
            scores=scores_list,
            areas=areas_list,
            segmap=labeled_map,
            sky_centroids=sky_centroids,
        )
