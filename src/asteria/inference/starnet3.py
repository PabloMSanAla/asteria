"""
StarNet3 Astronomical Neural Star Removal Inference Engine.

Runs star removal on astronomical exposures preprocessed with LeakySB photometric stretching.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from asteria.models.starnet3 import StarNet3Model, ProgressiveMultiScaleUNetWithRefiner
from asteria.inference.device import (
    get_optimal_device,
    get_device_info,
    get_free_memory,
    get_adaptive_batch_size,
    empty_device_cache,
)
from asteria.inference.segmentor import _build_tile_window, _resolve_stretch_kwargs
from asteria.io.reader import load_image, AstroImage
from asteria.stretch.leakysb import stretch_image
from asteria.weights.starnet3_manager import resolve_starnet3_checkpoint_path, download_starnet3_model

logger = logging.getLogger(__name__)


class StarNet3:
    """
    StarNet3 neural astronomical star removal engine.

    Executes neural star removal on astronomical images preconditioned with
    LeakySB photometric stretching, producing continuous starless reconstructions
    and complementary star intensity masks.

    Parameters
    ----------
    model : StarNet3Model or nn.Module
        Instantiated StarNet3 neural network.
    device : torch.device or str, optional
        Target accelerator device ('cuda', 'mps', 'cpu'). Auto-selected if None.
    """

    _build_tile_window = staticmethod(_build_tile_window)

    def __init__(
        self,
        model: StarNet3Model,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = get_optimal_device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def download(
        cls,
        version: str = "latest",
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Download StarNet3 pre-trained weights to disk."""
        return download_starnet3_model(version=version, cache_dir=cache_dir)

    @classmethod
    def from_pretrained(
        cls,
        version_or_path: Union[str, Path] = "latest",
        device: Optional[Union[str, torch.device]] = None,
        dim: int = 24,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8, 16),
        unet_scales: Tuple[int, ...] = (2, 16),
        use_pos_guide: Optional[bool] = None,
        **model_kwargs,
    ) -> "StarNet3":
        """
        Load a StarNet3 model from pre-trained weights.

        Parameters
        ----------
        version_or_path : str or Path, default 'latest'
            Version string (e.g. 'v0.1.0', 'latest') or direct path to a .pt checkpoint.
        device : str or torch.device, optional
            Hardware device. Auto-selects CUDA -> MPS -> CPU if None.
        dim : int, default 24
            Base feature dimension of the UNet.
        dim_mults : tuple of int, default (1, 2, 4, 8, 16)
            Channel depth multipliers.
        unet_scales : tuple of int, default (2, 16)
            Multi-scale downsample stages.
        use_pos_guide : bool, optional
            Whether model was trained with guide mask support. If None, auto-detected
            from checkpoint state dict keys.
        **model_kwargs
            Additional arguments passed to StarNet3Model constructor.

        Returns
        -------
        StarNet3
        """
        ckpt_path = resolve_starnet3_checkpoint_path(version_or_path)
        dev = get_optimal_device(device)

        ckpt = torch.load(ckpt_path, map_location=dev)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        # Sanitize keys: strip DDP 'module.' or compiled '_orig_mod.' prefixes
        cleaned_sd = {}
        for k, v in state_dict.items():
            k_clean = k
            if k_clean.startswith("module."):
                k_clean = k_clean[len("module."):]
            if k_clean.startswith("_orig_mod."):
                k_clean = k_clean[len("_orig_mod."):]
            cleaned_sd[k_clean] = v

        # Auto-detect architecture parameters from state dict if available
        if "shared_unet.unet.final_conv.0.block1.proj.weight" in cleaned_sd:
            dim = cleaned_sd["shared_unet.unet.final_conv.0.block1.proj.weight"].shape[0]
        elif "unet.final_conv.0.block1.proj.weight" in cleaned_sd:
            dim = cleaned_sd["unet.final_conv.0.block1.proj.weight"].shape[0]

        refiner_n_feat = model_kwargs.pop("refiner_n_feat", 64)
        if "fullres_refiner.fuse_to_feat.weight" in cleaned_sd:
            refiner_n_feat = cleaned_sd["fullres_refiner.fuse_to_feat.weight"].shape[0]

        refiner_branch_width = model_kwargs.pop("refiner_branch_width", 8)
        if "fullres_refiner.branch_projs.0.proj.0.weight" in cleaned_sd:
            refiner_branch_width = cleaned_sd["fullres_refiner.branch_projs.0.proj.0.weight"].shape[0]

        adapter_hidden_dim = model_kwargs.pop("adapter_hidden_dim", 32)
        if "first_stage_adapter.conv_in.weight" in cleaned_sd:
            adapter_hidden_dim = cleaned_sd["first_stage_adapter.conv_in.weight"].shape[0]

        sam_hidden_dim = model_kwargs.pop("sam_hidden_dim", 32)
        if "sam_modules.0.conv1.weight" in cleaned_sd:
            sam_hidden_dim = cleaned_sd["sam_modules.0.conv1.weight"].shape[0]

        stage_input_channels = model_kwargs.pop("stage_input_channels", 8)
        if "first_stage_adapter.conv_out.weight" in cleaned_sd:
            stage_input_channels = cleaned_sd["first_stage_adapter.conv_out.weight"].shape[0]

        refiner_num_orb = model_kwargs.pop("refiner_num_orb", 4)
        orb_indices = [
            int(k.split(".")[2])
            for k in cleaned_sd
            if k.startswith("fullres_refiner.orbs.") and k.split(".")[2].isdigit()
        ]
        if orb_indices:
            refiner_num_orb = max(orb_indices) + 1

        refiner_orb_num_cab = model_kwargs.pop("refiner_orb_num_cab", 2)
        cab_indices = [
            int(k.split(".")[4])
            for k in cleaned_sd
            if k.startswith("fullres_refiner.orbs.0.body.") and k.split(".")[4].isdigit()
        ]
        if cab_indices:
            refiner_orb_num_cab = max(cab_indices)  # last element is conv

        refiner_num_fusion_blocks = model_kwargs.pop("refiner_num_fusion_blocks", 2)
        fusion_indices = [
            int(k.split(".")[2])
            for k in cleaned_sd
            if k.startswith("fullres_refiner.fusion_blocks.") and k.split(".")[2].isdigit()
        ]
        if fusion_indices:
            refiner_num_fusion_blocks = max(fusion_indices) + 1

        # Auto-detect pos_guide from weights if not explicitly overridden
        if use_pos_guide is None:
            use_pos_guide = any("guide_adapters" in k for k in cleaned_sd.keys())

        # Extract config if bundled in checkpoint dict
        if isinstance(ckpt, dict):
            dim = ckpt.get("dim", dim)
            if "unet_scales" in ckpt:
                unet_scales = tuple(ckpt["unet_scales"])
            if "dim_mults" in ckpt:
                dim_mults = tuple(ckpt["dim_mults"])

        model = StarNet3Model(
            dim=dim,
            dim_mults=dim_mults,
            unet_scales=unet_scales,
            use_pos_guide=use_pos_guide,
            refiner_n_feat=refiner_n_feat,
            refiner_branch_width=refiner_branch_width,
            adapter_hidden_dim=adapter_hidden_dim,
            sam_hidden_dim=sam_hidden_dim,
            stage_input_channels=stage_input_channels,
            refiner_num_orb=refiner_num_orb,
            refiner_orb_num_cab=refiner_orb_num_cab,
            refiner_num_fusion_blocks=refiner_num_fusion_blocks,
            **model_kwargs,
        )
        model.load_state_dict(cleaned_sd, strict=False)
        logger.info(f"[starnet3] Loaded weights from: {ckpt_path} onto {get_device_info(dev)}")

        return cls(model=model, device=dev)

    def _predict_single_patch(
        self,
        norm_3ch: np.ndarray,
        is_2d: bool,
        orig_h: int,
        orig_w: int,
        pos_guide: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run single un-tiled forward pass on a small image patch (CHW layout)."""
        h, w = orig_h, orig_w
        max_scale = max(self.model.unet_scales) if hasattr(self.model, "unet_scales") else 16
        align_mult = max_scale * 16  # e.g. 16 * 16 = 256

        pad_h = (align_mult - (h % align_mult)) % align_mult
        pad_w = (align_mult - (w % align_mult)) % align_mult

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(norm_3ch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        else:
            padded = norm_3ch

        input_t = torch.from_numpy(padded).unsqueeze(0).to(torch.float32).to(self.device)

        pos_guide_t = None
        if pos_guide is not None:
            g = pos_guide.astype(np.float32)
            if pad_h > 0 or pad_w > 0:
                g = np.pad(g, ((0, pad_h), (0, pad_w)), mode="reflect")
            pos_guide_t = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                out_t, mask_t, _ = self.model(input_t, pos_guide=pos_guide_t)

            out_np = out_t.squeeze(0)[:, :h, :w].cpu().numpy().astype(np.float32)
            out_np = np.clip(out_np, 0.0, 1.0)

            mask_np = mask_t.squeeze(0).squeeze(0)[:h, :w].cpu().numpy().astype(np.float32)
            mask_np = np.clip(mask_np, 0.0, 1.0)
        finally:
            del input_t
            if pos_guide_t is not None:
                del pos_guide_t
            empty_device_cache(self.device)

        if is_2d:
            starless = np.mean(out_np, axis=0)
        else:
            starless = np.transpose(out_np, (1, 2, 0))

        return starless, mask_np

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
        pos_guide: Optional[np.ndarray] = None,
        verbose: bool = False,
        **stretch_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run star removal inference on an astronomical image patch or full exposure.

        If the image dimensions exceed tile_size (default 512x512), inference is automatically
        partitioned into sliding tiles processed in batches dynamically sized to available
        device memory (CUDA VRAM, Apple Silicon MPS unified memory, or CPU RAM) to prevent OOM.

        Parameters
        ----------
        image : np.ndarray, AstroImage, str, or Path
            Input astronomical exposure or cutout.
        stretch_type : str, default 'leakysb'
            Intensity stretch algorithm ('leakysb', 'asinh', 'linear').
        tile_size : int, default 512
            Tile width/height in pixels. 512x512 tiles provide bounded memory execution.
            Set to 0 to force single un-tiled forward pass.
        overlap : int, default 64
            Overlap in pixels between adjacent tiles when tiling.
        batch_size : int, optional
            Number of tiles per forward pass batch. If None (default), dynamically
            calculated according to available free memory on CUDA, MPS, or CPU.
        memory_fraction : float, default 0.2
            Fraction of available free memory budgeted for the batch (e.g. 0.2 = 20%).
        zero_points : float, optional
            Photometric zero point for surface brightness transformation.
        pixel_scale : float, optional
            Detector plate scale in arcsec/pixel.
        mmin : float, optional
            Bright-end surface brightness magnitude cutoff.
        mmax : float, optional
            Faint-end surface brightness magnitude cutoff.
        leak_slope : float, optional
            Slope for sub-background values.
        negative_slope : float, optional
            Faint-end slope for LeakySB.
        positive_slope : float, optional
            Bright-end slope for LeakySB.
        pos_guide : np.ndarray, optional
            Optional 2D star location guide mask of shape (H, W).
        verbose : bool, default False
            Whether to log execution progress.
        **stretch_kwargs
            Additional arguments forwarded to stretch_image.

        Returns
        -------
        starless : np.ndarray
            Starless image in [0.0, 1.0], matching input spatial format.
        star_mask : np.ndarray
            2D star mask in [0.0, 1.0] (H, W).
        stretched : np.ndarray
            Input image after photometric stretching in [0.0, 1.0].
        """
        astro = load_image(image, sky=stretch_kwargs.get("sky", None))
        raw_data = astro.data
        h, w = raw_data.shape[:2]

        # Automatically route to batched tiling if the image exceeds tile dimensions
        if tile_size > 0 and (h > tile_size or w > tile_size):
            return self.predict_dense(
                image=astro,
                tile_size=tile_size,
                overlap=overlap,
                batch_size=batch_size,
                memory_fraction=memory_fraction,
                stretch_type=stretch_type,
                pos_guide=pos_guide,
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

        # Image fits within tile_size (or tiling bypassed with tile_size <= 0)
        is_2d = raw_data.ndim == 2
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

        if is_2d:
            norm_3ch = np.repeat(stretched[np.newaxis, :, :], 3, axis=0)  # (3, H, W)
        else:
            if stretched.shape[0] == 3:
                norm_3ch = stretched
            else:
                norm_3ch = np.transpose(stretched[:, :, :3], (2, 0, 1))

        starless, mask_np = self._predict_single_patch(
            norm_3ch=norm_3ch,
            is_2d=is_2d,
            orig_h=h,
            orig_w=w,
            pos_guide=pos_guide,
        )

        return starless, mask_np, stretched

    def predict_dense(
        self,
        image: Union[np.ndarray, AstroImage, str, Path],
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: Optional[int] = None,
        memory_fraction: float = 0.2,
        stretch_type: str = "leakysb",
        pos_guide: Optional[np.ndarray] = None,
        verbose: bool = False,
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
        Run dense star removal inference across large astronomical exposures.
        Uses sliding-window tiling with edge-aware cosine blending to prevent seams.
        Batches are dynamically sized to a fraction of available device memory (CUDA/MPS/CPU).

        Parameters
        ----------
        image : np.ndarray, AstroImage, str, or Path
            Input full exposure.
        tile_size : int, default 512
            Tile width/height in pixels.
        overlap : int, default 64
            Overlap in pixels between adjacent tiles.
        batch_size : int, optional
            Number of tiles per forward pass. If None (default), adaptively calculated
            based on currently available free memory.
        memory_fraction : float, default 0.2
            Fraction of free memory budgeted for the batch pass (default 20%).
        stretch_type : str, default 'leakysb'
            Intensity stretch algorithm.
        pos_guide : np.ndarray, optional
            Full-exposure star position guide mask (H, W).
        verbose : bool, default False
            Whether to log progress.
        **stretch_kwargs
            Additional arguments forwarded to stretch_image.

        Returns
        -------
        starless : np.ndarray
            Starless reconstructed image in [0.0, 1.0].
        star_mask : np.ndarray
            Full star mask in [0.0, 1.0] (H, W).
        stretched : np.ndarray
            Full input exposure after photometric stretch in [0.0, 1.0].
        """
        astro = load_image(image, sky=stretch_kwargs.get("sky", None))
        raw_data = astro.data
        is_2d = raw_data.ndim == 2
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

        if is_2d:
            norm_3ch = np.repeat(stretched[np.newaxis, :, :], 3, axis=0)  # (3, H, W)
        else:
            if stretched.shape[0] == 3:
                norm_3ch = stretched
            else:
                norm_3ch = np.transpose(stretched[:, :, :3], (2, 0, 1))

        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            starless, mask_np = self._predict_single_patch(
                norm_3ch=norm_3ch,
                is_2d=is_2d,
                orig_h=h,
                orig_w=w,
                pos_guide=pos_guide,
            )
            return starless, mask_np, stretched

        # Determine adaptive batch size if not explicitly provided
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

        accum_out = np.zeros((3, h, w), dtype=np.float32)
        accum_mask = np.zeros((1, h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        coords = [(ys, xs) for ys in y_starts for xs in x_starts]
        total_tiles = len(coords)

        if verbose:
            free_mem_mb = get_free_memory(self.device) / 1e6
            logger.info(
                f"[starnet3] Running dense star removal: {h}x{w} px with {total_tiles} tiles of {tile_size}x{tile_size} "
                f"(overlap {overlap} px, batch_size={eff_batch_size}, free_memory={free_mem_mb:.1f} MB)"
            )

        def _run_forward_with_retry(
            b_patches: List[np.ndarray],
            b_guides: List[Optional[np.ndarray]],
        ) -> Tuple[np.ndarray, np.ndarray]:
            nonlocal eff_batch_size
            while True:
                try:
                    b_t = torch.from_numpy(np.stack(b_patches, axis=0)).to(torch.float32).to(self.device)
                    bg_t = None
                    if any(g is not None for g in b_guides):
                        bg_np = np.stack(
                            [
                                g if g is not None else np.zeros((tile_size, tile_size), dtype=np.float32)
                                for g in b_guides
                            ],
                            axis=0,
                        )[:, np.newaxis, :, :]
                        bg_t = torch.from_numpy(bg_np).to(torch.float32).to(self.device)

                    with torch.inference_mode():
                        b_out, b_mask, _ = self.model(b_t, pos_guide=bg_t)

                    b_out_np = b_out.cpu().numpy().astype(np.float32)
                    b_mask_np = b_mask.cpu().numpy().astype(np.float32)
                    del b_t, b_out, b_mask
                    if bg_t is not None:
                        del bg_t
                    return b_out_np, b_mask_np
                except (torch.OutOfMemoryError, RuntimeError) as err:
                    empty_device_cache(self.device)
                    is_oom = isinstance(err, torch.OutOfMemoryError) or "out of memory" in str(err).lower()
                    if is_oom and len(b_patches) > 1:
                        new_bs = max(1, len(b_patches) // 2)
                        logger.warning(
                            f"[starnet3] Out of memory with batch size {len(b_patches)}. "
                            f"Adapting batch size to {new_bs} and retrying."
                        )
                        eff_batch_size = new_bs
                        mid = len(b_patches) // 2
                        out1, mask1 = _run_forward_with_retry(b_patches[:mid], b_guides[:mid])
                        out2, mask2 = _run_forward_with_retry(b_patches[mid:], b_guides[mid:])
                        return np.concatenate([out1, out2], axis=0), np.concatenate([mask1, mask2], axis=0)
                    raise err

        batch_patches = []
        batch_guides = []
        batch_meta = []

        pbar = tqdm(
            total=total_tiles,
            desc="[starnet3] Star Removal",
            unit="tile",
            disable=not verbose,
            leave=True,
        )

        for idx, (ys, xs) in enumerate(coords):
            ye = min(ys + tile_size, h)
            xe = min(xs + tile_size, w)

            patch = norm_3ch[:, ys:ye, xs:xe]
            ph, pw = patch.shape[1], patch.shape[2]

            # Pad edge patches to tile_size if necessary
            if ph < tile_size or pw < tile_size:
                pad_y = tile_size - ph
                pad_x = tile_size - pw
                patch = np.pad(patch, ((0, 0), (0, pad_y), (0, pad_x)), mode="reflect")

            guide_patch = None
            if pos_guide is not None:
                guide_patch = pos_guide[ys:ye, xs:xe].astype(np.float32)
                if ph < tile_size or pw < tile_size:
                    guide_patch = np.pad(guide_patch, ((0, pad_y), (0, pad_x)), mode="reflect")

            window = _build_tile_window(
                size=tile_size,
                overlap=overlap,
                is_top=(ys == 0),
                is_bottom=(ye >= h),
                is_left=(xs == 0),
                is_right=(xe >= w),
            )

            batch_patches.append(patch)
            batch_guides.append(guide_patch)
            batch_meta.append((ys, xs, ph, pw, window))

            if len(batch_patches) == eff_batch_size or idx == total_tiles - 1:
                b_out_np, b_mask_np = _run_forward_with_retry(batch_patches, batch_guides)

                for b_i, (b_ys, b_xs, b_ph, b_pw, b_win) in enumerate(batch_meta):
                    o_patch = b_out_np[b_i, :, :b_ph, :b_pw]
                    m_patch = b_mask_np[b_i, 0, :b_ph, :b_pw]
                    w_patch = b_win[:b_ph, :b_pw]

                    accum_out[:, b_ys : b_ys + b_ph, b_xs : b_xs + b_pw] += o_patch * w_patch[np.newaxis, :, :]
                    accum_mask[0, b_ys : b_ys + b_ph, b_xs : b_xs + b_pw] += m_patch * w_patch
                    weight_map[b_ys : b_ys + b_ph, b_xs : b_xs + b_pw] += w_patch

                pbar.update(len(batch_patches))
                batch_patches.clear()
                batch_guides.clear()
                batch_meta.clear()

        pbar.close()

        weight_map = np.maximum(weight_map, 1e-6)
        starless_chw = np.clip(accum_out / weight_map[np.newaxis, :, :], 0.0, 1.0)
        star_mask = np.clip(accum_mask[0] / weight_map, 0.0, 1.0)

        if is_2d:
            starless = np.mean(starless_chw, axis=0)
        else:
            starless = np.transpose(starless_chw, (1, 2, 0))

        return starless, star_mask, stretched


# Convenience alias
starnet3 = StarNet3
