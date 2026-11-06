"""
Matplotlib visualization routines for astronomical images and UNet++ predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Tuple, List
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import numpy as np


def plot_two_panel(
    stretched: np.ndarray,
    pred_mask: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 150,
    cmap_img: str = "gray",
    image_title: str = "Astronomical Image (LeakySB Stretch)",
    mask_title: str = "Predicted Segmentation Mask (Asteria UNet++)",
) -> plt.Figure:
    """
    Render a high-contrast side-by-side 2-panel figure:
    1. LeakySB-stretched astronomical exposure.
    2. Discrete 3-class segmentation map (Sky, Core Interior, Deblending Edge).

    Parameters
    ----------
    stretched : np.ndarray
        2D normalized [0, 1] astronomical image.
    pred_mask : np.ndarray
        2D integer prediction mask (0=Sky, 1=Core, 2=Edge).
    title : str, optional
    save_path : str or Path, optional
    figsize : Tuple[int, int], default (16, 8)
    dpi : int, default 150
    """
    h, w = stretched.shape
    total_px = h * w
    n_sky = int(np.sum(pred_mask == 0))
    n_core = int(np.sum(pred_mask == 1))
    n_edge = int(np.sum(pred_mask == 2))

    pct_sky = 100.0 * n_sky / total_px
    pct_core = 100.0 * n_core / total_px
    pct_edge = 100.0 * n_edge / total_px

    # Discrete 3-class colormap:
    # 0: Sky background (#0b0d1b - Dark charcoal / navy)
    # 1: Core interior (#00d4ff - Electric cyan)
    # 2: Boundary edge (#ff3366 - Vivid coral red)
    cmap_3class = ListedColormap(["#0b0d1b", "#00d4ff", "#ff3366"])
    norm_3class = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_3class.N)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    # Panel 1: Stretched Image
    im0 = axes[0].imshow(stretched, cmap=cmap_img, origin="lower")
    axes[0].set_title(f"{image_title}\nDimension: {w} × {h} px", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Pixel X", fontsize=10)
    axes[0].set_ylabel("Pixel Y", fontsize=10)
    axes[0].grid(False)

    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Normalized SB Flux [0, 1]", fontsize=9)

    # Panel 2: Predicted Mask
    axes[1].imshow(pred_mask, cmap=cmap_3class, norm=norm_3class, origin="lower")
    axes[1].set_title(
        f"{mask_title}\nCore: {pct_core:.2f}% | Edge: {pct_edge:.2f}% | Sky: {pct_sky:.2f}%",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_xlabel("Pixel X", fontsize=10)
    axes[1].grid(False)

    patches = [
        mpatches.Patch(color="#0b0d1b", label="Sky Background (0)"),
        mpatches.Patch(color="#00d4ff", label="Source Core (1)"),
        mpatches.Patch(color="#ff3366", label="Deblending Edge (2)"),
    ]
    axes[1].legend(
        handles=patches,
        loc="upper right",
        framealpha=0.85,
        facecolor="#1c2038",
        edgecolor="#444b6e",
        fontsize=9,
        labelcolor="white",
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def plot_instance_overlay(
    stretched: np.ndarray,
    masks: List[np.ndarray],
    centroids: Optional[List[Tuple[float, float]]] = None,
    boxes: Optional[List[List[float]]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 10),
    alpha: float = 0.45,
) -> plt.Figure:
    """Overlay discrete deblended source instances with randomized distinct colors."""
    h, w = stretched.shape
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(stretched, cmap="gray", origin="lower")

    if masks:
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        rng = np.random.RandomState(42)
        colors = 0.3 + 0.7 * rng.rand(max(len(masks), 50), 3)

        for idx, m in enumerate(masks):
            c = colors[idx % len(colors)]
            m_bool = m.astype(bool)
            overlay[m_bool, :3] = c
            overlay[m_bool, 3] = alpha

        ax.imshow(overlay, origin="lower")

    if centroids:
        for idx, (cx, cy) in enumerate(centroids):
            ax.plot(cx, cy, "r+", markersize=6, markeredgewidth=1.2)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")

    return fig


def plot_center_offset_multipanel(
    stretched: np.ndarray,
    pred_semantic: np.ndarray,
    pred_center: np.ndarray,
    pred_offset: np.ndarray,
    peaks: Optional[List[Tuple[float, float, float]]] = None,
    quiver_step: int = 8,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 14),
    dpi: int = 150,
) -> plt.Figure:
    """
    Render a comprehensive 4-panel diagnostic figure for Center Offset Voting:
    1. Input astronomical image (LeakySB stretch).
    2. Semantic foreground probability map.
    3. Center centroid keypoint heatmap with detected peaks.
    4. Offset vector displacement field (quiver) overlaid on foreground.

    Parameters
    ----------
    stretched : np.ndarray
        2D normalized image [0, 1].
    pred_semantic : np.ndarray
        2D foreground probability [0, 1].
    pred_center : np.ndarray
        2D center probability heatmap [0, 1].
    pred_offset : np.ndarray
        Offset vector field of shape (2, H, W) where [0] is delta_y and [1] is delta_x.
    peaks : List of (cy, cx, score), optional
        Detected center peak coordinates.
    quiver_step : int, default 8
        Subsampling interval for rendering quiver arrows.
    title : str, optional
    save_path : str or Path, optional
    figsize : Tuple[int, int], default (16, 14)
    dpi : int, default 150
    """
    h, w = stretched.shape
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)

    # Panel (0, 0): Stretched Input
    im0 = axes[0, 0].imshow(stretched, cmap="gray", origin="lower")
    axes[0, 0].set_title(f"Astronomical Exposure (LeakySB)\nDimension: {w} × {h} px", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Pixel Y", fontsize=10)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Normalized Flux")

    # Panel (0, 1): Semantic Foreground
    im1 = axes[0, 1].imshow(pred_semantic, cmap="viridis", origin="lower", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Semantic Foreground Probability", fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Probability $P_{fg}$")

    # Panel (1, 0): Center Heatmap + Peaks
    im2 = axes[1, 0].imshow(pred_center, cmap="magma", origin="lower", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Center Centroid Keypoint Heatmap", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Pixel X", fontsize=10)
    axes[1, 0].set_ylabel("Pixel Y", fontsize=10)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Keypoint Heatmap $P_{ctr}$")

    if peaks:
        for cy, cx, sc in peaks:
            axes[1, 0].plot(cx, cy, "c+", markersize=8, markeredgewidth=1.5)
            axes[1, 0].plot(cx, cy, "co", markersize=8, fillstyle="none", markeredgewidth=1.0)

    # Panel (1, 1): Offset Vector Field (Quiver)
    axes[1, 1].imshow(pred_semantic, cmap="Blues", origin="lower", alpha=0.35)
    y_sub, x_sub = np.mgrid[0:h:quiver_step, 0:w:quiver_step]
    dy_sub = pred_offset[0, ::quiver_step, ::quiver_step]
    dx_sub = pred_offset[1, ::quiver_step, ::quiver_step]
    mask_sub = pred_semantic[::quiver_step, ::quiver_step] > 0.30

    if np.any(mask_sub):
        axes[1, 1].quiver(
            x_sub[mask_sub],
            y_sub[mask_sub],
            dx_sub[mask_sub],
            dy_sub[mask_sub],
            color="#ff0055",
            scale=50,
            width=0.003,
            headwidth=3.5,
            headlength=4,
            pivot="tail",
        )
    axes[1, 1].set_title(
        f"2D Offset Vector Field $(\\Delta y, \\Delta x)$\nPointing toward source centroids",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Pixel X", fontsize=10)

    for ax in axes.flat:
        ax.grid(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig


def plot_center_offset_segmentation(
    stretched: np.ndarray,
    pred_semantic: Optional[np.ndarray] = None,
    pred_center: Optional[np.ndarray] = None,
    pred_offset: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
    fg_thresh: float = 0.50,
    center_thresh: float = 0.30,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 150,
    cmap_img: str = "gray",
    image_title: str = "Astronomical Exposure (LeakySB Stretch)",
    mask_title: str = "Segmented Mask (Center Offset Voting)",
    max_display_size: int = 2048,
) -> plt.Figure:
    """
    Render a high-contrast side-by-side 2-panel figure showing:
    1. LeakySB-stretched astronomical exposure.
    2. Segmented mask derived directly from Center Offset Voting model outcomes.

    Memory-efficient & fast: downsamples display rasterization if max(H, W) > max_display_size,
    avoids multi-gigabyte mask materialization, and uses hue-cycling for scalable multi-instance display.

    Parameters
    ----------
    stretched : np.ndarray
        2D normalized [0, 1] astronomical exposure.
    pred_semantic : np.ndarray, optional
        Predicted foreground probability map in [0, 1].
    pred_center : np.ndarray, optional
        Predicted center keypoint probability heatmap.
    pred_offset : np.ndarray, optional
        Predicted 2D displacement vector field of shape (2, H, W).
    pred_mask : np.ndarray, optional
        Precomputed 2D integer segmentation mask. If None, derived directly from
        (pred_semantic, pred_center, pred_offset) using ray-aligned offset voting.
    fg_thresh : float, default 0.50
        Foreground probability threshold.
    center_thresh : float, default 0.30
        Center keypoint peak detection threshold.
    title : str, optional
        Figure suptitle.
    save_path : str or Path, optional
    figsize : Tuple[int, int], default (16, 8)
    dpi : int, default 150
    cmap_img : str, default 'gray'
    image_title : str, default 'Astronomical Exposure (LeakySB Stretch)'
    mask_title : str, default 'Segmented Mask (Center Offset Voting)'
    max_display_size : int, default 2048
        Maximum dimension for Matplotlib display rasterization to avoid memory spikes
        on 4k+ frames (e.g. 4088x4088). Pixel coordinate axes retain full resolution.

    Returns
    -------
    plt.Figure
    """
    if pred_mask is None:
        if pred_center is not None and pred_offset is not None and pred_semantic is not None:
            from asteria.inference.center_offset import deblend_center_offset

            seg_mask, _, _, _, _, _ = deblend_center_offset(
                pred_semantic=pred_semantic,
                pred_center=pred_center,
                pred_offsets=pred_offset,
                fg_thresh=fg_thresh,
                center_thresh=center_thresh,
                return_masks=False,
            )
            if not np.any(seg_mask > 0) and np.any(pred_semantic >= fg_thresh):
                seg_mask = (pred_semantic >= fg_thresh).astype(np.int32)
        elif pred_semantic is not None:
            seg_mask = (pred_semantic >= fg_thresh).astype(np.int32)
        else:
            raise ValueError(
                "Must supply either pred_mask, or pred_semantic (with optional pred_center and pred_offset)."
            )
    else:
        seg_mask = np.asarray(pred_mask, dtype=np.int32)

    h, w = stretched.shape
    total_px = h * w
    unique_ids = np.unique(seg_mask)
    num_instances = int(np.sum(unique_ids > 0))
    fg_px = int(np.sum(seg_mask > 0))
    pct_fg = 100.0 * fg_px / max(1, total_px)

    # Downsample display arrays if dimension exceeds max_display_size
    step = int(np.ceil(max(h, w) / max_display_size)) if max_display_size > 0 else 1
    if step > 1:
        disp_img = stretched[::step, ::step]
        disp_mask = seg_mask[::step, ::step]
    else:
        disp_img = stretched
        disp_mask = seg_mask

    extent = [-0.5, w - 0.5, -0.5, h - 0.5]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    # Panel 1: Stretched Image
    im0 = axes[0].imshow(disp_img, cmap=cmap_img, origin="lower", extent=extent)
    axes[0].set_title(f"{image_title}\nDimension: {w} × {h} px", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Pixel X", fontsize=10)
    axes[0].set_ylabel("Pixel Y", fontsize=10)
    axes[0].grid(False)
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Normalized SB Flux [0, 1]", fontsize=9)

    # Panel 2: Segmented Mask
    if num_instances > 1 or (num_instances == 1 and np.max(unique_ids) > 1):
        n_colors = 64
        rng = np.random.RandomState(42)
        colors = [[0.043, 0.051, 0.106, 1.0]]  # dark navy background #0b0d1b
        for _ in range(n_colors):
            c = list(0.3 + 0.7 * rng.rand(3)) + [1.0]
            colors.append(c)
        cmap_inst = ListedColormap(colors)
        norm_inst = BoundaryNorm(np.arange(-0.5, n_colors + 1.5, 1), cmap_inst.N)

        plot_mask = np.where(disp_mask > 0, ((disp_mask - 1) % n_colors) + 1, 0)
        axes[1].imshow(plot_mask, cmap=cmap_inst, norm=norm_inst, origin="lower", extent=extent)
        axes[1].set_title(
            f"{mask_title}\nDeblended: {num_instances} Sources | Coverage: {pct_fg:.2f}% ({fg_px:,} px)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
    else:
        cmap_bin = ListedColormap(["#0b0d1b", "#00d4ff"])
        norm_bin = BoundaryNorm([-0.5, 0.5, 1.5], 2)
        axes[1].imshow(disp_mask, cmap=cmap_bin, norm=norm_bin, origin="lower", extent=extent)
        axes[1].set_title(
            f"{mask_title}\nForeground: {pct_fg:.2f}% ({fg_px:,} px) | Sky: {100.0 - pct_fg:.2f}%",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        patches = [
            mpatches.Patch(color="#0b0d1b", label="Sky Background (0)"),
            mpatches.Patch(color="#00d4ff", label="Source Foreground (1)"),
        ]
        axes[1].legend(
            handles=patches,
            loc="upper right",
            framealpha=0.85,
            facecolor="#1c2038",
            edgecolor="#444b6e",
            fontsize=9,
            labelcolor="white",
        )

    axes[1].set_xlabel("Pixel X", fontsize=10)
    axes[1].grid(False)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")

    return fig
