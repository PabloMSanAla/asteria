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
