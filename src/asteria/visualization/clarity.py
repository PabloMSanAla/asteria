"""
Pixel Clarity interactive visualization routines for Asteria astronomical exposures.
=====================================================================================

Provides high-performance interactive display of astronomical exposures with overlaid
segmentation masks and deblending boundaries using Pixel Clarity (pixel-clarity).
"""

from __future__ import annotations

import atexit
import logging
from pathlib import Path
import tempfile
from typing import Any, List, Optional, Tuple, Union

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

try:
    import pixel_clarity as pc
    HAS_PIXEL_CLARITY = True
except ImportError:
    pc = None
    HAS_PIXEL_CLARITY = False

logger = logging.getLogger(__name__)


def create_mask_edge_overlay(
    stretched: np.ndarray,
    pred_mask: np.ndarray,
    mask_color: Tuple[float, float, float] = (0.0, 0.83, 1.0),  # Electric Cyan
    edge_color: Tuple[float, float, float] = (1.0, 0.2, 0.4),   # Vivid Coral Red
    mask_alpha: float = 0.45,
    edge_alpha: float = 0.80,
) -> np.ndarray:
    """
    Construct a high-contrast RGB overlay combining a stretched astronomical exposure
    with segmentation mask interiors and deblending boundary edges.

    Parameters
    ----------
    stretched : np.ndarray
        2D normalized [0.0, 1.0] astronomical exposure (H, W).
    pred_mask : np.ndarray
        2D integer segmentation mask (H, W):
        - Class 0: Sky background
        - Class 1: Source interior (mask)
        - Class 2: Deblending edge (boundary)
        Alternatively, arbitrary multi-instance integer label map (where 0 is background).
    mask_color : Tuple[float, float, float], default (0.0, 0.83, 1.0)
        RGB color for source interior masks.
    edge_color : Tuple[float, float, float], default (1.0, 0.2, 0.4)
        RGB color for deblending edges.
    mask_alpha : float, default 0.45
        Opacity for interior masks in [0.0, 1.0].
    edge_alpha : float, default 0.80
        Opacity for boundary edges in [0.0, 1.0].

    Returns
    -------
    np.ndarray
        RGB float32 image array of shape (H, W, 3) in [0.0, 1.0].
    """
    h, w = stretched.shape
    base_gray = np.clip(np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    overlay_rgb = np.repeat(base_gray[:, :, np.newaxis], 3, axis=2).astype(np.float32)

    c_mask = np.array(mask_color, dtype=np.float32)
    c_edge = np.array(edge_color, dtype=np.float32)

    unique_vals = np.unique(pred_mask)

    # Case A: Standard Asteria 3-class semantic mask (0: Sky, 1: Core, 2: Edge)
    if set(unique_vals).issubset({0, 1, 2}):
        m_interior = pred_mask == 1
        m_edges = pred_mask == 2

        # If only binary mask was provided without edge class, extract morphology edges
        if not np.any(m_edges) and np.any(m_interior):
            dilated = ndimage.binary_dilation(m_interior, iterations=1)
            m_edges = dilated & (~m_interior)

        if np.any(m_interior):
            overlay_rgb[m_interior] = (1.0 - mask_alpha) * overlay_rgb[m_interior] + mask_alpha * c_mask

        if np.any(m_edges):
            overlay_rgb[m_edges] = (1.0 - edge_alpha) * overlay_rgb[m_edges] + edge_alpha * c_edge

    # Case B: Multi-instance labeled map (1..N instances)
    else:
        nonzero = pred_mask > 0
        if np.any(nonzero):
            # Compute instance boundaries via gradient filter
            grad_y = np.abs(np.diff(pred_mask, axis=0, prepend=pred_mask[:1, :])) > 0
            grad_x = np.abs(np.diff(pred_mask, axis=1, prepend=pred_mask[:, :1])) > 0
            inst_edges = (grad_y | grad_x) & nonzero

            # Source interior is nonzero minus edges
            inst_interior = nonzero & (~inst_edges)

            overlay_rgb[inst_interior] = (1.0 - mask_alpha) * overlay_rgb[inst_interior] + mask_alpha * c_mask
            overlay_rgb[inst_edges] = (1.0 - edge_alpha) * overlay_rgb[inst_edges] + edge_alpha * c_edge

    return np.clip(overlay_rgb, 0.0, 1.0)


def _prepare_mask_bits(pred_mask: np.ndarray) -> np.ndarray:
    """
    Convert segmentation mask into bitmask flags for Pixel Clarity MASK HDU:
    - Bit 0 (value 1): Source interior (core mask)
    - Bit 1 (value 2): Deblending boundary edge
    """
    h, w = pred_mask.shape
    mask_bits = np.zeros((h, w), dtype=np.uint8)
    unique_vals = np.unique(pred_mask)

    if set(unique_vals).issubset({0, 1, 2}):
        m_interior = pred_mask == 1
        m_edges = pred_mask == 2
        if not np.any(m_edges) and np.any(m_interior):
            dilated = ndimage.binary_dilation(m_interior, iterations=1)
            m_edges = dilated & (~m_interior)
        mask_bits[m_interior] |= 1
        mask_bits[m_edges] |= 2
    else:
        # Multi-instance label map
        nonzero = pred_mask > 0
        if np.any(nonzero):
            grad_y = np.abs(np.diff(pred_mask, axis=0, prepend=pred_mask[:1, :])) > 0
            grad_x = np.abs(np.diff(pred_mask, axis=1, prepend=pred_mask[:, :1])) > 0
            inst_edges = (grad_y | grad_x) & nonzero
            inst_interior = nonzero & (~inst_edges)
            mask_bits[inst_interior] |= 1
            mask_bits[inst_edges] |= 2

    return mask_bits


def _prepare_mask_fits(
    stretched: np.ndarray,
    pred_mask: np.ndarray,
) -> Path:
    """
    Serialize stretched exposure (PrimaryHDU) and segmentation bitmask (ImageHDU named 'MASK')
    into a temporary multi-extension FITS file for Pixel Clarity interactive layer display.
    """
    mask_bits = _prepare_mask_bits(pred_mask)
    hdul = fits.HDUList([
        fits.PrimaryHDU(data=stretched.astype(np.float32)),
        fits.ImageHDU(data=mask_bits, name="MASK"),
    ])
    tmp = tempfile.NamedTemporaryFile(prefix="asteria_mask_layer_", suffix=".fits", delete=False)
    fits_path = Path(tmp.name)
    tmp.close()
    hdul.writeto(str(fits_path), overwrite=True)
    atexit.register(lambda: fits_path.unlink(missing_ok=True))
    return fits_path


def _prepare_overlay_png(
    stretched: np.ndarray,
    pred_mask: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    **overlay_kwargs,
) -> Tuple[np.ndarray, Path]:
    """
    Generate high-contrast RGB overlay and serialize to PNG on disk.
    Pixel Clarity requires RGB images to be stored as image files (e.g. PNG)
    rather than 3D numpy arrays so dimensions are correctly interpreted.
    """
    overlay = create_mask_edge_overlay(stretched=stretched, pred_mask=pred_mask, **overlay_kwargs)
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            png_path = p.with_suffix(".png")
        else:
            png_path = p
        plt.imsave(str(png_path), overlay)
        if png_path != p:
            plt.imsave(str(p), overlay)
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="asteria_overlay_", suffix=".png", delete=False)
        png_path = Path(tmp.name)
        tmp.close()
        plt.imsave(str(png_path), overlay)
        atexit.register(lambda: png_path.unlink(missing_ok=True))

    return overlay, png_path


def show_pixel_clarity(
    stretched: np.ndarray,
    pred_mask: np.ndarray,
    title: Optional[str] = None,
    layout: Optional[str] = None,
    height: int = 600,
    save_path: Optional[Union[str, Path]] = None,
    mode: str = "layer",
    side_by_side: bool = False,
    blink: bool = False,
    **settings,
) -> Any:
    """
    Display astronomical exposure and overlaid segmentation mask/edges using Pixel Clarity.
    Optimized for interactive Jupyter Notebook cells.

    Parameters
    ----------
    stretched : np.ndarray
        2D normalized astronomical exposure.
    pred_mask : np.ndarray
        2D segmentation mask (0=Sky, 1=Core, 2=Edge, or instance labels).
    title : str, optional
        Viewer title.
    layout : {"grid", "pixel", "sky"}, optional
        Multi-image layout if side_by_side is True.
    height : int, default 600
        Viewer widget height in pixels.
    save_path : str or Path, optional
        Optional path to save the overlay image to disk.
    mode : {"layer", "composite"}, default "layer"
        Display mode:
        - "layer": Mounts mask as an interactive overlay layer on top of the image
          (can be turned on/off and opacity adjusted via the viewer's sidebar).
        - "composite": Static baked RGB composite overlay.
    side_by_side : bool, default False
        If True, displays both the overlay and the raw stretched exposure side-by-side.
    blink : bool, default False
        If True, enables rapid blinking between overlay and raw exposure (Spacebar).
    **settings
        Additional settings forwarded to pixel_clarity.show.

    Returns
    -------
    pixel_clarity.Viewer
        Interactive widget viewer instance.
    """
    if not HAS_PIXEL_CLARITY:
        raise ImportError(
            "The 'pixel-clarity' package is required for Pixel Clarity visualization. "
            "Install it via: pip install pixel-clarity"
        )

    title_str = title or "Asteria UNet++: Stretched Exposure & Mask Layer"

    if save_path:
        # Always write out the RGB composite overlay to save_path if requested
        _prepare_overlay_png(stretched=stretched, pred_mask=pred_mask, save_path=save_path)

    if side_by_side or layout == "grid":
        _, png_path = _prepare_overlay_png(stretched=stretched, pred_mask=pred_mask)
        data = [str(png_path), stretched.astype(np.float32)]
        names = ["Mask & Edge Overlay", "Stretched Exposure"]
        settings.setdefault("layout", "grid")
    elif blink:
        _, png_path = _prepare_overlay_png(stretched=stretched, pred_mask=pred_mask)
        data = [str(png_path), stretched.astype(np.float32)]
        names = ["Mask & Edge Overlay", "Stretched Exposure"]
        settings.setdefault("blink", True)
    elif mode == "composite":
        _, png_path = _prepare_overlay_png(stretched=stretched, pred_mask=pred_mask)
        data = str(png_path)
        names = ["Mask & Edge Overlay"]
    else:
        # Default: Native interactive overlay layer in FITS container
        fits_path = _prepare_mask_fits(stretched=stretched, pred_mask=pred_mask)
        data = str(fits_path)
        names = [title_str]

    viewer = pc.show(
        data,
        names=names,
        title=title_str,
        height=height,
        **settings,
    )
    return viewer


def open_pixel_clarity(
    stretched: np.ndarray,
    pred_mask: np.ndarray,
    title: Optional[str] = None,
    layout: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    mode: str = "layer",
    side_by_side: bool = False,
    blink: bool = False,
    **settings,
) -> Any:
    """
    Open interactive Pixel Clarity visualization in a standalone browser tab.
    Displays the astronomical exposure with high-contrast segmentation mask and edges
    mounted as an interactive overlay layer that can be turned on/off and opacity-adjusted.

    Parameters
    ----------
    stretched : np.ndarray
        2D normalized astronomical exposure.
    pred_mask : np.ndarray
        2D segmentation mask (0=Sky, 1=Core, 2=Edge, or instance labels).
    title : str, optional
        Browser window/tab title.
    layout : {"grid", "pixel"}, optional
        Multi-image layout if side_by_side is True.
    save_path : str or Path, optional
        Optional path to save the overlay image to disk.
    mode : {"layer", "composite"}, default "layer"
        Display mode:
        - "layer": Mounts mask as an interactive overlay layer on top of the image
          (can be turned on/off and opacity adjusted via the viewer's sidebar).
        - "composite": Static baked RGB composite overlay.
    side_by_side : bool, default False
        If True, displays both the overlay and the raw stretched exposure side-by-side.
    blink : bool, default False
        If True, enables rapid blinking between overlay and raw exposure (Spacebar).
    **settings
        Additional settings forwarded to pixel_clarity.popup.

    Returns
    -------
    pixel_clarity.Popup
        Interactive popup browser tab controller.
    """
    if not HAS_PIXEL_CLARITY:
        raise ImportError(
            "The 'pixel-clarity' package is required for Pixel Clarity visualization. "
            "Install it via: pip install pixel-clarity"
        )

    title_str = title or "Asteria UNet++: Stretched Exposure & Mask Layer"

    if save_path:
        # Always write out the RGB composite overlay to save_path if requested
        _prepare_overlay_png(stretched=stretched, pred_mask=pred_mask, save_path=save_path)

    if side_by_side or layout == "grid":
        _, png_path = _prepare_overlay_png(stretched=stretched, pred_mask=pred_mask)
        data = [str(png_path), stretched.astype(np.float32)]
        names = ["Mask & Edge Overlay", "Stretched Exposure"]
        settings.setdefault("layout", "grid")
    elif blink:
        _, png_path = _prepare_overlay_png(stretched=stretched, pred_mask=pred_mask)
        data = [str(png_path), stretched.astype(np.float32)]
        names = ["Mask & Edge Overlay", "Stretched Exposure"]
        settings.setdefault("blink", True)
    elif mode == "composite":
        _, png_path = _prepare_overlay_png(stretched=stretched, pred_mask=pred_mask)
        data = str(png_path)
        names = ["Mask & Edge Overlay"]
    else:
        # Default: Native interactive overlay layer in FITS container
        fits_path = _prepare_mask_fits(stretched=stretched, pred_mask=pred_mask)
        data = str(fits_path)
        names = [title_str]

    popup = pc.popup(
        data,
        names=names,
        title=title_str,
        **settings,
    )
    return popup


def _prepare_starnet_mef(
    raw_image: Union[np.ndarray, Any],
    starless_flux: np.ndarray,
    header: Optional[Union[dict, fits.Header]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create a 2-HDU multi-extension FITS file containing:
    1. PrimaryHDU: Original raw exposure data (not stretched, physical flux).
    2. ImageHDU ('STARLESS'): Starless prediction with inverse LeakySB applied.
    """
    if hasattr(raw_image, "data") and not isinstance(raw_image, np.ndarray):
        raw_arr = np.asarray(raw_image.data, dtype=np.float32)
        if header is None and hasattr(raw_image, "header"):
            header = raw_image.header
    else:
        raw_arr = np.asarray(raw_image, dtype=np.float32)

    def _make_fits_header(h: Optional[Union[dict, fits.Header]]) -> fits.Header:
        if h is None:
            return fits.Header()
        if isinstance(h, fits.Header):
            return h.copy()
        fh = fits.Header()
        for k, v in h.items():
            if isinstance(k, str) and len(k) <= 8 and isinstance(v, (int, float, str, bool)):
                fh[k] = v
        return fh

    hdr_orig = _make_fits_header(header)
    hdr_orig["EXTNAME"] = "ORIGINAL"

    hdr_starless = _make_fits_header(header)
    hdr_starless["EXTNAME"] = "STARLESS"

    hdul = fits.HDUList([
        fits.PrimaryHDU(data=raw_arr.astype(np.float32), header=hdr_orig),
        fits.ImageHDU(data=np.asarray(starless_flux, dtype=np.float32), header=hdr_starless, name="STARLESS"),
    ])

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        hdul.writeto(str(out_p), overwrite=True)
        return out_p

    tmp = tempfile.NamedTemporaryFile(prefix="asteria_starnet_mef_", suffix=".fits", delete=False)
    fits_path = Path(tmp.name)
    tmp.close()
    hdul.writeto(str(fits_path), overwrite=True)
    atexit.register(lambda: fits_path.unlink(missing_ok=True))
    return fits_path


def show_starnet_clarity(
    raw_image: Union[np.ndarray, Any],
    starless_flux: np.ndarray,
    title: Optional[str] = None,
    header: Optional[fits.Header] = None,
    height: int = 600,
    save_path: Optional[Union[str, Path]] = None,
    **settings,
) -> Any:
    """
    Display interactive Pixel Clarity widget in Jupyter Notebook for StarNet3 star removal.
    Mounts a 2-HDU multi-extension FITS file:
    - HDU 0 (PrimaryHDU): Raw input astronomical exposure (physical flux, not stretched).
    - HDU 1 (ImageHDU): Starless prediction with inverse LeakySB applied (matching physical flux scale).

    Parameters
    ----------
    raw_image : np.ndarray or AstroImage
        Original astronomical image array (un-stretched physical flux).
    starless_flux : np.ndarray
        Starless prediction reconstructed into physical flux via inverse LeakySB.
    title : str, optional
        Viewer title.
    header : fits.Header, optional
        FITS header with WCS/metadata.
    height : int, default 600
        Viewer widget height in pixels.
    save_path : str or Path, optional
        Optional destination to save the multi-extension FITS file.
    **settings
        Additional options forwarded to pixel_clarity.show.

    Returns
    -------
    pixel_clarity.Viewer
        Interactive widget viewer instance.
    """
    if not HAS_PIXEL_CLARITY:
        raise ImportError(
            "The 'pixel-clarity' package is required for Pixel Clarity visualization. "
            "Install it via: pip install pixel-clarity"
        )

    title_str = title or "StarNet3: Raw Exposure & Inverted Starless Prediction"
    fits_path = _prepare_starnet_mef(
        raw_image=raw_image,
        starless_flux=starless_flux,
        header=header,
        save_path=save_path,
    )

    viewer = pc.show(
        str(fits_path),
        names=[title_str],
        title=title_str,
        height=height,
        **settings,
    )
    return viewer


def open_starnet_clarity(
    raw_image: Union[np.ndarray, Any],
    starless_flux: np.ndarray,
    title: Optional[str] = None,
    header: Optional[fits.Header] = None,
    save_path: Optional[Union[str, Path]] = None,
    **settings,
) -> Any:
    """
    Open interactive Pixel Clarity visualization in a standalone browser tab for StarNet3.
    Displays a 2-HDU multi-extension FITS file:
    - HDU 0 (PrimaryHDU): Raw input astronomical exposure (physical flux, not stretched).
    - HDU 1 (ImageHDU): Starless prediction with inverse LeakySB applied (matching physical flux scale).

    Parameters
    ----------
    raw_image : np.ndarray or AstroImage
        Original astronomical image array (un-stretched physical flux).
    starless_flux : np.ndarray
        Starless prediction reconstructed into physical flux via inverse LeakySB.
    title : str, optional
        Browser window/tab title.
    header : fits.Header, optional
        FITS header with WCS/metadata.
    save_path : str or Path, optional
        Optional destination to save the multi-extension FITS file.
    **settings
        Additional options forwarded to pixel_clarity.popup.

    Returns
    -------
    pixel_clarity.Popup
        Interactive popup browser tab controller.
    """
    if not HAS_PIXEL_CLARITY:
        raise ImportError(
            "The 'pixel-clarity' package is required for Pixel Clarity visualization. "
            "Install it via: pip install pixel-clarity"
        )

    title_str = title or "StarNet3: Raw Exposure & Inverted Starless Prediction"
    fits_path = _prepare_starnet_mef(
        raw_image=raw_image,
        starless_flux=starless_flux,
        header=header,
        save_path=save_path,
    )

    popup = pc.popup(
        str(fits_path),
        names=[title_str],
        title=title_str,
        **settings,
    )
    return popup
