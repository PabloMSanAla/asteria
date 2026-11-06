"""
Astronomical surface brightness photometric transformations.
Features the Leaky Surface Brightness (leakySB) non-linear stretch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union, Tuple, Any
import numpy as np
import torch

ArrayOrTensor = Union[np.ndarray, torch.Tensor]


def _norm_leaky_sb_forward_1d(
    f_norm: torch.Tensor,
    span: float,
    negative_slope: float,
    positive_slope: float,
    y_zero: float = 0.015,
    y_one: float = 0.96,
) -> torch.Tensor:
    """
    Bijective 1D transformation mapping all fluxes (-inf, +inf) into (0.0, 1.0).
    `y_zero = 0.015` maintains sky background deep black.
    """
    mid_span = y_one - y_zero
    k_tangent = (2.5 / np.log(10.0)) / span
    top_room = 1.0 - y_one

    f_safe = torch.clamp(f_norm, min=1.0)
    u = (2.5 * torch.log10(f_safe)) / span

    # Normal log regime [mmax, mmin] -> [y_zero, y_one]
    y_mid = y_zero + mid_span * u

    # Bright regime (< mmin) -> (y_one, 1.0)
    s_pos = (positive_slope * mid_span) / max(top_room, 1e-8)
    u_excess = torch.clamp(u - 1.0, min=0.0)
    y_high = 1.0 - (top_room / (1.0 + s_pos * u_excess))
    y_pos_branch = torch.where(u <= 1.0, y_mid, y_high)

    # Faint and negative regime (> mmax and fluxes < 0) -> (0.0, y_zero)
    s_neg = (negative_slope * k_tangent * mid_span) / max(y_zero, 1e-8)
    f_deficit = torch.clamp(1.0 - f_norm, min=0.0)
    y_low = y_zero / (1.0 + s_neg * f_deficit)

    return torch.where(f_norm >= 1.0, y_pos_branch, y_low)


def _norm_leaky_sb_inverse_1d(
    y: torch.Tensor,
    span: float,
    negative_slope: float,
    positive_slope: float,
    y_zero: float = 0.015,
    y_one: float = 0.96,
) -> torch.Tensor:
    """Exact analytical inverse of `_norm_leaky_sb_forward_1d`."""
    mid_span = y_one - y_zero
    k_tangent = (2.5 / np.log(10.0)) / span
    top_room = 1.0 - y_one

    s_pos = (positive_slope * mid_span) / max(top_room, 1e-8)
    s_neg = (negative_slope * k_tangent * mid_span) / max(y_zero, 1e-8)

    u_mid = (y - y_zero) / mid_span
    y_high_safe = torch.clamp(1.0 - y, min=1e-14)
    u_high = 1.0 + ((top_room / y_high_safe) - 1.0) / s_pos

    u = torch.where(y <= y_one, u_mid, u_high)
    f_pos_branch = 10.0 ** ((u * span) / 2.5)

    y_low_safe = torch.clamp(y, min=1e-14)
    f_low = 1.0 - (((y_zero / y_low_safe) - 1.0) / s_neg)

    return torch.where(y >= y_zero, f_pos_branch, f_low)


def _soft_box_forward(x: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
    """
    Smoothly compress any chrominance/intensity outside [margin, 1 - margin]
    into (0.0, 1.0) bijectively.
    """
    low_lim = margin
    high_lim = 1.0 - margin

    x_low = low_lim / (1.0 + (low_lim - torch.clamp(x, max=low_lim)) / low_lim)
    x_high = 1.0 - margin / (1.0 + (torch.clamp(x, min=high_lim) - high_lim) / margin)

    out = torch.where(x < low_lim, x_low, x)
    out = torch.where(x > high_lim, x_high, out)
    return out


def _soft_box_inverse(y: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
    """Exact inverse of `_soft_box_forward`."""
    low_lim = margin
    high_lim = 1.0 - margin

    y_low_safe = torch.clamp(y, min=1e-14)
    x_low = low_lim - low_lim * ((low_lim / y_low_safe) - 1.0)

    y_high_safe = torch.clamp(1.0 - y, min=1e-14)
    x_high = high_lim + margin * ((margin / y_high_safe) - 1.0)

    out = torch.where(y < low_lim, x_low, y)
    out = torch.where(y > high_lim, x_high, out)
    return out


def _chroma_weight(
    lum_n: torch.Tensor,
    y_lum: torch.Tensor,
    span: float,
    negative_slope: float,
    y_zero: float,
    color_boost: float,
) -> torch.Tensor:
    """
    Strictly positive chrominance weight preserving true color proportions (Lupton-style).
    """
    k_tangent = (2.5 / np.log(10.0)) / span
    slope_neg = negative_slope * k_tangent
    w_base = torch.where(
        lum_n >= 1.0,
        ((y_lum - y_zero) + slope_neg) / torch.clamp(lum_n, min=1.0),
        torch.full_like(lum_n, slope_neg),
    )
    return w_base * color_boost


def leakySB(
    r: ArrayOrTensor,
    g: Optional[ArrayOrTensor] = None,
    b: Optional[ArrayOrTensor] = None,
    zero_points: float | Sequence[float] = 31.4,
    pixel_scale: float | Sequence[float] = 0.262,
    mmin: float = 13.0,
    mmax: float = 32.0,
    negative_slope: float = 0.1,
    positive_slope: float = 0.05,
    color_boost: float = 1.2,
    preserve_color_ratios: bool = True,
    y_zero: float = 0.015,
    y_one: float = 0.96,
    channels_last: bool = True,
    output_dtype: torch.dtype | None = torch.float64,
    sky: float = 0.0,
) -> ArrayOrTensor:
    """
    Convert physical FITS fluxes (R, G, B or monochrome) into normalized surface
    brightness in (0.0, 1.0), preserving dynamic range and reversible without clipping.

    Parameters
    ----------
    r : np.ndarray or torch.Tensor
        Red channel array, or monochrome 2D image.
    g : np.ndarray or torch.Tensor, optional
        Green channel array. Defaults to `r` if None.
    b : np.ndarray or torch.Tensor, optional
        Blue channel array. Defaults to `r` if None.
    zero_points : float or Sequence[float], default 31.4
        Photometric zero point(s).
    pixel_scale : float or Sequence[float], default 0.262
        Plate scale in arcsec/pixel.
    mmin : float, default 13.0
        Bright-end surface brightness magnitude cutoff.
    mmax : float, default 32.0
        Faint-end surface brightness magnitude cutoff.
    negative_slope : float, default 0.1
    positive_slope : float, default 0.05
    color_boost : float, default 1.2
    preserve_color_ratios : bool, default True
    y_zero : float, default 0.015
    y_one : float, default 0.96
    channels_last : bool, default True
    output_dtype : torch.dtype or None, default torch.float64
    sky : float, default 0.0
        Background sky value to subtract prior to stretching.

    Returns
    -------
    np.ndarray or torch.Tensor
        Normalized RGB image in (0.0, 1.0).
    """
    if sky != 0.0:
        r = r - float(sky)
        if g is not None and g is not r:
            g = g - float(sky)
        if b is not None and b is not r:
            b = b - float(sky)

    if g is None:
        g = r
    if b is None:
        b = r

    is_numpy = isinstance(r, np.ndarray)
    device = r.device if isinstance(r, torch.Tensor) else "cpu"
    span = float(mmax - mmin)

    zps = (zero_points,) * 3 if isinstance(zero_points, (int, float)) else tuple(zero_points)
    pss = (pixel_scale,) * 3 if isinstance(pixel_scale, (int, float)) else tuple(pixel_scale)

    norm_bands = []
    for ch, zp, ps in zip((r, g, b), zps, pss):
        ch_arr = np.asarray(ch, dtype=np.float64) if isinstance(ch, np.ndarray) else ch
        t = torch.as_tensor(ch_arr, dtype=torch.float64, device=device)
        if torch.isnan(t).any() or torch.isinf(t).any():
            finite_mask = torch.isfinite(t)
            fill_val = torch.median(t[finite_mask]) if finite_mask.any() else torch.tensor(0.0, dtype=torch.float64, device=device)
            t = torch.where(finite_mask, t, fill_val)
        f_mmax = (float(ps) ** 2) * (10.0 ** (0.4 * (float(zp) - float(mmax))))
        norm_bands.append(t / f_mmax)

    r_n, g_n, b_n = norm_bands
    stack_dim = -1 if channels_last else 0

    if preserve_color_ratios:
        lum_n = (r_n + g_n + b_n) / 3.0
        y_lum = _norm_leaky_sb_forward_1d(
            lum_n, span, negative_slope, positive_slope, y_zero=y_zero, y_one=y_one
        )
        w = _chroma_weight(lum_n, y_lum, span, negative_slope, y_zero, color_boost)

        r_raw = y_lum + w * (r_n - lum_n)
        g_raw = y_lum + w * (g_n - lum_n)
        b_raw = y_lum + w * (b_n - lum_n)

        rgb = torch.stack(
            [
                _soft_box_forward(r_raw),
                _soft_box_forward(g_raw),
                _soft_box_forward(b_raw),
            ],
            dim=stack_dim,
        )
    else:
        rgb = torch.stack(
            [
                _norm_leaky_sb_forward_1d(r_n, span, negative_slope, positive_slope, y_zero, y_one),
                _norm_leaky_sb_forward_1d(g_n, span, negative_slope, positive_slope, y_zero, y_one),
                _norm_leaky_sb_forward_1d(b_n, span, negative_slope, positive_slope, y_zero, y_one),
            ],
            dim=stack_dim,
        )

    if output_dtype is not None:
        rgb = rgb.to(output_dtype)

    return rgb.detach().cpu().numpy() if is_numpy else rgb


def leakySB_inverse(
    rgb: ArrayOrTensor,
    zero_points: float | Sequence[float] = 31.4,
    pixel_scale: float | Sequence[float] = 0.262,
    mmin: float = 13.0,
    mmax: float = 32.0,
    negative_slope: float = 0.1,
    positive_slope: float = 0.05,
    color_boost: float = 1.2,
    preserve_color_ratios: bool = True,
    y_zero: float = 0.015,
    y_one: float = 0.96,
) -> tuple[ArrayOrTensor, ArrayOrTensor, ArrayOrTensor]:
    """Exact inverse of `leakySB` recovering original physical fluxes."""
    is_numpy = isinstance(rgb, np.ndarray)
    rgb_arr = np.asarray(rgb, dtype=np.float64) if isinstance(rgb, np.ndarray) else rgb
    t = torch.as_tensor(rgb_arr, dtype=torch.float64)
    span = float(mmax - mmin)

    if t.ndim == 3 and t.shape[-1] == 3:
        r_box, g_box, b_box = t[..., 0], t[..., 1], t[..., 2]
    elif t.ndim == 3 and t.shape[0] == 3:
        r_box, g_box, b_box = t[0], t[1], t[2]
    else:
        raise ValueError(f"Expected 3D RGB image, got {t.shape}")

    zps = (zero_points,) * 3 if isinstance(zero_points, (int, float)) else tuple(zero_points)
    pss = (pixel_scale,) * 3 if isinstance(pixel_scale, (int, float)) else tuple(pixel_scale)

    if preserve_color_ratios:
        r_raw = _soft_box_inverse(r_box)
        g_raw = _soft_box_inverse(g_box)
        b_raw = _soft_box_inverse(b_box)

        y_lum = (r_raw + g_raw + b_raw) / 3.0
        lum_n = _norm_leaky_sb_inverse_1d(
            y_lum, span, negative_slope, positive_slope, y_zero=y_zero, y_one=y_one
        )
        w = _chroma_weight(lum_n, y_lum, span, negative_slope, y_zero, color_boost)
        r_n = lum_n + (r_raw - y_lum) / w
        g_n = lum_n + (g_raw - y_lum) / w
        b_n = lum_n + (b_raw - y_lum) / w
        norm_bands = (r_n, g_n, b_n)
    else:
        norm_bands = tuple(
            _norm_leaky_sb_inverse_1d(ch, span, negative_slope, positive_slope, y_zero, y_one)
            for ch in (r_box, g_box, b_box)
        )

    orig_bands = []
    for ch_n, zp, ps in zip(norm_bands, zps, pss):
        f_mmax = (float(ps) ** 2) * (10.0 ** (0.4 * (float(zp) - float(mmax))))
        orig_flux = (ch_n * f_mmax).to(torch.float32)
        orig_bands.append(orig_flux.detach().cpu().numpy() if is_numpy else orig_flux)

    return orig_bands[0], orig_bands[1], orig_bands[2]


def leaky_surface_brightness(
    data: np.ndarray,
    zero_points: float = 31.4,
    pixel_scale: float = 0.11,
    mmin: float = 13.0,
    mmax: float = 32.0,
    negative_slope: float = 0.1,
    positive_slope: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """
    Apply Leaky Surface Brightness (leakySB) non-linear transformation.

    Preserves dynamic range across bright stellar cores while lifting faint
    diffuse galaxy wings, with a linear leak below sky background to preserve
    statistical noise characteristics without hard clipping.

    Parameters
    ----------
    data : np.ndarray
        Raw 2D astronomical image array in physical flux units.
    zero_points : float, default 31.4
        Photometric zero point magnitude (Roman Space Telescope WFI default: 31.4, or 20.472 for MJy/sr).
        Aliases: zeropoint, zero_point, zp.
    pixel_scale : float, default 0.11
        Detector plate scale in arcseconds per pixel (Roman WFI default: 0.11).
        Aliases: pixelscale, ps.
    mmin : float, default 13.0
        Bright-end surface brightness magnitude cutoff (mag / arcsec^2).
    mmax : float, default 32.0
        Faint-end surface brightness magnitude cutoff (mag / arcsec^2).
    negative_slope : float, default 0.1
        Slope for faint and negative flux values.
    positive_slope : float, default 0.05
        Slope for bright flux values.
        Aliases: leak_slope, leakslope.

    Returns
    -------
    np.ndarray
        Normalized surface brightness array in range [0.0, 1.0], float32.
    """
    # Support parameter aliases
    for k in ("zeropoint", "zero_point", "zp"):
        if k in kwargs and kwargs[k] is not None:
            zero_points = float(kwargs[k])
            break

    for k in ("pixelscale", "ps"):
        if k in kwargs and kwargs[k] is not None:
            pixel_scale = float(kwargs[k])
            break

    for k in ("leakslope", "leak_slope"):
        if k in kwargs and kwargs[k] is not None:
            positive_slope = float(kwargs[k])
            break

    if "negative_slope" in kwargs and kwargs["negative_slope"] is not None:
        negative_slope = float(kwargs["negative_slope"])
    if "positive_slope" in kwargs and kwargs["positive_slope"] is not None:
        positive_slope = float(kwargs["positive_slope"])

    img = np.array(data, dtype=np.float32, copy=True)
    valid_mask = np.isfinite(img)
    if not np.any(valid_mask):
        return np.zeros_like(img, dtype=np.float32)

    bg_val = float(np.nanmedian(img))
    img[~valid_mask] = bg_val

    rgb = leakySB(
        img,
        img,
        img,
        zero_points=zero_points,
        pixel_scale=pixel_scale,
        mmin=mmin,
        mmax=mmax,
        negative_slope=negative_slope,
        positive_slope=positive_slope,
        channels_last=True,
    )
    return np.clip(rgb[..., 0], 0.0, 1.0).astype(np.float32)


def asinh_stretch(data: np.ndarray, non_linear_factor: float = 0.005) -> np.ndarray:
    """Asinh non-linear stretch (Lupton et al. 2004)."""
    arr = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    p_low, p_high = np.percentile(arr, (1.0, 99.8))
    clipped = np.clip(arr, p_low, p_high) - p_low
    max_val = np.max(clipped) + 1e-9
    norm = clipped / max_val
    stretched = np.arcsinh(norm / non_linear_factor) / np.arcsinh(1.0 / non_linear_factor)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def linear_stretch(data: np.ndarray, pmin: float = 0.5, pmax: float = 99.5) -> np.ndarray:
    """Standard percentile linear normalization."""
    arr = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = np.percentile(arr, (pmin, pmax))
    if vmax <= vmin:
        vmax = vmin + 1e-5
    norm = (arr - vmin) / (vmax - vmin)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def stretch_image(
    data: np.ndarray,
    stretch_type: str = "leakysb",
    sky: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    Unified entry point for astronomical stretching.

    Parameters
    ----------
    data : np.ndarray
        2D astronomical array.
    stretch_type : str, default 'leakysb'
        Algorithm: 'leakysb', 'asinh', 'linear'.
    sky : float, default 0.0
        Sky background value subtracted from data prior to stretching.
    """
    # Pop any redundant 'sky' in kwargs to prevent double subtraction
    kw_sky = kwargs.pop("sky", None)
    eff_sky = sky if sky != 0.0 else (kw_sky if kw_sky is not None else 0.0)

    if eff_sky != 0.0:
        data = data - float(eff_sky)

    mode = stretch_type.lower()
    if mode in ("leakysb", "leaky_sb"):
        return leaky_surface_brightness(data, **kwargs)
    elif mode == "asinh":
        return asinh_stretch(data, **kwargs)
    elif mode == "linear":
        return linear_stretch(data, **kwargs)
    else:
        return leaky_surface_brightness(data, **kwargs)


def inverse_leaky_surface_brightness(
    data: np.ndarray,
    zero_points: float = 31.4,
    pixel_scale: float = 0.11,
    mmin: float = 13.0,
    mmax: float = 32.0,
    negative_slope: float = 0.1,
    positive_slope: float = 0.05,
    sky: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    Invert normalized Leaky Surface Brightness (leakySB) array back into physical flux.

    Reconstructs physical photon fluxes from [0.0, 1.0] stretched representations,
    adding back any subtracted background sky level.

    Parameters
    ----------
    data : np.ndarray
        Normalized image array in [0.0, 1.0] (2D monochrome or 3D CHW/HWC).
    zero_points : float, default 31.4
        Photometric zero point magnitude.
    pixel_scale : float, default 0.11
        Detector plate scale in arcsec/pixel.
    mmin : float, default 13.0
        Bright-end surface brightness magnitude cutoff.
    mmax : float, default 32.0
        Faint-end surface brightness magnitude cutoff.
    negative_slope : float, default 0.1
        Slope for faint and negative flux values.
    positive_slope : float, default 0.05
        Slope for bright flux values (leak_slope).
    sky : float, default 0.0
        Background sky level that was subtracted prior to forward stretching.

    Returns
    -------
    np.ndarray
        Physical flux image array in original input units (float32).
    """
    for k in ("zeropoint", "zero_point", "zp"):
        if k in kwargs and kwargs[k] is not None:
            zero_points = float(kwargs[k])
            break

    for k in ("pixelscale", "ps"):
        if k in kwargs and kwargs[k] is not None:
            pixel_scale = float(kwargs[k])
            break

    for k in ("leakslope", "leak_slope"):
        if k in kwargs and kwargs[k] is not None:
            positive_slope = float(kwargs[k])
            break

    if "negative_slope" in kwargs and kwargs["negative_slope"] is not None:
        negative_slope = float(kwargs["negative_slope"])
    if "positive_slope" in kwargs and kwargs["positive_slope"] is not None:
        positive_slope = float(kwargs["positive_slope"])
    if "sky" in kwargs and kwargs["sky"] is not None:
        sky = float(kwargs["sky"])

    arr = np.asarray(data, dtype=np.float32)
    is_2d = arr.ndim == 2

    if is_2d:
        rgb = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] == 3:
        rgb = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        rgb = arr
    else:
        raise ValueError(f"Expected 2D (H, W) or 3D RGB array, got shape {arr.shape}")

    r, g, b = leakySB_inverse(
        rgb,
        zero_points=zero_points,
        pixel_scale=pixel_scale,
        mmin=mmin,
        mmax=mmax,
        negative_slope=negative_slope,
        positive_slope=positive_slope,
        preserve_color_ratios=True,
    )

    if is_2d:
        flux = r + float(sky)
    else:
        flux = np.stack([r, g, b], axis=0 if arr.shape[0] == 3 else -1) + float(sky)

    return flux.astype(np.float32)


def inverse_stretch_image(
    data: np.ndarray,
    stretch_type: str = "leakysb",
    sky: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    Unified entry point for inverting astronomical surface brightness stretch.

    Parameters
    ----------
    data : np.ndarray
        Normalized image array in [0.0, 1.0].
    stretch_type : str, default 'leakysb'
        Algorithm name ('leakysb', 'leaky_sb').
    sky : float, default 0.0
        Background sky level that was subtracted prior to stretching.
    **kwargs
        Additional photometric parameters forwarded to inverse_leaky_surface_brightness.

    Returns
    -------
    np.ndarray
        Reconstructed image in physical flux units (float32).
    """
    kw_sky = kwargs.pop("sky", None)
    eff_sky = sky if sky != 0.0 else (kw_sky if kw_sky is not None else 0.0)

    mode = stretch_type.lower()
    if mode in ("leakysb", "leaky_sb"):
        return inverse_leaky_surface_brightness(data, sky=eff_sky, **kwargs)
    else:
        return inverse_leaky_surface_brightness(data, sky=eff_sky, **kwargs)
