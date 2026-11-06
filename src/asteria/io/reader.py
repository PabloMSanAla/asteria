"""
Astronomical image reader supporting FITS (.fits, .fits.gz), Roman ASDF (.asdf), and NumPy arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any, Dict
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

logger = logging.getLogger(__name__)

try:
    from astropy.wcs import WCS
    HAS_WCS = True
except ImportError:
    HAS_WCS = False
    WCS = Any

try:
    import roman_datamodels as rdm
    from roman_datamodels import datamodels
    HAS_RDM = True
except ImportError:
    HAS_RDM = False

try:
    import asdf
    HAS_ASDF = True
except ImportError:
    HAS_ASDF = False


@dataclass
class AstroImage:
    """
    Standard container for an ingested astronomical image.

    Attributes
    ----------
    data : np.ndarray
        2D float32 pixel array (H, W).
    header : Optional[dict]
        FITS or ASDF metadata dictionary.
    wcs : Optional[Any]
        Celestial World Coordinate System transformation if available (astropy WCS or gwcs).
    file_type : str
        Format identifier ('fits', 'asdf', 'array').
    path : Optional[Path]
        Source file path on disk.
    zero_point : float
        Photometric zero point for surface brightness transformation (31.4 for counts, 20.472 for MJy/sr).
    unit : str
        Physical unit of data array ('DN/s', 'MJy/sr', etc.).
    level : Optional[int]
        Roman data product level (2 for Level 2 calibrated exposure, 3 for Level 3 mosaic, None otherwise).
    """
    data: np.ndarray
    header: Optional[dict] = None
    wcs: Optional[Any] = None
    file_type: str = "array"
    path: Optional[Path] = None
    zero_point: float = 31.4
    unit: str = "DN/s"
    level: Optional[int] = None
    sky: float = 0.0
    pixel_scale: Optional[float] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def pixel_to_world(self, x: float, y: float):
        """Convert pixel (X, Y) to celestial coordinates (RA, Dec) if WCS is available."""
        if self.wcs is None:
            raise ValueError("No celestial WCS header is attached to this image.")
        if hasattr(self.wcs, "pixel_to_world"):
            return self.wcs.pixel_to_world(x, y)
        elif callable(self.wcs):
            return self.wcs(x, y)
        raise ValueError(f"WCS object of type {type(self.wcs)} does not support coordinate conversion.")


def _measure_sky_median(data: np.ndarray) -> float:
    """Measure background sky value using astropy sigma_clipped_stats median."""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return 0.0
    _, med, _ = sigma_clipped_stats(finite)
    return float(med)


def load_image(
    image: Union[str, Path, np.ndarray, AstroImage],
    zero_point: Optional[float] = None,
    sky: Optional[float] = None,
    level: Optional[int] = None,
    pixel_scale: Optional[float] = None,
) -> AstroImage:
    """
    Ingest an astronomical image from a file path or array.

    Parameters
    ----------
    image : str, Path, np.ndarray, or AstroImage
        Path to .fits, .fits.gz, or .asdf file, or a 2D NumPy array.
    zero_point : float, optional
        Photometric zero point override.
    sky : float, optional
        Sky background value to subtract before stretching. If None, defaults to 0.0,
        except for Roman Level 2 exposures where it is automatically estimated via
        astropy.stats.sigma_clipped_stats median.
    level : int, optional
        Product level override (e.g. 2 for Level 2 exposure).
    pixel_scale : float, optional
        Detector pixel plate scale in arcsec/pixel. Automatically defaults to 1.0
        for Roman Level 2 and Level 3 ASDF surface brightness products (MJy/sr).

    Returns
    -------
    AstroImage
        Normalized astronomical image container.
    """
    if isinstance(image, AstroImage):
        if sky is not None:
            image.sky = float(sky)
        if pixel_scale is not None:
            image.pixel_scale = float(pixel_scale)
        return image

    if isinstance(image, np.ndarray):
        if image.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {image.shape}")
        arr_data = np.ascontiguousarray(image, dtype=np.float32)
        if level == 2 and sky is None:
            resolved_sky = _measure_sky_median(arr_data)
        elif sky is not None:
            resolved_sky = float(sky)
        else:
            resolved_sky = 0.0

        resolved_ps = float(pixel_scale) if pixel_scale is not None else (1.0 if level in (2, 3) else None)

        return AstroImage(
            data=arr_data,
            file_type="array",
            zero_point=zero_point if zero_point is not None else 31.4,
            unit="DN/s",
            level=level,
            sky=resolved_sky,
            pixel_scale=resolved_ps,
        )

    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Astronomical image not found at: {path}")

    suffix = path.name.lower()

    # ASDF Format (Roman Space Telescope Level 2 and Level 3 images)
    if suffix.endswith(".asdf"):
        if not HAS_RDM and not HAS_ASDF:
            raise ImportError(
                "Reading Roman Space Telescope .asdf files requires 'roman_datamodels'. "
                "Install it using: pip install roman_datamodels"
            )

        data = None
        header_meta = None
        wcs_obj = None
        level: Optional[int] = None
        unit: str = "DN/s"
        zero_point: float = 31.4

        # 1. Attempt reading via roman_datamodels API
        if HAS_RDM:
            try:
                model = rdm.open(str(path))
                model_meta = getattr(model, "meta", None)
                model_type = getattr(model_meta, "model_type", "") if model_meta is not None else ""

                # Level 2 Roman image (WfiImage / ImageModel)
                if isinstance(model, datamodels.ImageModel) or model_type in ("ImageModel", "WfiImage"):
                    level = 2
                    raw_data = np.asarray(model.data, dtype=np.float32)
                    wcs_obj = getattr(model_meta, "wcs", None)
                    header_meta = model_meta.to_flat_dict() if hasattr(model_meta, "to_flat_dict") else {}

                    # Photometry conversion from DN/s to MJy/sr
                    phot = getattr(model_meta, "photometry", None)
                    conv_factor = getattr(phot, "conversion_megajanskys", None) if phot is not None else None
                    if conv_factor is None and isinstance(phot, dict):
                        conv_factor = phot.get("conversion_megajanskys")

                    if conv_factor is not None and conv_factor > 0 and conv_factor != -999999.0:
                        data = raw_data * float(conv_factor)
                        logger.info(
                            f"[asteria] Roman Level 2 image: applied photometry conversion_megajanskys={conv_factor:.6e} to MJy/sr."
                        )
                    else:
                        data = raw_data
                        logger.warning(
                            f"[asteria] Roman Level 2 image: conversion_megajanskys not found or invalid; leaving data unscaled."
                        )

                    unit = "MJy/sr"
                    zero_point = 20.472

                # Level 3 Roman mosaic (WfiMosaic / MosaicModel)
                elif isinstance(model, datamodels.MosaicModel) or model_type in ("MosaicModel", "WfiMosaic"):
                    level = 3
                    raw_data = np.asarray(model.data, dtype=np.float32)
                    wcs_obj = getattr(model_meta, "wcs", None)
                    header_meta = model_meta.to_flat_dict() if hasattr(model_meta, "to_flat_dict") else {}

                    # Level 3 mosaics are already in surface brightness units (MJy/sr)
                    data = raw_data
                    unit = "MJy/sr"
                    zero_point = 20.472
                    logger.info("[asteria] Roman Level 3 mosaic: image is already in MJy/sr (zp=20.472).")

                elif hasattr(model, "data") and model.data is not None:
                    raw_data = np.asarray(model.data, dtype=np.float32)
                    wcs_obj = getattr(model_meta, "wcs", None)
                    header_meta = model_meta.to_flat_dict() if hasattr(model_meta, "to_flat_dict") else {}
                    data = raw_data
                    level = None
                    unit = "DN/s"
                    zero_point = 31.4

            except Exception as e:
                logger.debug(f"[asteria] roman_datamodels.open failed on {path}: {e}; falling back to asdf tree parser.")

        # 2. Fallback: Parse ASDF tree directly if roman_datamodels didn't parse it
        if data is None and HAS_ASDF:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with asdf.open(str(path), ignore_unrecognized_tag=True, ignore_missing_extensions=True) as af:
                    tree = af.tree
                    roman_tree = tree.get("roman", tree)
                    meta = roman_tree.get("meta", {}) if isinstance(roman_tree, dict) else {}
                    model_type = meta.get("model_type", "") if isinstance(meta, dict) else getattr(meta, "model_type", "")

                    raw_data = None
                    if isinstance(roman_tree, dict) and "data" in roman_tree:
                        raw_data = np.asarray(roman_tree["data"], dtype=np.float32)
                    elif "data" in tree:
                        raw_data = np.asarray(tree["data"], dtype=np.float32)
                    elif "science" in tree:
                        raw_data = np.asarray(tree["science"], dtype=np.float32)
                    else:
                        for k, v in roman_tree.items():
                            if hasattr(v, "shape") and len(v.shape) == 2:
                                raw_data = np.asarray(v, dtype=np.float32)
                                break

                    if raw_data is None:
                        raise KeyError(f"Could not locate image science data array in ASDF file: {path}")

                    wcs_obj = meta.get("wcs", None) if isinstance(meta, dict) else getattr(meta, "wcs", None)
                    header_meta = dict(meta) if isinstance(meta, dict) else {}

                    # Identify Level 2 vs Level 3
                    if model_type in ("ImageModel", "WfiImage") or (isinstance(meta, dict) and "photometry" in meta):
                        level = 2
                        phot = meta.get("photometry", {}) if isinstance(meta, dict) else getattr(meta, "photometry", {})
                        conv_factor = phot.get("conversion_megajanskys") if isinstance(phot, dict) else getattr(phot, "conversion_megajanskys", None)
                        if conv_factor is not None and conv_factor > 0 and conv_factor != -999999.0:
                            data = raw_data * float(conv_factor)
                        else:
                            data = raw_data
                        unit = "MJy/sr"
                        zero_point = 20.472
                    elif model_type in ("MosaicModel", "WfiMosaic") or (isinstance(meta, dict) and "resample" in meta):
                        level = 3
                        data = raw_data
                        unit = "MJy/sr"
                        zero_point = 20.472
                    else:
                        data = raw_data
                        unit = "DN/s"
                        zero_point = 31.4

        if data is None:
            raise ValueError(f"Could not locate image science data array in ASDF file: {path}")

        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D image data in ASDF, got shape {data.shape}")

        if level == 2 and sky is None:
            resolved_sky = _measure_sky_median(data)
            logger.info(
                f"[asteria] Roman Level 2 image: measured background sky={resolved_sky:.6e} using sigma_clipped_stats median."
            )
        elif sky is not None:
            resolved_sky = float(sky)
        else:
            resolved_sky = 0.0

        resolved_ps = float(pixel_scale) if pixel_scale is not None else (1.0 if level in (2, 3) else None)

        return AstroImage(
            data=np.ascontiguousarray(data, dtype=np.float32),
            header=header_meta,
            wcs=wcs_obj,
            file_type="asdf",
            path=path,
            zero_point=zero_point,
            unit=unit,
            level=level,
            sky=resolved_sky,
            pixel_scale=resolved_ps,
        )

    # Standard FITS Format (.fits, .fits.gz)
    with fits.open(str(path), memmap=True) as hdul:
        data_arr = None
        target_header = None
        wcs_obj = None

        # Look for first valid 2D image HDU
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                data_arr = np.array(hdu.data, dtype=np.float32)
                target_header = dict(hdu.header)
                if HAS_WCS:
                    try:
                        wcs_cand = WCS(hdu.header)
                        if wcs_cand.has_celestial:
                            wcs_obj = wcs_cand
                    except Exception:
                        wcs_obj = None
                break

        if data_arr is None:
            raise ValueError(f"No valid 2D image HDU found in FITS file: {path}")

        fits_level = level
        if fits_level is None and target_header is not None:
            prod_lvl = target_header.get("PROD_LVL", target_header.get("LEVEL", None))
            if prod_lvl in (2, "2", "L2", "LEVEL2", "level2"):
                fits_level = 2
            elif prod_lvl in (3, "3", "L3", "LEVEL3", "level3"):
                fits_level = 3

        if fits_level == 2 and sky is None:
            resolved_sky = _measure_sky_median(data_arr)
            logger.info(
                f"[asteria] Level 2 FITS image: measured background sky={resolved_sky:.6e} using sigma_clipped_stats median."
            )
        elif sky is not None:
            resolved_sky = float(sky)
        else:
            resolved_sky = 0.0

        resolved_fits_ps = float(pixel_scale) if pixel_scale is not None else (1.0 if fits_level in (2, 3) else None)

        return AstroImage(
            data=np.ascontiguousarray(data_arr, dtype=np.float32),
            header=target_header,
            wcs=wcs_obj,
            file_type="fits",
            path=path,
            zero_point=zero_point if zero_point is not None else 31.4,
            unit="DN/s",
            level=fits_level,
            sky=resolved_sky,
            pixel_scale=resolved_fits_ps,
        )

