"""
Astronomical image reader supporting FITS (.fits, .fits.gz), Roman ASDF (.asdf), and NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any
import numpy as np
from astropy.io import fits

try:
    from astropy.wcs import WCS
    HAS_WCS = True
except ImportError:
    HAS_WCS = False
    WCS = Any


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
    wcs : Optional[WCS]
        Celestial World Coordinate System transformation if available.
    file_type : str
        Format identifier ('fits', 'asdf', 'array').
    path : Optional[Path]
        Source file path on disk.
    """
    data: np.ndarray
    header: Optional[dict] = None
    wcs: Optional[Any] = None
    file_type: str = "array"
    path: Optional[Path] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    def pixel_to_world(self, x: float, y: float):
        """Convert pixel (X, Y) to celestial coordinates (RA, Dec) if WCS is available."""
        if self.wcs is None:
            raise ValueError("No celestial WCS header is attached to this image.")
        return self.wcs.pixel_to_world(x, y)


def load_image(image: Union[str, Path, np.ndarray, AstroImage]) -> AstroImage:
    """
    Ingest an astronomical image from a file path or array.

    Parameters
    ----------
    image : str, Path, np.ndarray, or AstroImage
        Path to .fits, .fits.gz, or .asdf file, or a 2D NumPy array.

    Returns
    -------
    AstroImage
        Normalized astronomical image container.
    """
    if isinstance(image, AstroImage):
        return image

    if isinstance(image, np.ndarray):
        if image.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {image.shape}")
        return AstroImage(
            data=np.ascontiguousarray(image, dtype=np.float32),
            file_type="array",
        )

    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Astronomical image not found at: {path}")

    suffix = path.name.lower()

    # ASDF Format (Roman Space Telescope Level 2)
    if suffix.endswith(".asdf"):
        try:
            import asdf
        except ImportError:
            raise ImportError(
                "Reading .asdf files requires the optional 'asdf' package. "
                "Install it using: pip install asteria[asdf] (or pip install asdf)"
            )
        with asdf.open(str(path)) as af:
            if "roman" in af and "data" in af["roman"]:
                raw_data = np.array(af["roman"]["data"], dtype=np.float32)
                wcs_obj = af["roman"].get("meta", {}).get("wcs", None)
            elif "data" in af:
                raw_data = np.array(af["data"], dtype=np.float32)
                wcs_obj = af.get("meta", {}).get("wcs", None)
            else:
                raise KeyError(f"Could not locate image science data array in ASDF file: {path}")

            if raw_data.ndim != 2:
                raise ValueError(f"Expected 2D image data in ASDF, got shape {raw_data.shape}")

            return AstroImage(
                data=np.ascontiguousarray(raw_data, dtype=np.float32),
                header=None,
                wcs=wcs_obj,
                file_type="asdf",
                path=path,
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

        return AstroImage(
            data=np.ascontiguousarray(data_arr, dtype=np.float32),
            header=target_header,
            wcs=wcs_obj,
            file_type="fits",
            path=path,
        )
