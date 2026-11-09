"""I/O routines for astronomical FITS, ASDF, and array data."""

from asteria.io.reader import (
    AstroImage,
    load_image,
    is_s3_url,
    parse_s3_url,
    download_s3_file,
)

__all__ = [
    "AstroImage",
    "load_image",
    "is_s3_url",
    "parse_s3_url",
    "download_s3_file",
]
