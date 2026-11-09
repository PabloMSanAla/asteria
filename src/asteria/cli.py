"""
Command-line interface for Asteria astronomical inference.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from asteria import __version__
from asteria.inference.segmentor import AsteriaSegmentor
from asteria.inference.starnet3 import StarNet3
from asteria.weights.starnet3_manager import download_starnet3_model
from asteria.visualization.plot import plot_two_panel
from asteria.visualization.clarity import (
    create_mask_edge_overlay,
    open_pixel_clarity,
    open_starnet_clarity,
    _prepare_starnet_mef,
)
from asteria.inference.segmentor import _resolve_stretch_kwargs
from asteria.stretch.leakysb import inverse_leaky_surface_brightness
from asteria.io.reader import load_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asteria",
        description="Asteria: Fast astronomical UNet++ inference & segmentation engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        default=None,
        help="Path to astronomical image (.fits, .fits.gz, or .asdf).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Path to fine-tuned UNet++ model checkpoint (.pt) or release version tag (e.g. 'v0.1.0', 'latest').",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Pre-download official Asteria UNet++ weights to local cache without running inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Hardware accelerator ('cuda', 'mps', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Sliding tile size in pixels (set to 0 for unchunked direct inference).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap in pixels between adjacent tiles for seamless window blending.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of tiles processed concurrently in forward batches (dynamically calculated based on free memory if omitted).",
    )
    parser.add_argument(
        "--stretch",
        type=str,
        default="leakysb",
        choices=["leakysb", "asinh", "linear"],
        help="Astronomical surface brightness stretch algorithm.",
    )
    parser.add_argument(
        "--zp",
        "--zeropoint",
        "--zero_point",
        dest="zp",
        type=float,
        default=None,
        help="Photometric zero point for surface brightness stretch (default: 20.472 for MJy/sr Roman ASDF, 31.4 for FITS).",
    )
    parser.add_argument(
        "--sky",
        type=float,
        default=None,
        help="Sky background value to subtract before photometric LeakySB stretch (default: 0.0, automatically measured via sigma_clipped_stats median for Roman Level 2 images).",
    )
    parser.add_argument(
        "--pixel_scale",
        "--pixelscale",
        dest="pixel_scale",
        type=float,
        default=None,
        help="Detector plate scale in arcseconds per pixel (default: 0.11 for Roman WFI).",
    )
    parser.add_argument(
        "--mmin",
        type=float,
        default=None,
        help="Bright-end surface brightness magnitude cutoff in mag/arcsec^2 (default: 13.0).",
    )
    parser.add_argument(
        "--mmax",
        type=float,
        default=None,
        help="Faint-end surface brightness magnitude cutoff in mag/arcsec^2 (default: 32.0).",
    )
    parser.add_argument(
        "--leak_slope",
        "--leakslope",
        dest="leak_slope",
        type=float,
        default=None,
        help="Slope of linear response for negative/sub-background flux values (default: 0.05).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File path to save the 2-panel diagnostic figure (e.g. output.png).",
    )
    parser.add_argument(
        "--export_fits",
        type=str,
        default=None,
        help="Path to save multi-extension FITS containing the predicted segmentation mask.",
    )
    parser.add_argument(
        "--no_display",
        "--no_clarity",
        dest="no_display",
        action="store_true",
        help="Suppress interactive Pixel Clarity display window.",
    )
    parser.add_argument(
        "--clarity",
        "--pixel_clarity",
        dest="clarity",
        action="store_true",
        default=True,
        help="Launch interactive Pixel Clarity display window (enabled by default; use --no_display or --no_clarity to disable).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"asteria {__version__}",
    )
    return parser


def main(argv=None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in ("starnet", "starnet3"):
        return starnet3_main(raw_args[1:])

    parser = build_parser()
    args = parser.parse_args(raw_args)

    if getattr(args, "download", False):
        from asteria.weights.manager import resolve_checkpoint_path
        print(f"[asteria] Pre-downloading official UNet++ weights ({args.checkpoint})...")
        cached_p = resolve_checkpoint_path(args.checkpoint)
        print(f"[asteria] Successfully cached UNet++ weights at: {cached_p}")
        return 0

    if args.image is None:
        parser.print_help()
        return 0

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Astronomical image not found: {image_path}", file=sys.stderr)
        return 1

    print("=" * 65)
    print(f"  Asteria Astronomical Inference Engine (v{__version__})")
    print("=" * 65)
    print(f"Input Image:   {image_path}")
    print(f"Checkpoint:    {args.checkpoint}")

    # Load image
    astro = load_image(image_path, sky=args.sky)
    h, w = astro.data.shape
    print(f"Dimensions:    {w} × {h} pixels ({w * h / 1e6:.2f} Megapixels)")
    if astro.level is not None:
        ps_info = f", ps={astro.pixel_scale}" if astro.pixel_scale is not None else ""
        print(f"Product:       Roman Space Telescope Level {astro.level} ({astro.unit}, zp={astro.zero_point}{ps_info})")
    elif astro.unit != "DN/s":
        print(f"Unit:          {astro.unit} (zp={astro.zero_point})")
    if astro.sky != 0.0:
        print(f"Sky Subtracted:{astro.sky:12.6f} {astro.unit}")

    # Load segmentor
    segmentor = AsteriaSegmentor.from_pretrained(args.checkpoint, device=args.device)
    print(f"Accelerator:   {segmentor.device}")

    # Prepare stretch kwargs
    stretch_kwargs = {}
    if args.zp is not None:
        stretch_kwargs["zero_points"] = args.zp
    if args.sky is not None:
        stretch_kwargs["sky"] = args.sky
    elif astro.sky != 0.0:
        stretch_kwargs["sky"] = astro.sky
    if args.pixel_scale is not None:
        stretch_kwargs["pixel_scale"] = args.pixel_scale
    elif astro.pixel_scale is not None:
        stretch_kwargs["pixel_scale"] = astro.pixel_scale
    if args.mmin is not None:
        stretch_kwargs["mmin"] = args.mmin
    if args.mmax is not None:
        stretch_kwargs["mmax"] = args.mmax
    if args.leak_slope is not None:
        stretch_kwargs["leak_slope"] = args.leak_slope

    # Run inference
    t0 = time.time()
    pred_mask, probs, stretched = segmentor.predict_dense(
        image=astro,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        stretch_type=args.stretch,
        verbose=True,
        **stretch_kwargs,
    )
    elapsed = time.time() - t0

    # Class statistics
    total_px = pred_mask.size
    n_sky = int(np.sum(pred_mask == 0))
    n_core = int(np.sum(pred_mask == 1))
    n_edge = int(np.sum(pred_mask == 2))

    print(f"\n--- Segmentation Results ({elapsed:.2f}s) ---")
    print(f"Class 0 (Sky Background):  {n_sky:>10,} px ({100.0 * n_sky / total_px:6.2f}%)")
    print(f"Class 1 (Source Interior): {n_core:>10,} px ({100.0 * n_core / total_px:6.2f}%)")
    print(f"Class 2 (Deblending Edge): {n_edge:>10,} px ({100.0 * n_edge / total_px:6.2f}%)")

    # Construct high-contrast overlay of mask and edges on stretched image
    overlay = create_mask_edge_overlay(stretched=stretched, pred_mask=pred_mask)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(out_p), overlay)
        print(f"\nSaved overlay image to: {out_p.resolve()}")

    # Export FITS if requested
    if args.export_fits:
        export_p = Path(args.export_fits)
        export_p.parent.mkdir(parents=True, exist_ok=True)
        prim = fits.PrimaryHDU(data=astro.data)
        seg = fits.ImageHDU(data=pred_mask.astype(np.uint8), name="PRED_MASK")
        hdul = fits.HDUList([prim, seg])
        hdul.writeto(str(export_p), overwrite=True)
        print(f"Saved multi-extension FITS to: {export_p.resolve()}")

    if not args.no_display and getattr(args, "clarity", True):
        print(f"\n[asteria] Opening interactive view with Pixel Clarity (stretched exposure & mask/edge overlay)...")
        open_pixel_clarity(
            stretched=stretched,
            pred_mask=pred_mask,
            title=f"Asteria UNet++: {image_path.name}",
        )

    return 0


def build_starnet3_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="starnet3",
        description="StarNet3: Astronomical neural star removal with LeakySB stretch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        default=None,
        help="Path to astronomical image (.fits, .fits.gz, or .asdf).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Path to fine-tuned StarNet3 checkpoint (.pt) or release version tag (e.g. 'v0.1.0', 'latest').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Hardware accelerator ('cuda', 'mps', 'cpu'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Sliding tile size in pixels (set to 0 for unchunked direct inference).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap in pixels between adjacent tiles for seamless window blending.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of tiles processed concurrently in forward batches (dynamically calculated based on free memory if omitted).",
    )
    parser.add_argument(
        "--stretch",
        type=str,
        default="leakysb",
        choices=["leakysb", "asinh", "linear"],
        help="Astronomical surface brightness stretch algorithm.",
    )
    parser.add_argument(
        "--zp",
        "--zeropoint",
        "--zero_point",
        dest="zp",
        type=float,
        default=None,
        help="Photometric zero point for surface brightness stretch.",
    )
    parser.add_argument(
        "--sky",
        type=float,
        default=None,
        help="Sky background value to subtract before photometric LeakySB stretch (default: 0.0, automatically measured via sigma_clipped_stats median for Roman Level 2 images).",
    )
    parser.add_argument(
        "--pixel_scale",
        "--pixelscale",
        dest="pixel_scale",
        type=float,
        default=None,
        help="Detector plate scale in arcseconds per pixel.",
    )
    parser.add_argument(
        "--mmin",
        type=float,
        default=None,
        help="Bright-end surface brightness magnitude cutoff in mag/arcsec^2.",
    )
    parser.add_argument(
        "--mmax",
        type=float,
        default=None,
        help="Faint-end surface brightness magnitude cutoff in mag/arcsec^2.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save starless output image (e.g. starless.fits or starless.png).",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to save star intensity mask.",
    )
    parser.add_argument(
        "--export_fits",
        type=str,
        default=None,
        help="Path to save multi-extension FITS containing raw exposure (HDU 1) and inverse LeakySB starless prediction (HDU 2).",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Suppress interactive Pixel Clarity display window.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Trigger download of pre-trained StarNet3 weights to cache and exit.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"StarNet3 (Asteria {__version__})",
    )
    return parser


def starnet3_main(argv: list[str] | None = None) -> int:
    parser = build_starnet3_parser()
    args = parser.parse_args(argv)

    if args.download:
        print("[starnet3] Notice: StarNet3 remote weights are not yet published for automatic download.")
        print("Please provide a local checkpoint path with --checkpoint /path/to/weights.pth or export STARNET3_WEIGHTS_PATH=/path/to/weights.pth")
        return 1

    if not args.image:
        parser.print_help()
        print("\n[starnet3] Error: Must supply an image path.")
        return 1

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[starnet3] Error: Input image '{image_path}' not found.")
        return 1

    if args.checkpoint == "latest" and not (os.environ.get("STARNET3_WEIGHTS_PATH") and Path(os.environ["STARNET3_WEIGHTS_PATH"]).exists()):
        print("\n[starnet3] Notice: StarNet3 remote weights are not yet published for automatic download.")
        print("Only UNet++ is available for automatic download.")
        print("Please specify a local checkpoint path via --checkpoint /path/to/weights.pth or export STARNET3_WEIGHTS_PATH=/path/to/weights.pth")
        return 1

    print(f"[starnet3] Initializing StarNet3 engine ({args.checkpoint})...")
    starnet = StarNet3.from_pretrained(args.checkpoint, device=args.device)

    astro = load_image(image_path, sky=args.sky)
    print(f"[starnet3] Loaded {astro.file_type.upper()} image: {astro.data.shape} px on {starnet.device}")
    if astro.level is not None:
        ps_str = f", ps={astro.pixel_scale}" if astro.pixel_scale is not None else ""
        print(f"[starnet3] Product:       Roman Space Telescope Level {astro.level} ({astro.unit}, zp={astro.zero_point}{ps_str})")
    if astro.sky != 0.0:
        print(f"[starnet3] Sky Subtracted: {astro.sky:.6f} {astro.unit}")

    stretch_kwargs = {}
    if args.zp is not None:
        stretch_kwargs["zero_points"] = args.zp
    if args.sky is not None:
        stretch_kwargs["sky"] = args.sky
    elif astro.sky != 0.0:
        stretch_kwargs["sky"] = astro.sky
    if args.pixel_scale is not None:
        stretch_kwargs["pixel_scale"] = args.pixel_scale
    elif astro.pixel_scale is not None:
        stretch_kwargs["pixel_scale"] = astro.pixel_scale
    if args.mmin is not None:
        stretch_kwargs["mmin"] = args.mmin
    if args.mmax is not None:
        stretch_kwargs["mmax"] = args.mmax

    t0 = time.time()
    if args.tile_size > 0:
        starless, mask, stretched = starnet.predict_dense(
            image=astro,
            tile_size=args.tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            stretch_type=args.stretch,
            verbose=True,
            **stretch_kwargs,
        )
    else:
        starless, mask, stretched = starnet.predict(
            image=astro,
            stretch_type=args.stretch,
            **stretch_kwargs,
        )
    elapsed = time.time() - t0
    print(f"[starnet3] Inference completed in {elapsed:.2f}s!")

    # Reconstruct physical photon fluxes from predicted starless image using inverse LeakySB
    resolved_stretch = _resolve_stretch_kwargs(
        astro=astro,
        stretch_type=args.stretch,
        **stretch_kwargs,
    )
    starless_flux = inverse_leaky_surface_brightness(starless, **resolved_stretch)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() in (".fits", ".fit", ".fts"):
            prim = fits.PrimaryHDU(data=starless.astype(np.float32))
            prim.writeto(str(out_p), overwrite=True)
        else:
            plt.imsave(str(out_p), starless, cmap="gray" if starless.ndim == 2 else None)
        print(f"[starnet3] Saved starless image to: {out_p.resolve()}")

    if args.mask:
        mask_p = Path(args.mask)
        mask_p.parent.mkdir(parents=True, exist_ok=True)
        if mask_p.suffix.lower() in (".fits", ".fit", ".fts"):
            prim = fits.PrimaryHDU(data=mask.astype(np.float32))
            prim.writeto(str(mask_p), overwrite=True)
        else:
            plt.imsave(str(mask_p), mask, cmap="inferno")
        print(f"[starnet3] Saved star mask to: {mask_p.resolve()}")

    # Export multi-extension FITS with raw un-stretched data (HDU 1) and inverse LeakySB prediction (HDU 2)
    if args.export_fits:
        export_p = Path(args.export_fits)
        _prepare_starnet_mef(
            raw_image=astro.data,
            starless_flux=starless_flux,
            header=astro.header,
            save_path=export_p,
        )
        print(f"[starnet3] Saved multi-extension FITS to: {export_p.resolve()}")

    # Open interactive Pixel Clarity display showing raw image and inverted starless prediction in 2 HDUs
    if not args.no_display:
        print(f"\n[starnet3] Opening interactive view with Pixel Clarity (multi-extension FITS: raw & starless inverse LeakySB)...")
        open_starnet_clarity(
            raw_image=astro.data,
            starless_flux=starless_flux,
            header=astro.header,
            title=f"StarNet3: {image_path.name}",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
