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
from asteria.visualization.plot import plot_two_panel
from asteria.io.reader import load_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asteria",
        description="Asteria: Fast astronomical UNet++ inference & deblender.",
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
        default=2,
        help="Number of tiles processed concurrently in forward batches.",
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
        "--deblend",
        action="store_true",
        help="Perform marker-controlled watershed deblending and print source catalog.",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Suppress interactive matplotlib display window.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"asteria {__version__}",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
    astro = load_image(image_path)
    h, w = astro.data.shape
    print(f"Dimensions:    {w} × {h} pixels ({w * h / 1e6:.2f} Megapixels)")
    if astro.level is not None:
        print(f"Product:       Roman Space Telescope Level {astro.level} ({astro.unit}, zp={astro.zero_point})")
    elif astro.unit != "DN/s":
        print(f"Unit:          {astro.unit} (zp={astro.zero_point})")

    # Load segmentor
    segmentor = AsteriaSegmentor.from_pretrained(args.checkpoint, device=args.device)
    print(f"Accelerator:   {segmentor.device}")

    # Prepare stretch kwargs
    stretch_kwargs = {}
    if args.zp is not None:
        stretch_kwargs["zero_points"] = args.zp
    if args.pixel_scale is not None:
        stretch_kwargs["pixel_scale"] = args.pixel_scale
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

    # Watershed Deblending if requested
    if args.deblend:
        print("\nRunning marker-controlled watershed deblending...")
        res = segmentor.segment(
            image=astro,
            tile_size=args.tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            stretch_type=args.stretch,
            **stretch_kwargs,
        )
        print(f"Extracted {len(res.masks)} distinct celestial source instances!")
        if len(res.masks) > 0:
            print(f"{'ID':<4} {'Centroid (X, Y)':<22} {'BBox [x0, y0, x1, y1]':<26} {'Area (px)':<10}")
            print("-" * 65)
            for i in range(min(10, len(res.masks))):
                cx, cy = res.centroids[i]
                b = [round(float(v), 1) for v in res.boxes[i]]
                print(f"{i+1:<4} ({cx:6.1f}, {cy:6.1f})      {str(b):<26} {res.areas[i]:<10}")

    # Generate Figure
    fig = plot_two_panel(
        stretched=stretched,
        pred_mask=pred_mask,
        title=f"Asteria UNet++: {image_path.name}",
        save_path=args.output,
    )

    if args.output:
        print(f"\nSaved 2-panel figure to: {Path(args.output).resolve()}")

    # Export FITS if requested
    if args.export_fits:
        export_p = Path(args.export_fits)
        export_p.parent.mkdir(parents=True, exist_ok=True)
        prim = fits.PrimaryHDU(data=astro.data)
        seg = fits.ImageHDU(data=pred_mask.astype(np.uint8), name="PRED_MASK")
        hdul = fits.HDUList([prim, seg])
        hdul.writeto(str(export_p), overwrite=True)
        print(f"Saved multi-extension FITS to: {export_p.resolve()}")

    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
