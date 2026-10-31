# Asteria 🌌

[![PyPI version](https://img.shields.io/pypi/v/asteria.svg?color=blue)](https://pypi.org/project/asteria/)
[![Python Versions](https://img.shields.io/pypi/pyversions/asteria.svg)](https://pypi.org/project/asteria/)
[![CI Status](https://github.com/PabloMSanAla/asteria/actions/workflows/ci.yml/badge.svg)](https://github.com/PabloMSanAla/asteria/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/asteria/badge/?version=latest)](https://asteria.readthedocs.io/en/latest/?badge=latest)

**Asteria** is a lightweight, high-performance deep learning inference engine and instance deblender for astronomical imaging, powered by boundary-aware **UNet++ (Nested U-Net)**.

Designed for astronomical surveys like the **Nancy Grace Roman Space Telescope**, Asteria provides promptless, end-to-end source extraction that separates crowded, touching stars and galaxies in a single neural forward pass.

---

## Key Features

- **Lightweight & Self-Contained**: Pure PyTorch UNet++ architecture without bulky dependencies or heavyweight vision foundation models.
- **Hardware Acceleration with Automatic Fallback**: Auto-detects and optimizes execution across **NVIDIA CUDA** (with TF32 and AMP), **Apple Silicon MPS** (Metal Performance Shaders on macOS), and universal **CPU**.
- **Physical Boundary-Aware Deblending**: Trained with 3-class supervision:
  - `0`: Sky Background
  - `1`: Source Interior / Core
  - `2`: Deblending Boundary / Contact Interface
- **Seamless Tiled Dense Inference (`predict_dense`)**: Sliding-window 2D Hann/cosine window blending to process arbitrary image sizes (from local cutouts up to full $4088 \times 4088$ Roman WFI exposures) with bounded memory ($< 350$ MB VRAM).
- **Photometric LeakySB Stretch**: Non-linear surface brightness transformation that preserves stellar point-spread functions (PSFs) while lifting faint, diffuse galactic wings.
- **Unified Astronomical I/O**: Direct support for 2D astronomical **FITS** (`.fits`, `.fits.gz`), Roman **ASDF** (`.asdf`), and raw **NumPy** arrays.
- **Instant Deblending**: Marker-controlled watershed expansion turning dense neural predictions into discrete celestial catalogs with centroids, bounding boxes, and photometry.

---

## Installation

### From PyPI
```bash
pip install asteria
```

### With Roman ASDF Support
```bash
pip install "asteria[asdf]"
```

### From Source
```bash
git clone https://github.com/PabloMSanAla/asteria.git
cd asteria
pip install -e .
```

---

## Quickstart

### 1. Python API

```python
import asteria

# 1. Load pre-trained model (auto-downloads weights on first run, auto-selects device)
segmentor = asteria.load_model("latest")

# 2. Run dense 3-class segmentation inference on a FITS or ASDF exposure
pred_mask, probs, stretched = segmentor.predict_dense(
    "scene_0001_F106.fits",
    tile_size=512,
    overlap=64,
)

# 3. Render publication-ready side-by-side 2-panel figure
fig = asteria.plot_two_panel(
    stretched=stretched,
    pred_mask=pred_mask,
    save_path="prediction.png",
)

# 4. Extract discrete, deblended astronomical sources (marker-controlled watershed)
catalog = segmentor.segment("scene_0001_F106.fits")
print(f"Extracted {len(catalog.masks)} distinct celestial objects!")
for idx, (cx, cy) in enumerate(catalog.centroids[:5]):
    print(f"Source #{idx + 1}: centroid=({cx:.1f}, {cy:.1f}), area={catalog.areas[idx]} px")
```

### 2. Command-Line Interface (CLI)

Asteria includes the `asteria-predict` CLI command:

```bash
# Basic prediction and display
asteria-predict scene.fits

# Full-field tiled inference with saving and FITS export
asteria-predict scene.fits \
    --tile_size 512 \
    --overlap 64 \
    --output results.png \
    --export_fits segmap.fits \
    --deblend \
    --no_display
```

---

## Model Architecture & Weights History

| Weight Version | Backbone | Resolution | Training Set | Epochs | Interior IoU | Edge IoU (tol=2px) | Macro IoU | Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`v0.1.0`** (latest) | UNet++ (32-512) | $512 \times 512$ | Roman WFI F106 Synthetic | 20 | **0.816** | **0.655** | **0.732** | NASA GH200 (480GB) |

For comprehensive architectural details and benchmark comparisons, see the [Asteria Documentation](https://asteria.readthedocs.io).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
