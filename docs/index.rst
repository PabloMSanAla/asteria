.. Asteria documentation master file

Asteria: Astronomical UNet++ Inference & Deblending
===================================================

.. image:: https://img.shields.io/pypi/v/asteria.svg?color=blue
   :target: https://pypi.org/project/asteria/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/asteria.svg
   :target: https://pypi.org/project/asteria/
   :alt: Python Versions

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

**Asteria** is a lightweight, high-performance deep learning inference engine and instance deblender for astronomical imaging, powered by boundary-aware **UNet++ (Nested U-Net)** convolutional networks.

Developed for deep survey facilities like the **Nancy Grace Roman Space Telescope (WFI)**, Asteria replaces multi-stage astronomical source detection pipelines with an ultra-fast, promptless neural model capable of segmenting and deblending overlapping stars and galaxies in a single pass.

Key Capabilities
----------------

* **Promptless Single-Pass Segmentation**: No point clicks, bounding box prompts, or candidate lattices required.
* **Physical Boundary-Aware Deblending**: Separates touching galaxies and crowded stellar cores via an explicit 3-class target space (Sky, Interior Core, Deblending Edge).
* **Hardware Accelerated with Automatic Fallbacks**: Native auto-detection and execution on **NVIDIA CUDA** (with TF32 & AMP bfloat16), **Apple Silicon MPS** (Metal Performance Shaders on macOS), and universal **CPU**.
* **Seamless Full-Field Dense Tiling (`predict_dense`)**: Sliding-window 2D Hann/cosine window blending enabling inference across full $4088 \times 4088$ detector mosaics with bounded memory ($< 350$ MB VRAM).
* **Calibrated LeakySB Photometric Stretch**: Non-linear surface brightness stretch preserving dynamic range across $>10^5$ flux ratios without saturating stellar point spread functions (PSFs).
* **Native Astronomical I/O**: Direct support for 2D astronomical **FITS** (`.fits`, `.fits.gz`), Roman **ASDF** (`.asdf`), and raw **NumPy** arrays.

Installation
------------

Standard installation via ``pip``:

.. code-block:: bash

   pip install asteria

To enable Roman Space Telescope ASDF Level 2 support:

.. code-block:: bash

   pip install "asteria[asdf]"

Quickstart
----------

.. code-block:: python

   import asteria

   # 1. Initialize segmentor (auto-downloads latest weights and selects device)
   segmentor = asteria.load_model("latest")

   # 2. Run dense 3-class segmentation on an astronomical exposure
   pred_mask, probs, stretched = segmentor.predict_dense(
       "scene_0001_F106.fits",
       tile_size=512,
       overlap=64,
   )

   # 3. Plot high-contrast 2-panel diagnostic figure
   fig = asteria.plot_two_panel(
       stretched=stretched,
       pred_mask=pred_mask,
       save_path="prediction_summary.png",
   )

   # 4. Deblend touching instances with marker-controlled watershed
   catalog = segmentor.segment("scene_0001_F106.fits")
   print(f"Detected {len(catalog.masks)} distinct celestial sources!")

.. toctree::
   :maxdepth: 2
   :caption: Documentation & Guides

   architecture
   dataset
   weights_history
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
