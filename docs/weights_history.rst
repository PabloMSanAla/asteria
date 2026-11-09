.. _weights_history:

Weights History & Benchmark Results
====================================

This page tracks all official Asteria model weight releases, documenting training configurations, hyperparameter choices, and quantitative validation metrics.

.. contents:: Table of Contents
   :depth: 2
   :local:

Summary Table of Releases
-------------------------

.. list-table:: Official Model Weights Registry
   :widths: 12 18 12 15 15 15 13
   :header-rows: 1

   * - Version
     - Release Date
     - Backbone
     - Training Hardware
     - Interior IoU
     - Edge IoU (tol=2px)
     - Macro IoU
   * - **v0.1.0** (Latest)
     - 2026-10-31
     - UNet++ (32-512)
     - NASA GH200 (480GB)
     - **0.816**
     - **0.655**
     - **0.732**

---

Version 0.1.0 (Initial Public Release)
--------------------------------------

* **Release Date**: October 31, 2026
* **Weights Checkpoint**: `unetpp_v0.1.0.pt`
* **SHA-256 Checksum**: ``71feeee2bc372693c361c234d2c114a1cc2d0805e03bc9e2c1c95004b3d21df4``
* **Download URL**: `Google Drive Checkpoint <https://drive.google.com/file/d/1HfPMtcR0jjEfHIlAC-gEWAQeIFoljBwC/view?usp=sharing>`_

Model Specifications
^^^^^^^^^^^^^^^^^^^^

* **Architecture**: Nested U-Net (UNet++) with dense skip connections
* **Input Channels**: 3 (LeakySB photometric stretch representation)
* **Output Classes**: 3 (Sky Background: 0, Source Interior: 1, Deblending Edge: 2)
* **Feature Depths**: ``(32, 64, 128, 256, 512)``
* **Total Parameters**: ``9,159,747`` (36.6 MB in FP32)
* **Deep Supervision**: Disabled at inference for maximum inference throughput

Training Setup & Hardware
^^^^^^^^^^^^^^^^^^^^^^^^^

* **Supercomputer**: NASA High-End Computing Capability (HECC) / Pleiades Grace Hopper Partition
* **Compute Node**: Single NVIDIA Grace Hopper GH200 (72 ARM Neoverse V2 cores, 480 GB LPDDR5X CPU RAM, 96 GB HBM3 GPU VRAM)
* **Epochs**: 20
* **Batch Size**: 16 ($512 \times 512$ cutouts per forward batch)
* **Optimizer**: AdamW (initial learning rate $\eta = 2.0 \times 10^{-4}$, weight decay $1.0 \times 10^{-4}$)
* **Learning Rate Schedule**: Cosine Annealing with 2-epoch linear warmup
* **Precision**: Mixed Precision (PyTorch native ``bfloat16`` AMP) with TensorFloat-32 (TF32) tensor cores enabled
* **Gradient Clipping**: Maximum $L_2$ norm = $1.0$
* **Loss Function**: Weighted Cross-Entropy ($w=[0.5, 2.5, 8.0]$) + Multi-Class Dice Loss

Quantitative Validation Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluated across unseen test cutouts extracted from independent Roman WFI F106 synthetic fields:

.. list-table::
   :widths: 60 40
   :header-rows: 1

   * - Metric
     - Measured Value
   * - **Validation Loss**
     - ``0.2814``
   * - **Macro Average IoU**
     - **``0.7318``**
   * - **Source Interior (Core) IoU (Class 1)**
     - **``0.8164``**
   * - **Distance-Tolerant Edge IoU (tol=2px) (Class 2)**
     - **``0.6552``**
   * - **Soft Continuous Edge IoU**
     - **``0.7241``**
   * - **Strict Single-Pixel Edge IoU**
     - ``0.5120``

Key Findings & Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Clean Deblending**: The distance-tolerant edge IoU (0.655) and interior IoU (0.816) demonstrate that the network reliably isolates object cores and forms unbroken boundary barriers between touching galaxies.
2. **Inference Speed**:
   * Single $512 \times 512$ cutout: $\sim 14$ ms on NVIDIA GH200 / A100, $\sim 85$ ms on Apple Silicon M-series (MPS), $\sim 310$ ms on CPU.
   * Full $4088 \times 4088$ Roman WFI exposure (tiled sliding window, 81 overlapping tiles): $\sim 2.1$ s on GH200, $\sim 8.9$ s on Apple Silicon MPS.

---

Roadmap & Future Weights Tracking
---------------------------------

As new training runs and multi-filter weight models are developed, this document will be updated with:

* **v0.2.0 (Planned)**: Multi-band Roman imaging weights trained jointly across F106, F129, and F158 filters.
* **v0.3.0 (Planned)**: Deep transfer learning weights incorporating real Hubble Space Telescope (HST) and JWST NIRCam deep field observations.
