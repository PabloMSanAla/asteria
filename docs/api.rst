.. _api:

API Reference
=============

This section documents the public classes, functions, and modules in Asteria.

Top-Level Functions
-------------------

.. autofunction:: asteria.load_model

Inference & Deblending
----------------------

.. autoclass:: asteria.AsteriaSegmentor
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: asteria.deblend_sources

.. autoclass:: asteria.SegmentationResult
   :show-inheritance:

.. autofunction:: asteria.get_optimal_device

.. autofunction:: asteria.get_device_info

Neural Network Architecture
---------------------------

.. autoclass:: asteria.UNetPlusPlus
   :members:
   :undoc-members:
   :show-inheritance:

Photometric Stretches
---------------------

.. autofunction:: asteria.leaky_surface_brightness

.. autofunction:: asteria.asinh_stretch

.. autofunction:: asteria.linear_stretch

.. autofunction:: asteria.stretch_image

Astronomical Image I/O
----------------------

.. autofunction:: asteria.load_image

.. autoclass:: asteria.AstroImage
   :show-inheritance:

Visualization
-------------

.. autofunction:: asteria.plot_two_panel

.. autofunction:: asteria.plot_instance_overlay

Model Weights Registry
----------------------

.. autoclass:: asteria.weights.registry.WeightMetadata
   :show-inheritance:

.. autofunction:: asteria.weights.registry.get_weight_metadata
