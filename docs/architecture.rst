.. _architecture:

Model Architecture
==================

Asteria is powered by a **Nested U-Net (UNet++)** deep convolutional neural network, engineered specifically for high-contrast astronomical imaging and crowded-field source extraction.

.. contents:: Table of Contents
   :depth: 2
   :local:

Why UNet++ for Astronomical Source Detection?
---------------------------------------------

Traditional astronomical source extraction pipelines (such as SExtractor or standard thresholding) struggle in crowded stellar regions and galaxy clusters where object halos overlap. While standard U-Nets use direct skip connections between encoder and decoder stages, they suffer from a **semantic gap**: low-level encoder features contain rich spatial coordinates but high noise, whereas high-level decoder features contain abstract semantic context at degraded spatial resolution.

UNet++ (Zhou et al., 2018; 2019) overcomes this limitation by bridging the semantic gap through **nested, dense skip pathways**.

Mathematical Formulation
-------------------------

Let $x^{i,j}$ denote the output of node $X^{i,j}$, where:

* $i \in \{0, 1, 2, 3, 4\}$ indexes the downsampling depth along the encoder backbone.
* $j \in \{0, 1, 2, 3, 4\}$ indexes the dense convolution layer along the horizontal skip pathway.

The forward accumulation at each intermediate node is computed as:

.. math::

   x^{i,j} = \begin{cases}
   \mathcal{H}_{\text{enc}}\left( \mathcal{D}(x^{i-1, 0}) \right), & j = 0 \\
   \mathcal{H}\left( \left[ [x^{i,k}]_{k=0}^{j-1}, \mathcal{U}(x^{i+1, j-1}) \right] \right), & j > 0
   \end{cases}

where:

* $\mathcal{H}(\cdot)$ denotes a double convolution block (Conv2D $\to$ BatchNorm $\to$ ReLU $\to$ Conv2D $\to$ BatchNorm $\to$ ReLU).
* $\mathcal{D}(\cdot)$ denotes $2 \times 2$ max pooling with stride 2.
* $\mathcal{U}(\cdot)$ denotes bilinear upsampling with scale factor 2.
* $[\cdot]$ denotes channel-wise feature concatenation.

Through these nested skip pathways, feature maps from earlier stages are progressively merged and refined before reaching the decoder nodes, ensuring that fine astronomical boundary structures are preserved with sub-pixel fidelity.

Three-Class Boundary-Aware Target Formulation
---------------------------------------------

Standard binary segmentation (source vs. background) inevitably merges touching or overlapping astronomical sources into contiguous blobs, requiring heuristic post-processing deblenders.

Asteria formulates source detection as a **3-class boundary-aware semantic segmentation** task:

* **Class 0 (Sky Background)**: Deep space sky noise and intergalactic regions.
* **Class 1 (Source Interior / Core)**: High surface brightness galactic cores and stellar point spread functions (PSFs).
* **Class 2 (Deblending Boundary / Edge)**: The outer perimeter of celestial sources and, critically, the contact interface between overlapping or touching objects.

Deblending via Marker-Controlled Watershed
------------------------------------------

During inference, Asteria leverages the explicit boundary prediction to split touching galaxies:

1. **Topological Seed Markers**: Identified by thresholding the interior probability map ($P_{\text{core}} \ge 0.45$). Connected components generate distinct integer markers for each galaxy core.
2. **Total Foreground Mask**: Computed from $P_{\text{core}} + P_{\text{edge}} \ge 0.50$ (or $P_{\text{bg}} < 0.50$).
3. **Watershed Expansion**: Inverted Euclidean distance transforms expand the core markers outward until they collide along the predicted Class 2 boundary line, cleanly separating touching objects into discrete instances without morphological distortion.

Loss Function & Class Weighting
-------------------------------

Due to the extreme class imbalance in deep astronomical exposures (where the sky background occupies $\sim 85-90\%$ of pixels, source cores $\sim 2-5\%$, and 1-pixel boundaries $\sim 5-10\%$), models trained with standard unweighted Cross-Entropy collapse toward predicting the background.

Asteria was trained using a **Combined Weighted Cross-Entropy and Multi-Class Dice Loss**:

.. math::

   \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{WCE}} + \lambda \mathcal{L}_{\text{Dice}}

with empirically optimized class weights:

.. math::

   \mathbf{w}_{\text{CE}} = [w_{\text{bg}} = 0.5, \; w_{\text{core}} = 2.5, \; w_{\text{edge}} = 8.0]

The high penalty ($8.0$) on boundary errors forces the network to prioritize thin edge continuity, which is the decisive factor for successful instance deblending.

Tiled Dense Sliding-Window Inference (`predict_dense`)
------------------------------------------------------

To process large scientific exposures (such as the $4088 \times 4088$ detector arrays of the Nancy Grace Roman Space Telescope) without GPU out-of-memory (OOM) failures or artificial boundary seams:

1. The exposure is decomposed into overlapping $512 \times 512$ tiles with a default 64-pixel margin.
2. Each tile is evaluated in mini-batches through the neural network.
3. Softmax probability maps are accumulated using a **2D separable Hann / cosine window**:

.. math::

   W(x, y) = \sin\left(\frac{\pi x}{T}\right) \cdot \sin\left(\frac{\pi y}{T}\right)

4. The accumulated probability tensor is normalized by the total weight matrix, completely eliminating edge discontinuity artifacts and bounding memory consumption to $< 350$ MB VRAM.
