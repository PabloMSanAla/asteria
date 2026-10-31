.. _dataset:

Training Dataset & Photometric Normalization
=============================================

This page outlines the astronomical dataset, synthetic simulation pipeline, and photometric transformations used to train official Asteria weights.

.. contents:: Table of Contents
   :depth: 2
   :local:

Survey Facility: Nancy Grace Roman Space Telescope
--------------------------------------------------

Asteria's pre-trained weights were developed for the **Nancy Grace Roman Space Telescope (Roman)**, NASA's next flagship astrophysics observatory. Roman's Wide Field Instrument (WFI) features a field of view 100 times larger than the Hubble Space Telescope, comprised of eighteen $4088 \times 4088$ pixel Hawaii-2RG (H2RG) HgCdTe sensor chip assemblies (SCAs).

Key Instrument Parameters:

* **Plate Scale**: $0.11$ arcseconds per pixel.
* **Filter Band**: **F106** (Wide near-infrared band, central wavelength $\lambda_c \approx 1.06 \, \mu\text{m}$).
* **Photometric Zero Point**: $31.4$ AB magnitudes.
* **Detector Array**: $4088 \times 4088$ pixels per SCA.

Synthetic Scene Simulation via `romanisim`
------------------------------------------

To generate high-fidelity training data with exact ground-truth instance segmentation maps, scenes were generated using `romanisim`, the official open-source image simulator for the Roman Space Telescope:

1. **Stellar Populations**: Realistic point spread functions (PSFs) computed using WebbPSF / Roman optical models, spanning dynamic ranges from faint Milky Way stars ($m \sim 26$) to saturating foreground giants ($m \sim 15$).
2. **Extragalactic Galaxy Distributions**: Deep cosmological fields populated with Sersic profile galaxies (de Vaucouleurs bulges with $n=4$ and exponential disks with $n=1$), with realistic morphological ellipticities, position angles, and half-light radii.
3. **Crowded Field Blends**: Controlled object clustering ensuring realistic densities of overlapping and touching galaxies.
4. **Physical Detector Noise**:
   * Poisson photon shot noise on celestial source fluxes and zodiacal light background.
   * Detector read noise ($15 \, e^-$ RMS).
   * Dark current and cosmic ray hits.
   * Inter-pixel capacitance (IPC) blurring.

WebDataset Pipeline & Cutout Generation
---------------------------------------

The raw $4088 \times 4088$ simulated exposures were converted into a high-throughput **WebDataset** streaming pipeline:

* **Cutout Resolution**: $512 \times 512$ pixels.
* **Dataset Volume**: Over 10,000 extracted astronomical cutouts packed into sequentially indexed POSIX tar shards.
* **Pre-computed 3-Class Targets**: Each cutout contains a paired NumPy float32 raw flux patch and uint8 integer segmentation mask with boundary channels generated via morphological binary dilation.

Photometric Preprocessing: LeakySB Stretch
------------------------------------------

Standard computer vision preprocessing (such as min-max or 0-255 linear scaling) fails on astronomical data due to the enormous dynamic range ($> 10^5$) between stellar cores and diffuse galaxy envelopes.

Asteria employs the **Leaky Surface Brightness (`leakySB`)** stretch:

1. **Surface Brightness Conversion**: Positive flux values $F$ are transformed into physical surface brightness magnitudes $\mu$ (mag / arcsec$^2$):

   .. math::

      \mu = \text{ZP} - 2.5 \log_{10}\left( \frac{F}{\Omega_{\text{pix}}} \right)

   where $\text{ZP} = 31.4$ and pixel solid angle $\Omega_{\text{pix}} = 0.11^2 = 0.0121 \, \text{arcsec}^2$.

2. **Inverted Min-Max Normalization**: Magnitudes are scaled between bright cutoff $m_{\min} = 13.0$ (mapping to 1.0) and faint limit $m_{\max} = 32.0$ (mapping to 0.0):

   .. math::

      I_{\text{norm}} = \frac{m_{\max} - \mu}{m_{\max} - m_{\min}}

3. **Linear Leak on Sky Noise**: For pixels with sub-background or negative flux (caused by background noise fluctuations), a small linear leak response is maintained rather than hard zero-clipping. This preserves the Gaussian shape of the sky background noise, preventing the neural network from overfitting to artificial step discontinuities.
