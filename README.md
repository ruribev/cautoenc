# Conditional Variational Autoencoder for Reconstructing Topographic Profiles

This repository contains a minimal implementation of the Conditional Variational
Autoencoder (CVAE) used to reconstruct the **Quaternary Nazca Peneplain (QNP)**
profiles described in the unpublished paper *"Quaternary tectonic shortening and
uplift of the Peruvian forearc due to subduction of the Nazca Ridge: a
quantitative approach"* by Luis Ayala-Carazas, Willem Viveen*, Patrice Baby,
Rodrigo Uribe-Ventura, Steven Binnie, Jorge Sanjurjo-Sánchez and
César Beltrán-Castallon.  The code was originally written
in a large notebook but has been refactored into a small, easy to review
package.

## Structure

- `cvae/dataset.py` – utilities to generate synthetic or Underworld based profiles and helpers to load datasets.
- `cvae/model.py` – definition of the CVAE architecture (encoder, decoder and training model).
- `train.py` – example script that trains the CVAE on a synthetic dataset and saves the resulting model.
- `generate.py` – small utility to load a trained model and sample new profiles.
- `underworld_generation.py` – example Underworld setup to generate uplift simulations.

## Usage

1. Install the requirements (TensorFlow 2.x, NumPy and scikit‑learn):
   ```bash
   pip install tensorflow numpy scikit-learn
   ```
2. Run the training script:
   ```bash
   python train.py
   ```
   A small synthetic dataset will be generated in the `dataset/` directory and
the trained model will be saved under `models/cvae_model`.

3. Generate new profiles from a trained model:
   ```bash
   python generate.py models/cvae_model 0.0 0.1 -n 5
   ```

The architecture follows the specifications in the paper with dense layers of
512 and 256 units in the encoder, a 200‑dimensional latent space and symmetric
decoder.  Conditioning on the **elevation point** and **terrain gradient** at
each profile location allows the model to estimate missing QNP segments along
the cross sections.
