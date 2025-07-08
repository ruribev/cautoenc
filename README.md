# Conditional Variational Autoencoder Model for Reconstructing Pre-eroded Topographic Surfaces

A deep learning approach for reconstructing pre-eroded topographic surfaces using Conditional Variational Autoencoders (CVAE). This model enables the interpolation and reconstruction of missing elevation data in eroded topographic profiles, particularly useful for tectonic and erosional analysis.

![image](https://github.com/user-attachments/assets/af7f30b7-387b-49a0-84ae-9b6aef9b1b1b)

## Overview

This repository contains the implementation of a Conditional Variational Autoencoder designed to reconstruct topographic profiles from fragmentary elevation data. The model was developed for geological research focusing on the Quaternary Nazca Peneplain in Peru, enabling quantitative analysis of tectonic deformation over time.

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
