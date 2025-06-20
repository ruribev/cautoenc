# Conditional Variational Autoencoder for Topographic Profiles

This repository contains a minimal implementation of the Conditional Variational
Autoencoder (CVAE) used to reconstruct the Quaternary Normal Plateau (QNP)
profiles described in the accompanying paper.  The code has been extracted and
refactored from `all.py` into a small, easy to review package.

## Structure

- `cvae/dataset.py` – generation of synthetic topographic profiles and dataset
  utilities.
- `cvae/model.py` – definition of the CVAE architecture (encoder, decoder and
  training model).
- `train.py` – example script that trains the CVAE on a synthetic dataset and
  saves the resulting model.
- `all.py` – original notebook script kept for reference.

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

The architecture follows the specifications in the paper with dense layers of
512 and 256 units in the encoder, a 200‑dimensional latent space and symmetric
decoder.  Conditioning on the start and end elevation of each profile allows the
model to estimate missing QNP segments along the cross sections.
