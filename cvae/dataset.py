import os
import uuid
from typing import Tuple

import numpy as np
from .underworld_sample import SAMPLED_XC, SAMPLED_YC


def normalize(data: np.ndarray) -> np.ndarray:
    """Return the normalized version of ``data`` (zero mean, unit variance)."""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def generate_synthetic_profile(
    num_points: int = 300, range_parameter: float = 30.0, sill: float = 50.0
) -> np.ndarray:
    """Generate a synthetic topographic profile using correlated noise."""
    x_values = np.linspace(0, 1000, num_points)

    amplitude1 = np.random.uniform(10, 150)
    frequency1 = np.random.uniform(0.5, 3.0)
    amplitude2 = np.random.uniform(10, 150)
    frequency2 = np.random.uniform(0.5, 3.0)

    y_values = (
        amplitude1 * np.sin(frequency1 * 2 * np.pi * x_values / 1000)
        + amplitude2 * np.cos(frequency2 * 4 * np.pi * x_values / 1000)
    )

    fine_amplitude = np.random.uniform(1, 15)
    fine_frequency = np.random.uniform(1, 15)
    fine_variations = fine_amplitude * np.sin(fine_frequency * 2 * np.pi * x_values / 1000)

    def spherical_variogram(h: np.ndarray, range_a: float, s: float) -> np.ndarray:
        return np.where(h < range_a, s * (1.5 * h / range_a - 0.5 * (h / range_a) ** 3), s)

    distances = np.abs(x_values[:, None] - x_values[None, :])
    variogram_matrix = spherical_variogram(distances, range_parameter, sill)
    covariance_matrix = sill - variogram_matrix

    try:
        chol = np.linalg.cholesky(covariance_matrix)
    except np.linalg.LinAlgError:
        covariance_matrix = np.abs(covariance_matrix)
        chol = np.linalg.cholesky(covariance_matrix)

    noise = chol @ np.random.normal(size=num_points)
    y_values += fine_variations + noise

    return np.column_stack(((x_values / 250) - 2, normalize(y_values)))


def generate_underworld_profile(num_points: int = 300) -> np.ndarray:
    """Return an interpolated profile based on the Underworld sample."""
    x = np.linspace(SAMPLED_XC.min(), SAMPLED_XC.max(), num_points)
    y = np.interp(x, SAMPLED_XC, SAMPLED_YC)
    x_scaled = (x - x.min()) / (x.max() - x.min()) * 4 - 2
    return np.column_stack((x_scaled, normalize(y)))


def generate_profile(num_points: int = 300, source: str = "synthetic") -> np.ndarray:
    """Generate a profile either from Underworld data or the synthetic generator."""
    if source == "underworld":
        return generate_underworld_profile(num_points)
    if source == "synthetic":
        return generate_synthetic_profile(num_points)
    raise ValueError("source must be 'synthetic' or 'underworld'")


def create_dataset(
    num_samples: int,
    save_dir: str,
    source: str = "synthetic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ``num_samples`` profiles of the given ``source`` and save them.

    Each sample is represented by a pair of conditioning variables: the
    elevation point at the start of the profile and the terrain gradient
    calculated from the first and last points.
    """
    os.makedirs(save_dir, exist_ok=True)

    X, Y = [], []
    for _ in range(num_samples):
        profile = generate_profile(source=source)
        elevation_point = profile[0, 1]
        terrain_gradient = (
            profile[-1, 1] - profile[0, 1]
        ) / (profile[-1, 0] - profile[0, 0])
        X.append([elevation_point, terrain_gradient])
        Y.append(profile[:, 1])

    X_arr = np.array(X)
    Y_arr = np.array(Y)

    name = uuid.uuid4().hex
    np.save(os.path.join(save_dir, f"X_{name}.npy"), X_arr)
    np.save(os.path.join(save_dir, f"Y_{name}.npy"), Y_arr)
    return X_arr, Y_arr


def load_dataset(directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all ``X_*.npy`` and ``Y_*.npy`` files from ``directory``."""
    files = os.listdir(directory)
    X_files = sorted(f for f in files if f.startswith("X_"))
    Y_files = sorted(f for f in files if f.startswith("Y_"))

    X_all = np.concatenate([np.load(os.path.join(directory, f)) for f in X_files])
    Y_all = np.concatenate([np.load(os.path.join(directory, f)) for f in Y_files])
    return X_all, Y_all
