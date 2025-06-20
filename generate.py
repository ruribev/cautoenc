import argparse
import numpy as np
from tensorflow.keras.models import load_model

from cvae.model import LATENT_DIM


def simple_moving_average(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    ma = np.zeros_like(data)
    for i in range(window_size - 1, len(data) - window_size + 1):
        ma[i] = np.mean(data[i - window_size + 1 : i + 1])
    for i in range(window_size - 1):
        ma[i] = np.mean(data[: i + 1])
        ma[-(i + 1)] = np.mean(data[-(i + 2) :])
    return ma


def generate(model_path: str, elevation_point: float, terrain_gradient: float, num: int = 1) -> np.ndarray:
    model = load_model(model_path)
    decoder = model.decoder
    profiles = []
    condition = np.array([[elevation_point, terrain_gradient]])
    for _ in range(num):
        z = np.random.normal(size=(1, LATENT_DIM))
        profile = decoder.predict([z, condition], verbose=0)[0]
        profile[0] = elevation_point
        profile = simple_moving_average(profile)
        profiles.append(profile)
    return np.array(profiles)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate profiles using a trained CVAE model")
    parser.add_argument("model", help="Path to saved model")
    parser.add_argument("elevation_point", type=float, help="Elevation point")
    parser.add_argument("terrain_gradient", type=float, help="Terrain gradient")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of profiles")
    args = parser.parse_args()

    profiles = generate(args.model, args.elevation_point, args.terrain_gradient, args.num)
    np.set_printoptions(precision=3, suppress=True)
    print(profiles)


if __name__ == "__main__":
    main()
