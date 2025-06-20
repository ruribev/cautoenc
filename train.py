from sklearn.model_selection import train_test_split
import tensorflow as tf

from cvae.dataset import create_dataset
from cvae.model import CVAE, build_decoder, build_encoder


def main() -> None:
    X, Y = create_dataset(num_samples=5000, save_dir="dataset")
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

    encoder = build_encoder()
    decoder = build_decoder()
    model = CVAE(encoder, decoder)
    model.compile(optimizer="adam", loss="mse")

    model.fit(
        [X_train, Y_train],
        Y_train,
        epochs=50,
        batch_size=128,
        validation_data=([X_val, Y_val], Y_val),
        verbose=2,
    )

    model.save("models/cvae_model")


if __name__ == "__main__":
    main()
