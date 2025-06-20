from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Input, Lambda
from tensorflow.keras.models import Model

LATENT_DIM = 200


def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim: int = LATENT_DIM) -> Model:
    profile = Input(shape=(300,))
    condition = Input(shape=(2,))
    x = Concatenate()([profile, condition])
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    return Model([profile, condition], [z_mean, z_log_var, z], name="encoder")


def build_decoder(latent_dim: int = LATENT_DIM) -> Model:
    latent_inputs = Input(shape=(latent_dim,))
    condition = Input(shape=(2,))
    x = Concatenate()([latent_inputs, condition])
    x = Dense(256, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    outputs = Dense(300, activation="linear")(x)
    return Model([latent_inputs, condition], outputs, name="decoder")


class CVAE(Model):
    def __init__(self, encoder: Model, decoder: Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        condition, profile = inputs
        z_mean, z_log_var, z = self.encoder([profile, condition])
        reconstructed = self.decoder([z, condition])
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed
