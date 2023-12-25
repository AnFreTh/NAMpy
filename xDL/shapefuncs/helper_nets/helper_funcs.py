# Helper funcs for building MLP on transformer output ([cls] token)
import tensorflow as tf
import keras
import numpy as np


def build_cls_mlp(input_dim, factors, dropout):
    """
    Build a multi-layer perceptron (MLP) for a transformer output.

    Args:
        input_dim (int): Dimension of the input.
        factors (list): List of factors for hidden layer sizes.
        dropout (float): Dropout rate.

    Returns:
        tf.keras.Sequential: The MLP model.
    """
    hidden_units = [input_dim // f for f in factors]
    mlp_layers = []
    for units in hidden_units:
        # mlp_layers.append(tf.keras.layers.BatchNormalization()),
        mlp_layers.append(
            tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            )
        )
        mlp_layers.append(tf.keras.layers.Dropout(dropout))

    return tf.keras.Sequential(mlp_layers)


def build_shape_funcs(
    dropout=0.1, output_dim=1, hidden_units=[128, 128, 64], activation="relu"
):
    """
    Build shape functions using a multi-layer perceptron (MLP).

    Args:
        dropout (float): Dropout rate (default is 0.1).

    Returns:
        tf.keras.Sequential: The MLP model for shape functions.
    """
    hidden_units = [128, 128, 64]

    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(tf.keras.layers.BatchNormalization()),
        mlp_layers.append(tf.keras.layers.Dense(units, activation=activation))
        mlp_layers.append(tf.keras.layers.Dropout(dropout))

    mlp_layers.append(
        tf.keras.layers.Dense(output_dim, activation="linear", use_bias=False)
    )

    return tf.keras.Sequential(mlp_layers)


def get_random_features_initializer(initializer, shape):
    """Returns Initializer object for random features."""
    _SUPPORTED_RBF_KERNEL_TYPES = ["gaussian", "laplacian"]

    def _get_cauchy_samples(loc, scale, shape):
        probs = np.random.uniform(low=0.0, high=1.0, size=shape)
        return loc + scale * np.tan(np.pi * (probs - 0.5))

    random_features_initializer = initializer
    if isinstance(initializer, str):
        if initializer.lower() == "gaussian":
            random_features_initializer = keras.initializers.RandomNormal(stddev=1.0)
        elif initializer.lower() == "laplacian":
            random_features_initializer = keras.initializers.Constant(
                _get_cauchy_samples(loc=0.0, scale=1.0, shape=shape)
            )

        else:
            raise ValueError(
                f'Unsupported `kernel_initializer`: "{initializer}" '
                f"Expected one of: {_SUPPORTED_RBF_KERNEL_TYPES}"
            )
    return random_features_initializer


def get_default_scale(initializer, input_dim):
    if isinstance(initializer, str) and initializer.lower() == "gaussian":
        return np.sqrt(input_dim / 2.0)
    return 1.0
