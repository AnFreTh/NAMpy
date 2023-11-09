# Helper funcs for building MLP on transformer output ([cls] token)
import tensorflow as tf


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


def helper_normalization_net(input_list):
    layer1 = tf.keras.layers.Concatenate()
    layer2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    x = layer1(input_list)
    x = layer2(x)

    return tf.keras.Model(inputs=input_list, outputs=x, name="helper_net")
